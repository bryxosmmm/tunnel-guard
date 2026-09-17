// Native kernels for the Python-side detector glue.
//
// Every function here reproduces, value for value, the NumPy expression it
// replaces: same arithmetic order, same tie-breaking, same selection. Anything
// whose reference implementation lives inside Open3D (plane proposals, normal
// and covariance estimation) is deliberately NOT reimplemented: those outputs
// cannot be reproduced bit for bit and they gate real decisions.
//
// Nothing here defines a threshold, tolerance or algorithm of its own; it only
// evaluates predicates the detector already applies, in the same order.
#define PY_SSIZE_T_CLEAN
#include "native.h"

#include <algorithm>
#include <array>
#include <utility>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

using Key = std::array<int64_t, 3>;

struct KeyHash {
    size_t operator()(const Key& k) const noexcept {
        size_t h = 0;
        for (auto v : k) h ^= std::hash<int64_t>{}(v) + 0x9e3779b9U + (h << 6) + (h >> 2);
        return h;
    }
};

struct ReleaseGIL {
    PyThreadState* state = PyEval_SaveThread();
    ~ReleaseGIL() { PyEval_RestoreThread(state); }
};

class Buffer {
public:
    Buffer(PyObject* object, const char* what) : what_(what) {
        ok_ = PyObject_GetBuffer(object, &view_, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) == 0;
    }
    ~Buffer() { if (ok_) PyBuffer_Release(&view_); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    bool points() const {
        return ok_ && view_.ndim == 2 && view_.shape[1] == 3 && view_.itemsize == sizeof(double)
            && view_.format && std::strcmp(view_.format, "d") == 0;
    }
    bool matrix(Py_ssize_t cols) const {
        return ok_ && view_.ndim == 2 && view_.shape[1] == cols && view_.itemsize == sizeof(double)
            && view_.format && std::strcmp(view_.format, "d") == 0;
    }
    bool vector() const {
        return ok_ && view_.ndim == 1 && view_.itemsize == sizeof(double)
            && view_.format && std::strcmp(view_.format, "d") == 0;
    }
    bool integers() const {
        return ok_ && view_.ndim == 1 && view_.itemsize == sizeof(int64_t)
            && view_.format && (std::strcmp(view_.format, "l") == 0 || std::strcmp(view_.format, "q") == 0);
    }
    bool flags() const {
        return ok_ && view_.ndim == 1 && view_.itemsize == sizeof(bool)
            && view_.format && std::strcmp(view_.format, "?") == 0;
    }
    const double* doubles() const { return static_cast<const double*>(view_.buf); }
    const int64_t* int64s() const { return static_cast<const int64_t*>(view_.buf); }
    const bool* bools() const { return static_cast<const bool*>(view_.buf); }
    Py_ssize_t rows() const { return view_.ndim >= 1 ? view_.shape[0] : 0; }
    Py_ssize_t size() const { return view_.len / static_cast<Py_ssize_t>(view_.itemsize); }

private:
    Py_buffer view_{};
    bool ok_ = false;
    const char* what_;
};

PyObject* bytes_of(const void* data, size_t bytes) {
    return PyBytes_FromStringAndSize(static_cast<const char*>(data), static_cast<Py_ssize_t>(bytes));
}

unsigned worker_count(Py_ssize_t n) {
    if (n < 4096) return 1;
    const unsigned hardware = std::thread::hardware_concurrency();
    return std::max(1u, std::min(hardware ? hardware : 1u, 8u));
}
}  // namespace

// Kept measurement indices for a radial band: |p| within [min, max].
PyObject* range_indices(PyObject*, PyObject* args) {
    PyObject* object;
    double minimum, maximum;
    if (!PyArg_ParseTuple(args, "Odd", &object, &minimum, &maximum)) return nullptr;
    Buffer buffer(object, "points");
    if (!buffer.points()) {
        PyErr_SetString(PyExc_ValueError, "expected contiguous native float64 (N,3)");
        return nullptr;
    }
    const auto* data = buffer.doubles();
    const Py_ssize_t n = buffer.rows();
    std::vector<int64_t> kept;
    try {
        ReleaseGIL released;
        kept.reserve(static_cast<size_t>(n));
        for (Py_ssize_t i = 0; i < n; ++i) {
            const double x = data[3 * i], y = data[3 * i + 1], z = data[3 * i + 2];
            const double radius = std::sqrt(x * x + y * y + z * z);
            // A non-finite coordinate yields a non-finite radius and fails a bound.
            if (radius >= minimum && radius <= maximum) kept.push_back(static_cast<int64_t>(i));
        }
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    return bytes_of(kept.data(), kept.size() * sizeof(int64_t));
}

// Mutual-radius connectivity graph.
//
// The retained undirected pairs are exactly
//     { (i, j) : i < j and d2(i, j) <= min(r_i, r_j)^2 }
// with d2 accumulated as dx*dx + dy*dy + dz*dz, which is the predicate the NumPy
// path applies after its own radius query. That query is only a candidate filter
// and is a strict superset of the predicate, so exact grid enumeration of each
// point's radius ball leaves the pair set identical.
//
// Output is the symmetric CSR that coo_matrix((ones, (edge_i, edge_j))).tocsr()
// produced: sorted columns, unit weights, plus the per-row partner count.
PyObject* mutual_graph(PyObject*, PyObject* args) {
    PyObject* points_object;
    PyObject* radius_object;
    double cell;
    if (!PyArg_ParseTuple(args, "OOd", &points_object, &radius_object, &cell)) return nullptr;
    Buffer points(points_object, "points");
    Buffer radius(radius_object, "radius");
    if (!points.points() || !radius.vector() || radius.size() != points.rows()) {
        PyErr_SetString(PyExc_ValueError, "expected (N,3) float64 points and matching radius vector");
        return nullptr;
    }
    if (!(std::isfinite(cell) && cell > 0)) {
        PyErr_SetString(PyExc_ValueError, "grid cell must be finite and positive");
        return nullptr;
    }
    const double* data = points.doubles();
    const double* reach = radius.doubles();
    const Py_ssize_t n = points.rows();
    std::vector<int64_t> indptr(static_cast<size_t>(n) + 1, 0);
    std::vector<int64_t> indices;
    std::vector<int64_t> degree(static_cast<size_t>(n), 0);
    try {
        ReleaseGIL released;
        std::unordered_map<Key, std::vector<int64_t>, KeyHash> cells;
        cells.reserve(static_cast<size_t>(n) * 2);
        for (Py_ssize_t i = 0; i < n; ++i) {
            Key key{};
            for (int axis = 0; axis < 3; ++axis) {
                const double value = std::floor(data[3 * i + axis] / cell);
                if (!std::isfinite(value) || value < -0x1p62 || value >= 0x1p62)
                    throw std::invalid_argument("nonfinite or out-of-range grid coordinate");
                key[axis] = static_cast<int64_t>(value);
            }
            cells[key].push_back(static_cast<int64_t>(i));
        }
        const auto enumerate = [&](Py_ssize_t start, Py_ssize_t stop, std::vector<int64_t>& sink) {
            for (Py_ssize_t i = start; i < stop; ++i) {
                const double ri = reach[i];
                const double xi = data[3 * i], yi = data[3 * i + 1], zi = data[3 * i + 2];
                Key low{}, high{};
                for (int axis = 0; axis < 3; ++axis) {
                    const double value = data[3 * i + axis];
                    low[axis] = static_cast<int64_t>(std::floor((value - ri) / cell));
                    high[axis] = static_cast<int64_t>(std::floor((value + ri) / cell));
                }
                for (int64_t cx = low[0]; cx <= high[0]; ++cx)
                    for (int64_t cy = low[1]; cy <= high[1]; ++cy)
                        for (int64_t cz = low[2]; cz <= high[2]; ++cz) {
                            const auto found = cells.find(Key{cx, cy, cz});
                            if (found == cells.end()) continue;
                            for (int64_t j : found->second) {
                                if (j <= i) continue;
                                const double dx = xi - data[3 * j];
                                const double dy = yi - data[3 * j + 1];
                                const double dz = zi - data[3 * j + 2];
                                const double other = reach[j];
                                const double bound = ri < other ? ri : other;
                                if (dx * dx + dy * dy + dz * dz <= bound * bound) {
                                    sink.push_back(static_cast<int64_t>(i));
                                    sink.push_back(j);
                                }
                            }
                        }
            }
        };
        std::vector<std::vector<int64_t>> locals(worker_count(n));
        if (locals.size() == 1) {
            enumerate(0, n, locals.front());
        } else {
            std::vector<std::thread> pool;
            const Py_ssize_t chunk = (n + static_cast<Py_ssize_t>(locals.size()) - 1) / static_cast<Py_ssize_t>(locals.size());
            for (size_t worker = 0; worker < locals.size(); ++worker) {
                const Py_ssize_t start = static_cast<Py_ssize_t>(worker) * chunk;
                const Py_ssize_t stop = std::min(n, start + chunk);
                if (start >= stop) break;
                pool.emplace_back([&enumerate, start, stop, &sink = locals[worker]] { enumerate(start, stop, sink); });
            }
            for (auto& thread : pool) thread.join();
        }
        std::vector<int64_t> pairs;
        size_t total = 0;
        for (const auto& local : locals) total += local.size();
        pairs.reserve(total);
        for (const auto& local : locals) pairs.insert(pairs.end(), local.begin(), local.end());
        for (size_t index = 0; index < pairs.size(); index += 2) {
            ++degree[static_cast<size_t>(pairs[index])];
            ++degree[static_cast<size_t>(pairs[index + 1])];
        }
        for (Py_ssize_t i = 0; i < n; ++i) indptr[i + 1] = indptr[i] + degree[static_cast<size_t>(i)];
        indices.resize(static_cast<size_t>(indptr[n]));
        std::vector<int64_t> cursor(indptr.begin(), indptr.end() - 1);
        for (size_t index = 0; index < pairs.size(); index += 2) {
            const int64_t first = pairs[index], second = pairs[index + 1];
            indices[static_cast<size_t>(cursor[static_cast<size_t>(first)]++)] = second;
            indices[static_cast<size_t>(cursor[static_cast<size_t>(second)]++)] = first;
        }
        for (Py_ssize_t i = 0; i < n; ++i) {
            auto begin = indices.begin() + indptr[i];
            std::sort(begin, begin + degree[static_cast<size_t>(i)]);
        }
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    catch (const std::exception& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
    std::vector<uint8_t> weight(indices.size(), 1);
    PyObject* payload = PyTuple_New(4);
    PyTuple_SET_ITEM(payload, 0, bytes_of(indptr.data(), indptr.size() * sizeof(int64_t)));
    PyTuple_SET_ITEM(payload, 1, bytes_of(indices.data(), indices.size() * sizeof(int64_t)));
    PyTuple_SET_ITEM(payload, 2, bytes_of(weight.data(), weight.size()));
    PyTuple_SET_ITEM(payload, 3, bytes_of(degree.data(), degree.size() * sizeof(int64_t)));
    return payload;
}

// Points inside one patch's longitudinal window and removal band.
PyObject* patch_candidates(PyObject*, PyObject* args) {
    PyObject* object;
    PyObject* plane_object;
    double low, high, distance;
    if (!PyArg_ParseTuple(args, "OOddd", &object, &plane_object, &low, &high, &distance)) return nullptr;
    Buffer points(object, "leveled points");
    Buffer plane(plane_object, "plane");
    if (!points.points() || !plane.vector() || plane.size() != 4) {
        PyErr_SetString(PyExc_ValueError, "expected (N,3) points and a 4 element plane");
        return nullptr;
    }
    const double* data = points.doubles();
    const double* model = plane.doubles();
    const Py_ssize_t n = points.rows();
    std::vector<int64_t> ids;
    try {
        ReleaseGIL released;
        for (Py_ssize_t i = 0; i < n; ++i) {
            const double x = data[3 * i];
            if (!(x >= low && x <= high)) continue;
            const double signed_distance = x * model[0] + data[3 * i + 1] * model[1]
                + data[3 * i + 2] * model[2] + model[3];
            if (std::abs(signed_distance) <= distance) ids.push_back(static_cast<int64_t>(i));
        }
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    return bytes_of(ids.data(), ids.size() * sizeof(int64_t));
}

// Subset of the given points landing inside any of a patch's observed strips.
PyObject* strip_inside(PyObject*, PyObject* args) {
    PyObject* object;
    PyObject* ids_object;
    PyObject* strips_object;
    double margin;
    long transverse;
    if (!PyArg_ParseTuple(args, "OOOdl", &object, &ids_object, &strips_object, &margin, &transverse))
        return nullptr;
    Buffer points(object, "leveled points");
    Buffer ids(ids_object, "ids");
    Buffer strips(strips_object, "strips");
    if (!points.points() || !ids.integers() || !strips.matrix(4)) {
        PyErr_SetString(PyExc_ValueError, "expected points (N,3), int64 ids and (M,4) strips");
        return nullptr;
    }
    const double* data = points.doubles();
    const int64_t* index = ids.int64s();
    const double* strip = strips.doubles();
    const Py_ssize_t rows = ids.size(), count = strips.rows();
    std::vector<int64_t> inside;
    try {
        ReleaseGIL released;
        for (Py_ssize_t row = 0; row < rows; ++row) {
            const int64_t point = index[row];
            const double x = data[3 * point];
            const double lateral = data[3 * point + transverse];
            for (Py_ssize_t s = 0; s < count; ++s) {
                const double* box = strip + 4 * s;
                if (x >= box[0] - margin && x <= box[1] + margin
                        && lateral >= box[2] - margin && lateral <= box[3] + margin) {
                    inside.push_back(point);
                    break;
                }
            }
        }
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    return bytes_of(inside.data(), inside.size() * sizeof(int64_t));
}

// Samples whose distance band and normal alignment protect an attachment edge.
PyObject* protrusion_ids(PyObject*, PyObject* args) {
    PyObject* sample_object;
    PyObject* normals_object;
    PyObject* reliable_object;
    PyObject* plane_object;
    double depth, radius, alignment;
    if (!PyArg_ParseTuple(args, "OOOOddd", &sample_object, &normals_object, &reliable_object,
                          &plane_object, &depth, &radius, &alignment)) return nullptr;
    Buffer sample(sample_object, "sample");
    Buffer normals(normals_object, "normals");
    Buffer reliable(reliable_object, "reliable");
    Buffer plane(plane_object, "plane");
    if (!sample.points() || !normals.points() || !reliable.flags() || !plane.vector() || plane.size() != 4) {
        PyErr_SetString(PyExc_ValueError, "expected sample/normals (N,3), bool reliability and a 4 element plane");
        return nullptr;
    }
    if (normals.rows() != sample.rows() || reliable.size() != sample.rows()) {
        PyErr_SetString(PyExc_ValueError, "sample, normals and reliability must agree in length");
        return nullptr;
    }
    const double* points = sample.doubles();
    const double* directions = normals.doubles();
    const bool* usable = reliable.bools();
    const double* model = plane.doubles();
    const Py_ssize_t n = sample.rows();
    std::vector<int64_t> ids;
    try {
        ReleaseGIL released;
        for (Py_ssize_t i = 0; i < n; ++i) {
            if (!usable[i]) continue;
            const double raw = points[3 * i] * model[0] + points[3 * i + 1] * model[1]
                + points[3 * i + 2] * model[2] + model[3];
            const double distance = std::abs(raw);
            if (!(distance >= depth && distance <= radius)) continue;
            const double cosine = std::abs(directions[3 * i] * model[0] + directions[3 * i + 1] * model[1]
                                           + directions[3 * i + 2] * model[2]);
            if (cosine < alignment) ids.push_back(static_cast<int64_t>(i));
        }
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    return bytes_of(ids.data(), ids.size() * sizeof(int64_t));
}

// Per-query flag: some target point lies within the Euclidean protection radius.
PyObject* within_radius(PyObject*, PyObject* args) {
    PyObject* query_object;
    PyObject* target_object;
    double radius;
    if (!PyArg_ParseTuple(args, "OOd", &query_object, &target_object, &radius)) return nullptr;
    Buffer query(query_object, "query");
    Buffer target(target_object, "targets");
    if (!query.points() || !target.points()) {
        PyErr_SetString(PyExc_ValueError, "expected (N,3) float64 query and target points");
        return nullptr;
    }
    const double* q = query.doubles();
    const double* t = target.doubles();
    const Py_ssize_t queries = query.rows(), targets = target.rows();
    std::vector<uint8_t> hit(static_cast<size_t>(queries), 0);
    try {
        ReleaseGIL released;
        for (Py_ssize_t i = 0; i < queries; ++i) {
            const double xi = q[3 * i], yi = q[3 * i + 1], zi = q[3 * i + 2];
            for (Py_ssize_t j = 0; j < targets; ++j) {
                const double dx = xi - t[3 * j];
                const double dy = yi - t[3 * j + 1];
                const double dz = zi - t[3 * j + 2];
                // Same Euclidean form the tree query used: sqrt of the squared sum.
                if (std::sqrt(dx * dx + dy * dy + dz * dz) <= radius) {
                    hit[static_cast<size_t>(i)] = 1;
                    break;
                }
            }
        }
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    return bytes_of(hit.data(), hit.size());
}


// --- geometry classification -------------------------------------------------
//
// Mirrors TrackGeometry.ground/path/classify exactly: same interpolation form,
// same comparison order, same clipping. The reference implementation is the
// NumPy path in tunnel_guard/geometry.py and the two are compared array for
// array by scripts/check_kernels.py.

namespace {

// np.interp semantics: linear inside the anchor range, clamped outside, NaN in,
// NaN out. Slope uses the same expression NumPy evaluates.
void interp_into(const double* x, Py_ssize_t n, const double* xp, const double* fp,
                 Py_ssize_t m, double* out) {
    if (m == 0) return;
    if (m == 1) {
        for (Py_ssize_t i = 0; i < n; ++i) out[i] = fp[0];
        return;
    }
    for (Py_ssize_t i = 0; i < n; ++i) {
        const double value = x[i];
        if (std::isnan(value)) { out[i] = value; continue; }
        if (value <= xp[0]) { out[i] = fp[0]; continue; }
        if (value >= xp[m - 1]) { out[i] = fp[m - 1]; continue; }
        Py_ssize_t low = 0, high = m - 1;
        while (high - low > 1) {
            const Py_ssize_t middle = (low + high) / 2;
            if (xp[middle] <= value) low = middle; else high = middle;
        }
        const double slope = (fp[low + 1] - fp[low]) / (xp[low + 1] - xp[low]);
        // NumPy's compiled interp evaluates this product-sum with one rounding,
        // unlike its elementwise ufuncs; std::fma reproduces that exactly.
        out[i] = std::fma(slope, value - xp[low], fp[low]);
    }
}

// Nearest |x - anchor| without materialising the dense difference.
void nearest_anchor_into(const double* x, Py_ssize_t n, const double* anchor_x, Py_ssize_t m, double* out) {
    for (Py_ssize_t i = 0; i < n; ++i) {
        const double value = x[i];
        Py_ssize_t position = 0;
        if (!std::isnan(value)) {
            Py_ssize_t low = 0, high = m;
            while (low < high) {
                const Py_ssize_t middle = (low + high) / 2;
                if (anchor_x[middle] < value) low = middle + 1; else high = middle;
            }
            position = low;
        } else {
            position = m;
        }
        if (position < 1) position = 1;
        if (position > m - 1) position = m - 1;
        const double left = std::abs(value - anchor_x[position - 1]);
        const double right = std::abs(anchor_x[position] - value);
        out[i] = left < right ? left : right;
    }
}

// np.searchsorted(side="right") - 1, clipped into the segment table.
Py_ssize_t segment_index(const double* edges, Py_ssize_t m, double value) {
    Py_ssize_t low = 0, high = m;
    while (low < high) {
        const Py_ssize_t middle = (low + high) / 2;
        if (edges[middle] <= value) low = middle + 1; else high = middle;
    }
    Py_ssize_t index = low - 1;
    if (index < 0) index = 0;
    if (index > m - 1) index = m - 1;
    return index;
}
}  // namespace

// Returns core, context, height, observed, nominal_overlap, boundary.
PyObject* classify_geometry(PyObject*, PyObject* args) {
    PyObject *points_object, *plane_object, *ground_object, *rail_object, *envelope_object;
    double rail_head, ground_max_uncertainty, path_max_uncertainty, ground_max_extrapolation,
        path_max_extrapolation, rail_half_width, rail_vertical_margin, min_running_height,
        cluster_context_margin, segmentation_half_width, envelope_margin, rail_max_heading;
    if (!PyArg_ParseTuple(args, "OOOOO" "dddddddddddd", &points_object, &plane_object, &ground_object,
                          &rail_object, &envelope_object, &rail_head, &ground_max_uncertainty,
                          &path_max_uncertainty, &ground_max_extrapolation, &path_max_extrapolation,
                          &rail_half_width, &rail_vertical_margin, &min_running_height,
                          &cluster_context_margin, &segmentation_half_width, &envelope_margin,
                          &rail_max_heading)) return nullptr;
    Buffer points(points_object, "points");
    Buffer plane(plane_object, "plane");
    Buffer ground(ground_object, "ground anchors");
    Buffer rail(rail_object, "rail anchors");
    Buffer envelope(envelope_object, "envelope");
    if (!points.points() || !plane.vector() || plane.size() != 3 || !ground.matrix(3)
            || !rail.matrix(4) || !envelope.matrix(4)) {
        PyErr_SetString(PyExc_ValueError,
                        "expected points (N,3), plane (3), ground anchors (G,3), rail anchors (R,4), envelope (S,4)");
        return nullptr;
    }
    const Py_ssize_t n = points.rows();
    const Py_ssize_t g = ground.rows(), r = rail.rows(), s = envelope.rows();
    const double* data = points.doubles();
    const double* model = plane.doubles();
    if (g < 1 || r < 2 || s < 1) {
        PyErr_SetString(PyExc_ValueError, "classification requires ground anchors, two rail anchors and an envelope");
        return nullptr;
    }
    std::vector<double> height(static_cast<size_t>(n)), running(static_cast<size_t>(n)),
        lateral(static_cast<size_t>(n)), width(static_cast<size_t>(n)), center(static_cast<size_t>(n)),
        gauge(static_cast<size_t>(n)), ground_uncertainty(static_cast<size_t>(n)),
        path_uncertainty(static_cast<size_t>(n)), shift(static_cast<size_t>(n)),
        rail_uncertainty(static_cast<size_t>(n)), nearest(static_cast<size_t>(n));
    std::vector<uint8_t> core(static_cast<size_t>(n)), context(static_cast<size_t>(n)),
        observed(static_cast<size_t>(n)), overlap(static_cast<size_t>(n)), boundary(static_cast<size_t>(n));
    std::vector<double> xs(static_cast<size_t>(n));
    try {
        ReleaseGIL released;
        for (Py_ssize_t i = 0; i < n; ++i) xs[static_cast<size_t>(i)] = data[3 * i];
        // ground(): shift, uncertainty, z
        const double* ground_x = ground.doubles();
        const double* ground_shift = ground.doubles() + 1;
        const double* ground_spread = ground.doubles() + 2;
        std::vector<double> anchor_x(static_cast<size_t>(g)), anchor_shift(static_cast<size_t>(g)),
            anchor_spread(static_cast<size_t>(g));
        for (Py_ssize_t i = 0; i < g; ++i) {
            anchor_x[static_cast<size_t>(i)] = ground_x[3 * i];
            anchor_shift[static_cast<size_t>(i)] = ground_shift[3 * i];
            anchor_spread[static_cast<size_t>(i)] = ground_spread[3 * i];
        }
        interp_into(xs.data(), n, anchor_x.data(), anchor_shift.data(), g, shift.data());
        interp_into(xs.data(), n, anchor_x.data(), anchor_spread.data(), g, ground_uncertainty.data());
        nearest_anchor_into(xs.data(), n, anchor_x.data(), g, nearest.data());
        for (Py_ssize_t i = 0; i < n; ++i) {
            const size_t index = static_cast<size_t>(i);
            ground_uncertainty[index] += nearest[index] * 0.008;
            if (nearest[index] > ground_max_extrapolation) ground_uncertainty[index] = INFINITY;
            const double z = data[3 * i] * model[0] + data[3 * i + 1] * model[1] + model[2] + shift[index];
            height[index] = data[3 * i + 2] - z;
        }
        // path(): center, gauge, uncertainty
        const double* rail_x = rail.doubles();
        const double* rail_center = rail.doubles() + 1;
        const double* rail_gauge = rail.doubles() + 2;
        std::vector<double> anchors_x(static_cast<size_t>(r)), anchors_center(static_cast<size_t>(r)),
            anchors_gauge(static_cast<size_t>(r));
        for (Py_ssize_t i = 0; i < r; ++i) {
            anchors_x[static_cast<size_t>(i)] = rail_x[4 * i];
            anchors_center[static_cast<size_t>(i)] = rail_center[4 * i];
            anchors_gauge[static_cast<size_t>(i)] = rail_gauge[4 * i];
        }
        interp_into(xs.data(), n, anchors_x.data(), anchors_center.data(), r, center.data());
        interp_into(xs.data(), n, anchors_x.data(), anchors_gauge.data(), r, gauge.data());
        nearest_anchor_into(xs.data(), n, anchors_x.data(), r, rail_uncertainty.data());
        {
            const std::array<std::pair<int, int>, 2> edges = {{{0, 1}, {-1, -2}}};
            for (const auto& edge : edges) {
                const int here = edge.first, other = edge.second;
                const int index_here = here < 0 ? static_cast<int>(r) + here : here;
                const int index_other = other < 0 ? static_cast<int>(r) + other : other;
                double slope = (anchors_center[index_here] - anchors_center[index_other])
                    / (anchors_x[index_here] - anchors_x[index_other]);
                slope = std::max(-rail_max_heading, std::min(rail_max_heading, slope));
                const bool below = here == 0;
                for (Py_ssize_t i = 0; i < n; ++i) {
                    const double value = xs[static_cast<size_t>(i)];
                    if ((below && value < anchors_x[0]) || (!below && value > anchors_x[r - 1])) {
                        center[static_cast<size_t>(i)] = anchors_center[index_here]
                            + slope * (value - anchors_x[index_here]);
                    }
                }
            }
        }
        for (Py_ssize_t i = 0; i < n; ++i) {
            const size_t index = static_cast<size_t>(i);
            const double reach = rail_uncertainty[index];
            // NumPy forms the square first and scales it afterwards.
            path_uncertainty[index] = 0.06 + 0.008 * reach + 0.0003 * (reach * reach);
            if (reach > path_max_extrapolation) path_uncertainty[index] = INFINITY;
        }
        // classify()
        const double* segments = envelope.doubles();
        const double low_edge = segments[0];
        const double high_edge = segments[4 * (s - 1) + 1];
        // Segment lower edges are the first column of the contiguous table.
        std::vector<double> segment_edges(static_cast<size_t>(s));
        for (Py_ssize_t i = 0; i < s; ++i) segment_edges[static_cast<size_t>(i)] = segments[4 * i];
        const double slope_plane = model[1];
        // NumPy sums the squared components first and adds one afterwards.
        const double normal_scale = std::sqrt(1.0 + (model[0] * model[0] + model[1] * model[1]));
        const double lateral_scale = std::sqrt(1.0 + slope_plane * slope_plane);
        for (Py_ssize_t i = 0; i < n; ++i) {
            const size_t index = static_cast<size_t>(i);
            const double relative = height[index] - rail_head;
            const double running_height = relative / normal_scale;
            running[index] = running_height;
            const double dy = data[3 * i + 1] - center[index];
            const double lateral_value = (dy + slope_plane * (relative + slope_plane * dy)) / lateral_scale;
            lateral[index] = lateral_value;
            const Py_ssize_t segment = segment_index(segment_edges.data(), s, running_height);
            const double* box = segments + 4 * segment;
            double fraction = (running_height - box[0]) / (box[1] - box[0]);
            fraction = std::max(0.0, std::min(1.0, fraction));
            const double half_width = box[2] + fraction * (box[3] - box[2]) + envelope_margin;
            width[index] = half_width;
            const bool ground_ok = ground_uncertainty[index] <= ground_max_uncertainty;
            const bool path_ok = path_uncertainty[index] <= path_max_uncertainty;
            observed[index] = (ground_ok && path_ok) ? 1 : 0;
            const bool vertical = (running_height >= low_edge + ground_uncertainty[index])
                && (running_height <= high_edge);
            const double lateral_uncertainty = path_uncertainty[index] * lateral_scale;
            const bool on_rail = observed[index]
                && (std::abs(std::abs(lateral_value) - gauge[index] / 2.0) < rail_half_width + path_uncertainty[index])
                && (running_height <= rail_vertical_margin);
            const bool supported = ground_ok && vertical && !on_rail;
            const bool nominal = supported && observed[index] && (std::abs(lateral_value) <= half_width);
            core[index] = (nominal && (std::abs(lateral_value) + lateral_uncertainty <= half_width)) ? 1 : 0;
            boundary[index] = (supported && observed[index] && !core[index]
                               && (std::abs(lateral_value) - lateral_uncertainty <= half_width)) ? 1 : 0;
            const bool segmentation = (running_height >= min_running_height)
                && (running_height <= high_edge + cluster_context_margin);
            context[index] = (segmentation && !on_rail
                              && (std::abs(lateral_value) <= segmentation_half_width)) ? 1 : 0;
            overlap[index] = ((running_height >= low_edge) && (running_height <= high_edge) && !on_rail
                              && (std::abs(lateral_value) <= half_width)) ? 1 : 0;
        }
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    PyObject* payload = PyTuple_New(6);
    PyTuple_SET_ITEM(payload, 0, bytes_of(core.data(), core.size()));
    PyTuple_SET_ITEM(payload, 1, bytes_of(context.data(), context.size()));
    PyTuple_SET_ITEM(payload, 2, bytes_of(height.data(), height.size() * sizeof(double)));
    PyTuple_SET_ITEM(payload, 3, bytes_of(observed.data(), observed.size()));
    PyTuple_SET_ITEM(payload, 4, bytes_of(overlap.data(), overlap.size()));
    PyTuple_SET_ITEM(payload, 5, bytes_of(boundary.data(), boundary.size()));
    return payload;
}


// Track-bed reference: interpolated shift, nearest-anchor extrapolation penalty
// and the fitted plane, exactly as TrackGeometry.ground evaluates them.
PyObject* ground_values(PyObject*, PyObject* args) {
    PyObject *points_object, *plane_object, *anchor_object;
    double maximum;
    if (!PyArg_ParseTuple(args, "OOOd", &points_object, &plane_object, &anchor_object, &maximum)) return nullptr;
    Buffer points(points_object, "points");
    Buffer plane(plane_object, "plane");
    Buffer anchor(anchor_object, "ground anchors");
    if (!points.points() || !plane.vector() || plane.size() != 3 || !anchor.matrix(3) || anchor.rows() < 1) {
        PyErr_SetString(PyExc_ValueError, "expected points (N,3), plane (3) and ground anchors (G,3)");
        return nullptr;
    }
    const Py_ssize_t n = points.rows(), g = anchor.rows();
    const double* data = points.doubles();
    const double* model = plane.doubles();
    const double* anchors = anchor.doubles();
    std::vector<double> x(static_cast<size_t>(n)), z(static_cast<size_t>(n)),
        uncertainty(static_cast<size_t>(n));
    try {
        ReleaseGIL released;
        for (Py_ssize_t i = 0; i < n; ++i) x[static_cast<size_t>(i)] = data[3 * i];
        std::vector<double> anchor_x(static_cast<size_t>(g)), anchor_shift(static_cast<size_t>(g)),
            anchor_spread(static_cast<size_t>(g)), out_shift(static_cast<size_t>(n));
        for (Py_ssize_t i = 0; i < g; ++i) {
            anchor_x[static_cast<size_t>(i)] = anchors[3 * i];
            anchor_shift[static_cast<size_t>(i)] = anchors[3 * i + 1];
            anchor_spread[static_cast<size_t>(i)] = anchors[3 * i + 2];
        }
        interp_into(x.data(), n, anchor_x.data(), anchor_shift.data(), g, out_shift.data());
        interp_into(x.data(), n, anchor_x.data(), anchor_spread.data(), g, uncertainty.data());
        std::vector<double> reach(static_cast<size_t>(n));
        nearest_anchor_into(x.data(), n, anchor_x.data(), g, reach.data());
        for (Py_ssize_t i = 0; i < n; ++i) {
            const size_t index = static_cast<size_t>(i);
            uncertainty[index] = uncertainty[index] + reach[index] * 0.008;
            if (reach[index] > maximum) uncertainty[index] = INFINITY;
            z[index] = data[3 * i] * model[0] + data[3 * i + 1] * model[1] + model[2] + out_shift[index];
        }
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    PyObject* payload = PyTuple_New(2);
    PyTuple_SET_ITEM(payload, 0, bytes_of(z.data(), z.size() * sizeof(double)));
    PyTuple_SET_ITEM(payload, 1, bytes_of(uncertainty.data(), uncertainty.size() * sizeof(double)));
    return payload;
}

static PyMethodDef methods[] = {
    {"voxel_indices", voxel_indices, METH_VARARGS,
     "First measurement indices, lexicographic voxel order."},
    {"voxel_count", voxel_count, METH_VARARGS,
     "Number of distinct voxels for the same keys."},
    {"range_indices", range_indices, METH_VARARGS,
     "Measurement indices with |p| inside a radial band; non-finite fails a bound."},
    {"mutual_graph", mutual_graph, METH_VARARGS,
     "Mutual-radius symmetric CSR: indptr, indices, unit weights, per-row degree."},
    {"patch_candidates", patch_candidates, METH_VARARGS,
     "Indices inside one patch's longitudinal window and removal band."},
    {"strip_inside", strip_inside, METH_VARARGS,
     "Given points landing inside any of a patch's observed strips."},
    {"protrusion_ids", protrusion_ids, METH_VARARGS,
     "Samples whose distance band and normal alignment protect an attachment edge."},
    {"ground_values", ground_values, METH_VARARGS,
     "Track-bed reference height and extrapolation uncertainty for the current scan."},
    {"classify_geometry", classify_geometry, METH_VARARGS,
     "Envelope, rail-relative and ground support classification of the current scan."},
    {"within_radius", within_radius, METH_VARARGS,
     "Per-query flag: a target lies within the Euclidean protection radius."},
    {nullptr, nullptr, 0, nullptr}
};

static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native", nullptr, -1, methods};

PyMODINIT_FUNC PyInit__native() { return PyModule_Create(&module); }
