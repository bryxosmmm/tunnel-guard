// Native kernels for the detector's per-frame glue.
//
// Two rules govern this file.
//
// 1. Numerical identity. Every function reproduces, value for value, the NumPy
//    expression it replaces: same arithmetic order, same tie-breaking, same
//    selection. NumPy evaluates np.interp with one fused rounding and its
//    elementwise ufuncs with separate roundings; the kernels match both. Nothing
//    here defines a threshold or changes a decision.
//
// 2. No per-call allocation in steady state. Every scratch buffer lives in a
//    thread-local arena that grows to the largest frame seen and is then reused,
//    so a frame costs copies, not malloc/free or page faults. Callers receive
//    explicit copies because the arena is overwritten by the next call.
//
// Deliberately absent: Open3D plane proposals and normal/covariance estimation.
// Their outputs cannot be reproduced bit for bit (adaptive stopping depends on
// scheduling; the fast normal path is not the covariance eigenvector), and they
// gate real decisions.
#define PY_SSIZE_T_CLEAN
#include "native.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

Arena& arena() {
    static thread_local Arena instance;
    return instance;
}

void build_cells(const double* points, Py_ssize_t n, double size) {
    Arena& scratch = arena();
    KeyTable& table = scratch.table;
    table.reset(static_cast<size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        Key key{};
        if (!voxel_key_of(points + 3 * i, size, key))
            throw std::invalid_argument("nonfinite or out-of-range grid coordinate");
        bool inserted = false;
        const size_t slot = table.slot_of(key, inserted);
        ++table.values[slot];
    }
    const size_t capacity = table.keys.size();
    scratch.cell_start.assign(capacity + 1, 0);
    size_t placed = 0;
    for (size_t slot = 0; slot < capacity; ++slot) {
        scratch.cell_start[slot] = static_cast<int64_t>(placed);
        if (table.used[slot]) placed += static_cast<size_t>(table.values[slot]);
    }
    scratch.cell_start[capacity] = static_cast<int64_t>(placed);
    scratch.cell_items.resize(placed);
    scratch.cell_cursor.assign(capacity, 0);
    for (Py_ssize_t i = 0; i < n; ++i) {
        Key key{};
        voxel_key_of(points + 3 * i, size, key);
        bool inserted = false;
        const size_t slot = table.slot_of(key, inserted);
        scratch.cell_items[static_cast<size_t>(scratch.cell_start[slot]) + static_cast<size_t>(scratch.cell_cursor[slot]++)] = i;
    }
}

namespace {

struct ReleaseGIL {
    PyThreadState* state = PyEval_SaveThread();
    ~ReleaseGIL() { PyEval_RestoreThread(state); }
};

// Reusable scratch. Capacity is retained across frames; resizing within
// capacity never reallocates, which is the point of the arena.
struct Workspace {
    std::vector<double> d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10;
    std::vector<int64_t> i0, i1, i2, i3, i4;
    std::vector<uint8_t> b0, b1, b2, b3, b4;
    std::vector<std::pair<Key, int64_t>> ordered;
    std::vector<std::vector<int64_t>> sinks;
};

thread_local Workspace workspace;

class Buffer {
public:
    Buffer(PyObject* object, int flags = PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) {
        ok_ = PyObject_GetBuffer(object, &view_, flags) == 0;
    }
    ~Buffer() { if (ok_) PyBuffer_Release(&view_); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    bool ok() const { return ok_; }
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
    bool writable_flags() const {
        return ok_ && view_.ndim == 1 && view_.itemsize == sizeof(bool) && (view_.readonly == 0)
            && view_.format && std::strcmp(view_.format, "?") == 0;
    }
    const double* doubles() const { return static_cast<const double*>(view_.buf); }
    const int64_t* int64s() const { return static_cast<const int64_t*>(view_.buf); }
    const bool* bools() const { return static_cast<const bool*>(view_.buf); }
    bool* writable_bools() const { return static_cast<bool*>(view_.buf); }
    Py_ssize_t rows() const { return view_.ndim >= 1 ? view_.shape[0] : 0; }
    Py_ssize_t size() const { return view_.len / static_cast<Py_ssize_t>(view_.itemsize); }

private:
    Py_buffer view_{};
    bool ok_ = false;
};

PyObject* bytes_of(const void* data, size_t bytes) {
    return PyBytes_FromStringAndSize(static_cast<const char*>(data), static_cast<Py_ssize_t>(bytes));
}

unsigned worker_count(Py_ssize_t n) {
    if (n < 4096) return 1;
    const unsigned hardware = std::thread::hardware_concurrency();
    return std::max(1u, std::min(hardware ? hardware : 1u, 8u));
}

// Eigen-decomposition of a symmetric 3x3 matrix: eigenvalues ascending and the
// eigenvector of the smallest one.
//
// Jacobi rotations with a fixed sweep count: no convergence test, no branch on
// the spectrum, so the result is a deterministic function of the input and stays
// accurate when eigenvalues are repeated (where the closed-form trigonometric
// solution loses several digits to cancellation).
void symmetric_eigen3(double xx, double xy, double xz, double yy, double yz, double zz,
                      double eigenvalues[3], double normal[3]) {
    double a[3][3] = {{xx, xy, xz}, {xy, yy, yz}, {xz, yz, zz}};
    double v[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    for (int sweep = 0; sweep < 8; ++sweep) {
        for (int pair = 0; pair < 3; ++pair) {
            const int i = pair == 0 ? 0 : (pair == 1 ? 0 : 1);
            const int j = pair == 0 ? 1 : (pair == 1 ? 2 : 2);
            if (a[i][j] == 0.0) continue;
            const double theta = (a[j][j] - a[i][i]) / (2.0 * a[i][j]);
            const double sign = theta >= 0.0 ? 1.0 : -1.0;
            const double t = sign / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
            const double c = 1.0 / std::sqrt(t * t + 1.0);
            const double s = t * c;
            for (int k = 0; k < 3; ++k) {
                const double aik = a[i][k], ajk = a[j][k];
                a[i][k] = c * aik - s * ajk;
                a[j][k] = s * aik + c * ajk;
            }
            for (int k = 0; k < 3; ++k) {
                const double aki = a[k][i], akj = a[k][j];
                a[k][i] = c * aki - s * akj;
                a[k][j] = s * aki + c * akj;
            }
            for (int k = 0; k < 3; ++k) {
                const double vki = v[k][i], vkj = v[k][j];
                v[k][i] = c * vki - s * vkj;
                v[k][j] = s * vki + c * vkj;
            }
        }
    }
    int smallest = 0, middle = 1, largest = 2;
    const double diagonal[3] = {a[0][0], a[1][1], a[2][2]};
    if (diagonal[middle] < diagonal[smallest]) { const int swap = smallest; smallest = middle; middle = swap; }
    if (diagonal[largest] < diagonal[middle]) { const int swap = middle; middle = largest; largest = swap; }
    if (diagonal[middle] < diagonal[smallest]) { const int swap = smallest; smallest = middle; middle = swap; }
    eigenvalues[0] = diagonal[smallest];
    eigenvalues[1] = diagonal[middle];
    eigenvalues[2] = diagonal[largest];
    for (int k = 0; k < 3; ++k) normal[k] = v[k][smallest];
}

// Squared distance between a stored point and a query position.
double distance_squared(const double* points, int64_t index, double x, double y, double z) {
    const double dx = points[3 * index] - x;
    const double dy = points[3 * index + 1] - y;
    const double dz = points[3 * index + 2] - z;
    return dx * dx + dy * dy + dz * dz;
}

bool accept(PyObject* payload, std::vector<PyObject*>& owned, const char* name) {
    if (payload == nullptr) {
        PyErr_Format(PyExc_RuntimeError, "failed to build %s", name);
        return false;
    }
    owned.push_back(payload);
    return true;
}

// np.interp semantics: clamped outside the anchor range, NaN in, NaN out, and
// one fused rounding for the product-sum exactly as NumPy's compiled version.
void interp_into(const double* x, Py_ssize_t n, const double* xp, const double* fp,
                 Py_ssize_t m, double* out) {
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
        out[i] = std::fma(slope, value - xp[low], fp[low]);
    }
}

void nearest_anchor_into(const double* x, Py_ssize_t n, const double* anchor_x, Py_ssize_t m, double* out) {
    for (Py_ssize_t i = 0; i < n; ++i) {
        const double value = x[i];
        Py_ssize_t position = m;
        if (!std::isnan(value)) {
            Py_ssize_t low = 0, high = m;
            while (low < high) {
                const Py_ssize_t middle = (low + high) / 2;
                if (anchor_x[middle] < value) low = middle + 1; else high = middle;
            }
            position = low;
        }
        if (position < 1) position = 1;
        if (position > m - 1) position = m - 1;
        const double left = std::abs(value - anchor_x[position - 1]);
        const double right = std::abs(anchor_x[position] - value);
        out[i] = left < right ? left : right;
    }
}

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

// Indices inside one plane's distance band within an already windowed slice.
void band_candidates(const double* points, const int64_t* ids, Py_ssize_t count, const double* plane,
                     double distance, std::vector<int64_t>& out) {
    for (Py_ssize_t row = 0; row < count; ++row) {
        const int64_t point = ids[row];
        const double signed_distance = points[3 * point] * plane[0] + points[3 * point + 1] * plane[1]
            + points[3 * point + 2] * plane[2] + plane[3];
        if (std::abs(signed_distance) <= distance) out.push_back(point);
    }
}

void protrusion_samples(const double* sample, const double* normals, const bool* reliable, Py_ssize_t n,
                        const double* plane, double depth, double radius, double alignment,
                        std::vector<int64_t>& out) {
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (!reliable[i]) continue;
        const double raw = sample[3 * i] * plane[0] + sample[3 * i + 1] * plane[1]
            + sample[3 * i + 2] * plane[2] + plane[3];
        const double distance = std::abs(raw);
        if (!(distance >= depth && distance <= radius)) continue;
        const double cosine = std::abs(normals[3 * i] * plane[0] + normals[3 * i + 1] * plane[1]
                                       + normals[3 * i + 2] * plane[2]);
        if (cosine < alignment) out.push_back(static_cast<int64_t>(i));
    }
}

// Whether each query point has a target within the protection radius. Same
// Euclidean form as the tree query it replaces.
void clear_of_targets(const double* query, const int64_t* ids, Py_ssize_t count, const double* targets,
                      Py_ssize_t target_count, double radius, std::vector<int64_t>& out) {
    for (Py_ssize_t row = 0; row < count; ++row) {
        const int64_t point = ids[row];
        const double x = query[3 * point], y = query[3 * point + 1], z = query[3 * point + 2];
        bool hit = false;
        for (Py_ssize_t j = 0; j < target_count; ++j) {
            const double dx = x - targets[3 * j];
            const double dy = y - targets[3 * j + 1];
            const double dz = z - targets[3 * j + 2];
            if (std::sqrt(dx * dx + dy * dy + dz * dz) <= radius) { hit = true; break; }
        }
        if (!hit) out.push_back(point);
    }
}

void strips_of(const double* points, const int64_t* ids, Py_ssize_t count, const double* strips,
               Py_ssize_t strip_count, double margin, long transverse, std::vector<int64_t>& out) {
    for (Py_ssize_t row = 0; row < count; ++row) {
        const int64_t point = ids[row];
        const double x = points[3 * point];
        const double lateral = points[3 * point + transverse];
        for (Py_ssize_t s = 0; s < strip_count; ++s) {
            const double* box = strips + 4 * s;
            if (x >= box[0] - margin && x <= box[1] + margin
                    && lateral >= box[2] - margin && lateral <= box[3] + margin) {
                out.push_back(point);
                break;
            }
        }
    }
}
}  // namespace

// Kept measurement indices for a radial band: |p| within [min, max].
PyObject* range_indices(PyObject*, PyObject* args) {
    PyObject* object;
    double minimum, maximum;
    if (!PyArg_ParseTuple(args, "Odd", &object, &minimum, &maximum)) return nullptr;
    Buffer points(object);
    if (!points.points()) {
        PyErr_SetString(PyExc_ValueError, "expected contiguous native float64 (N,3)");
        return nullptr;
    }
    const auto* data = points.doubles();
    const Py_ssize_t n = points.rows();
    auto& kept = workspace.i0;
    try {
        ReleaseGIL released;
        kept.clear();
        for (Py_ssize_t i = 0; i < n; ++i) {
            const double x = data[3 * i], y = data[3 * i + 1], z = data[3 * i + 2];
            const double radius = std::sqrt(x * x + y * y + z * z);
            // A non-finite coordinate yields a non-finite radius and fails a bound.
            if (radius >= minimum && radius <= maximum) kept.push_back(static_cast<int64_t>(i));
        }
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    return bytes_of(kept.data(), kept.size() * sizeof(int64_t));
}

// Distinct voxel count per stack of evidence points, one call for all objects.
// The key derivation is shared with voxel_indices, so the count equals
// len(voxel_representatives(points, size)) for each stack.
PyObject* voxel_counts(PyObject*, PyObject* args) {
    PyObject* sequence;
    double size;
    if (!PyArg_ParseTuple(args, "Od", &sequence, &size)) return nullptr;
    if (!(std::isfinite(size) && size > 0)) {
        PyErr_SetString(PyExc_ValueError, "voxel size must be finite and positive");
        return nullptr;
    }
    PyObject* iterator = PyObject_GetIter(sequence);
    if (iterator == nullptr) return nullptr;
    auto& counts = workspace.i1;
    try {
        counts.clear();
        PyObject* item = nullptr;
        while ((item = PyIter_Next(iterator)) != nullptr) {
            Buffer points(item);
            if (!points.points()) {
                Py_DECREF(item);
                Py_DECREF(iterator);
                PyErr_SetString(PyExc_ValueError, "expected contiguous native float64 (N,3) per entry");
                return nullptr;
            }
            const double* data = points.doubles();
            const Py_ssize_t n = points.rows();
            arena().table.reset(static_cast<size_t>(n));
            Py_ssize_t distinct = 0;
            for (Py_ssize_t i = 0; i < n; ++i) {
                Key key{};
                if (!voxel_key_of(data + 3 * i, size, key))
                    throw std::invalid_argument("nonfinite or out-of-range voxel coordinate");
                bool inserted = false;
                arena().table.slot_of(key, inserted);
                if (inserted) ++distinct;
            }
            counts.push_back(static_cast<int64_t>(distinct));
            Py_DECREF(item);
        }
        Py_DECREF(iterator);
        if (PyErr_Occurred()) return nullptr;
    } catch (const std::bad_alloc&) {
        Py_DECREF(iterator);
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        Py_DECREF(iterator);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
    return bytes_of(counts.data(), counts.size() * sizeof(int64_t));
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
    Buffer points(points_object);
    Buffer radius(radius_object);
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
    auto& indptr = workspace.i0;
    auto& indices = workspace.i1;
    auto& degree = workspace.i2;
    try {
        ReleaseGIL released;
        build_cells(data, n, cell);
        Arena& scratch = arena();
        KeyTable& table = scratch.table;
        // Pair enumeration, threaded over points with arena-backed sinks.
        auto& sinks = workspace.sinks;
        const unsigned workers = worker_count(n);
        sinks.resize(workers);
        for (auto& sink : sinks) sink.clear();
        const auto collect = [&](Py_ssize_t start, Py_ssize_t stop, std::vector<int64_t>& sink) {
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
                            const size_t slot = table.find(Key{cx, cy, cz});
                            if (slot == static_cast<size_t>(-1)) continue;
                            const int64_t* begin = scratch.cell_items.data() + scratch.cell_start[slot];
                            const int64_t* end = scratch.cell_items.data() + scratch.cell_start[slot + 1];
                            for (const int64_t* item = begin; item != end; ++item) {
                                const int64_t j = *item;
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
        if (workers <= 1) {
            collect(0, n, sinks.front());
        } else {
            std::vector<std::thread> pool;
            const Py_ssize_t chunk = (n + static_cast<Py_ssize_t>(workers) - 1) / static_cast<Py_ssize_t>(workers);
            for (unsigned worker = 0; worker < workers; ++worker) {
                const Py_ssize_t start = static_cast<Py_ssize_t>(worker) * chunk;
                const Py_ssize_t stop = std::min(n, start + chunk);
                if (start >= stop) break;
                pool.emplace_back([&collect, start, stop, &sink = sinks[worker]] { collect(start, stop, sink); });
            }
            for (auto& thread : pool) thread.join();
        }
        size_t total = 0;
        for (const auto& sink : sinks) total += sink.size();
        indptr.assign(static_cast<size_t>(n) + 1, 0);
        degree.assign(static_cast<size_t>(n), 0);
        for (const auto& sink : sinks)
            for (size_t index = 0; index < sink.size(); index += 2) {
                ++degree[static_cast<size_t>(sink[index])];
                ++degree[static_cast<size_t>(sink[index + 1])];
            }
        for (Py_ssize_t i = 0; i < n; ++i) indptr[static_cast<size_t>(i) + 1] = indptr[static_cast<size_t>(i)] + degree[static_cast<size_t>(i)];
        indices.assign(static_cast<size_t>(indptr[static_cast<size_t>(n)]), 0);
        auto& cursor = workspace.i3;
        cursor.assign(indptr.begin(), indptr.end() - 1);
        for (const auto& sink : sinks)
            for (size_t index = 0; index < sink.size(); index += 2) {
                const int64_t first = sink[index], second = sink[index + 1];
                indices[static_cast<size_t>(cursor[static_cast<size_t>(first)]++)] = second;
                indices[static_cast<size_t>(cursor[static_cast<size_t>(second)]++)] = first;
            }
        for (Py_ssize_t i = 0; i < n; ++i) {
            auto begin = indices.begin() + indptr[static_cast<size_t>(i)];
            std::sort(begin, begin + degree[static_cast<size_t>(i)]);
        }
        (void)total;
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    catch (const std::exception& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
    std::vector<uint8_t>& weight = workspace.b0;
    weight.assign(indices.size(), 1);
    PyObject* payload = PyTuple_New(4);
    if (payload == nullptr) return nullptr;
    std::vector<PyObject*> owned;
    if (!accept(bytes_of(indptr.data(), indptr.size() * sizeof(int64_t)), owned, "indptr")
            || !accept(bytes_of(indices.data(), indices.size() * sizeof(int64_t)), owned, "indices")
            || !accept(bytes_of(weight.data(), weight.size()), owned, "weights")
            || !accept(bytes_of(degree.data(), degree.size() * sizeof(int64_t)), owned, "degree")) {
        Py_DECREF(payload);
        for (PyObject* object : owned) Py_DECREF(object);
        return nullptr;
    }
    for (size_t index = 0; index < owned.size(); ++index) PyTuple_SET_ITEM(payload, index, owned[index]);
    return payload;
}

// Points inside one patch's longitudinal window and removal band.
PyObject* patch_candidates(PyObject*, PyObject* args) {
    PyObject* object;
    PyObject* plane_object;
    double low, high, distance;
    if (!PyArg_ParseTuple(args, "OOddd", &object, &plane_object, &low, &high, &distance)) return nullptr;
    Buffer points(object);
    Buffer plane(plane_object);
    if (!points.points() || !plane.vector() || plane.size() != 4) {
        PyErr_SetString(PyExc_ValueError, "expected (N,3) points and a 4 element plane");
        return nullptr;
    }
    const double* data = points.doubles();
    const Py_ssize_t n = points.rows();
    auto& window = workspace.i0;
    auto& ids = workspace.i1;
    try {
        ReleaseGIL released;
        window.clear();
        for (Py_ssize_t i = 0; i < n; ++i) {
            const double x = data[3 * i];
            if (x >= low && x <= high) window.push_back(static_cast<int64_t>(i));
        }
        ids.clear();
        band_candidates(data, window.data(), static_cast<Py_ssize_t>(window.size()), plane.doubles(), distance, ids);
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
    Buffer points(object);
    Buffer ids(ids_object);
    Buffer strips(strips_object);
    if (!points.points() || !ids.integers() || !strips.matrix(4)) {
        PyErr_SetString(PyExc_ValueError, "expected points (N,3), int64 ids and (M,4) strips");
        return nullptr;
    }
    auto& inside = workspace.i2;
    try {
        inside.clear();
        strips_of(points.doubles(), ids.int64s(), ids.size(), strips.doubles(), strips.rows(), margin,
                  transverse, inside);
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
    Buffer sample(sample_object);
    Buffer normals(normals_object);
    Buffer reliable(reliable_object);
    Buffer plane(plane_object);
    if (!sample.points() || !normals.points() || !reliable.flags() || !plane.vector() || plane.size() != 4) {
        PyErr_SetString(PyExc_ValueError, "expected sample/normals (N,3), bool reliability and a 4 element plane");
        return nullptr;
    }
    if (normals.rows() != sample.rows() || reliable.size() != sample.rows()) {
        PyErr_SetString(PyExc_ValueError, "sample, normals and reliability must agree in length");
        return nullptr;
    }
    auto& ids = workspace.i3;
    try {
        ids.clear();
        protrusion_samples(sample.doubles(), normals.doubles(), reliable.bools(), sample.rows(),
                           plane.doubles(), depth, radius, alignment, ids);
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    return bytes_of(ids.data(), ids.size() * sizeof(int64_t));
}

// Per-query flag: some target point lies within the Euclidean protection radius.
PyObject* within_radius(PyObject*, PyObject* args) {
    PyObject* query_object;
    PyObject* target_object;
    double radius;
    if (!PyArg_ParseTuple(args, "OOd", &query_object, &target_object, &radius)) return nullptr;
    Buffer query(query_object);
    Buffer target(target_object);
    if (!query.points() || !target.points()) {
        PyErr_SetString(PyExc_ValueError, "expected (N,3) float64 query and target points");
        return nullptr;
    }
    const Py_ssize_t queries = query.rows();
    auto& ids = workspace.i4;
    auto& hit = workspace.b1;
    try {
        ids.clear();
        for (Py_ssize_t i = 0; i < queries; ++i) ids.push_back(static_cast<int64_t>(i));
        hit.assign(static_cast<size_t>(queries), 0);
        const double* q = query.doubles();
        const double* t = target.doubles();
        const Py_ssize_t targets = target.rows();
        for (Py_ssize_t i = 0; i < queries; ++i) {
            const double x = q[3 * i], y = q[3 * i + 1], z = q[3 * i + 2];
            for (Py_ssize_t j = 0; j < targets; ++j) {
                const double dx = x - t[3 * j];
                const double dy = y - t[3 * j + 1];
                const double dz = z - t[3 * j + 2];
                if (std::sqrt(dx * dx + dy * dy + dz * dz) <= radius) { hit[static_cast<size_t>(i)] = 1; break; }
            }
        }
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    return bytes_of(hit.data(), hit.size());
}

// Track-bed reference: interpolated shift, nearest-anchor extrapolation penalty
// and the fitted plane, exactly as TrackGeometry.ground evaluates them.
PyObject* ground_values(PyObject*, PyObject* args) {
    PyObject *points_object, *plane_object, *anchor_object;
    double maximum;
    if (!PyArg_ParseTuple(args, "OOOd", &points_object, &plane_object, &anchor_object, &maximum)) return nullptr;
    Buffer points(points_object);
    Buffer plane(plane_object);
    Buffer anchor(anchor_object);
    if (!points.points() || !plane.vector() || plane.size() != 3 || !anchor.matrix(3) || anchor.rows() < 1) {
        PyErr_SetString(PyExc_ValueError, "expected points (N,3), plane (3) and ground anchors (G,3)");
        return nullptr;
    }
    const Py_ssize_t n = points.rows(), g = anchor.rows();
    const double* data = points.doubles();
    const double* model = plane.doubles();
    const double* anchors = anchor.doubles();
    auto& x = workspace.d0;
    auto& z = workspace.d1;
    auto& uncertainty = workspace.d2;
    auto& anchor_x = workspace.d3;
    auto& anchor_shift = workspace.d4;
    auto& anchor_spread = workspace.d5;
    try {
        ReleaseGIL released;
        x.resize(static_cast<size_t>(n));
        z.resize(static_cast<size_t>(n));
        uncertainty.resize(static_cast<size_t>(n));
        anchor_x.resize(static_cast<size_t>(g));
        anchor_shift.resize(static_cast<size_t>(g));
        anchor_spread.resize(static_cast<size_t>(g));
        for (Py_ssize_t i = 0; i < n; ++i) x[static_cast<size_t>(i)] = data[3 * i];
        for (Py_ssize_t i = 0; i < g; ++i) {
            anchor_x[static_cast<size_t>(i)] = anchors[3 * i];
            anchor_shift[static_cast<size_t>(i)] = anchors[3 * i + 1];
            anchor_spread[static_cast<size_t>(i)] = anchors[3 * i + 2];
        }
        auto& shift = workspace.d6;
        auto& reach = workspace.d7;
        shift.resize(static_cast<size_t>(n));
        reach.resize(static_cast<size_t>(n));
        interp_into(x.data(), n, anchor_x.data(), anchor_shift.data(), g, shift.data());
        interp_into(x.data(), n, anchor_x.data(), anchor_spread.data(), g, uncertainty.data());
        nearest_anchor_into(x.data(), n, anchor_x.data(), g, reach.data());
        for (Py_ssize_t i = 0; i < n; ++i) {
            const size_t index = static_cast<size_t>(i);
            uncertainty[index] = uncertainty[index] + reach[index] * 0.008;
            if (reach[index] > maximum) uncertainty[index] = INFINITY;
            z[index] = data[3 * i] * model[0] + data[3 * i + 1] * model[1] + model[2] + shift[index];
        }
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    PyObject* payload = PyTuple_New(2);
    if (payload == nullptr) return nullptr;
    std::vector<PyObject*> owned;
    if (!accept(bytes_of(z.data(), z.size() * sizeof(double)), owned, "ground height")
            || !accept(bytes_of(uncertainty.data(), uncertainty.size() * sizeof(double)), owned, "ground uncertainty")) {
        Py_DECREF(payload);
        for (PyObject* object : owned) Py_DECREF(object);
        return nullptr;
    }
    for (size_t index = 0; index < owned.size(); ++index) PyTuple_SET_ITEM(payload, index, owned[index]);
    return payload;
}

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
    Buffer points(points_object);
    Buffer plane(plane_object);
    Buffer ground(ground_object);
    Buffer rail(rail_object);
    Buffer envelope(envelope_object);
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
    auto& height = workspace.d0;
    auto& running = workspace.d1;
    auto& lateral = workspace.d2;
    auto& width = workspace.d3;
    auto& center = workspace.d4;
    auto& gauge = workspace.d5;
    auto& ground_uncertainty = workspace.d6;
    auto& path_uncertainty = workspace.d7;
    auto& scratch_x = workspace.d8;
    auto& scratch_reach = workspace.d9;
    auto& core = workspace.b0;
    auto& context = workspace.b1;
    auto& observed = workspace.b2;
    auto& overlap = workspace.b3;
    auto& boundary = workspace.b4;
    try {
        ReleaseGIL released;
        const auto size = static_cast<size_t>(n);
        for (auto* vector : {&height, &running, &lateral, &width, &center, &gauge,
                             &ground_uncertainty, &path_uncertainty, &scratch_x, &scratch_reach})
            vector->resize(size);
        for (auto* vector : {&core, &context, &observed, &overlap, &boundary}) vector->resize(size);
        for (Py_ssize_t i = 0; i < n; ++i) scratch_x[static_cast<size_t>(i)] = data[3 * i];
        const double* ground_x = ground.doubles();
        const double* ground_shift = ground.doubles() + 1;
        const double* ground_spread = ground.doubles() + 2;
        std::vector<double> anchors_x(static_cast<size_t>(g));
        std::vector<double> anchors_shift(static_cast<size_t>(g));
        std::vector<double> anchors_spread(static_cast<size_t>(g));
        for (Py_ssize_t i = 0; i < g; ++i) {
            anchors_x[static_cast<size_t>(i)] = ground_x[3 * i];
            anchors_shift[static_cast<size_t>(i)] = ground_shift[3 * i];
            anchors_spread[static_cast<size_t>(i)] = ground_spread[3 * i];
        }
        interp_into(scratch_x.data(), n, anchors_x.data(), anchors_shift.data(), g, center.data());
        interp_into(scratch_x.data(), n, anchors_x.data(), anchors_spread.data(), g, ground_uncertainty.data());
        nearest_anchor_into(scratch_x.data(), n, anchors_x.data(), g, scratch_reach.data());
        for (Py_ssize_t i = 0; i < n; ++i) {
            const size_t index = static_cast<size_t>(i);
            ground_uncertainty[index] = ground_uncertainty[index] + scratch_reach[index] * 0.008;
            if (scratch_reach[index] > ground_max_extrapolation) ground_uncertainty[index] = INFINITY;
            const double z = data[3 * i] * model[0] + data[3 * i + 1] * model[1] + model[2] + center[index];
            height[index] = data[3 * i + 2] - z;
        }
        const double* rail_x = rail.doubles();
        const double* rail_center = rail.doubles() + 1;
        const double* rail_gauge = rail.doubles() + 2;
        std::vector<double> rail_anchor_x(static_cast<size_t>(r));
        std::vector<double> rail_anchor_center(static_cast<size_t>(r));
        std::vector<double> rail_anchor_gauge(static_cast<size_t>(r));
        for (Py_ssize_t i = 0; i < r; ++i) {
            rail_anchor_x[static_cast<size_t>(i)] = rail_x[4 * i];
            rail_anchor_center[static_cast<size_t>(i)] = rail_center[4 * i];
            rail_anchor_gauge[static_cast<size_t>(i)] = rail_gauge[4 * i];
        }
        interp_into(scratch_x.data(), n, rail_anchor_x.data(), rail_anchor_center.data(), r, lateral.data());
        interp_into(scratch_x.data(), n, rail_anchor_x.data(), rail_anchor_gauge.data(), r, gauge.data());
        nearest_anchor_into(scratch_x.data(), n, rail_anchor_x.data(), r, scratch_reach.data());
        {
            const std::array<std::pair<int, int>, 2> edges = {{{0, 1}, {-1, -2}}};
            for (const auto& edge : edges) {
                const int index_here = edge.first < 0 ? static_cast<int>(r) + edge.first : edge.first;
                const int index_other = edge.second < 0 ? static_cast<int>(r) + edge.second : edge.second;
                double slope = (rail_anchor_center[index_here] - rail_anchor_center[index_other])
                    / (rail_anchor_x[index_here] - rail_anchor_x[index_other]);
                slope = std::max(-rail_max_heading, std::min(rail_max_heading, slope));
                const bool below = edge.first == 0;
                for (Py_ssize_t i = 0; i < n; ++i) {
                    const double value = scratch_x[static_cast<size_t>(i)];
                    if ((below && value < rail_anchor_x[0]) || (!below && value > rail_anchor_x[r - 1])) {
                        lateral[static_cast<size_t>(i)] = rail_anchor_center[index_here]
                            + slope * (value - rail_anchor_x[index_here]);
                    }
                }
            }
        }
        for (Py_ssize_t i = 0; i < n; ++i) {
            const size_t index = static_cast<size_t>(i);
            const double reach = scratch_reach[index];
            // NumPy forms the square first and scales it afterwards.
            path_uncertainty[index] = 0.06 + 0.008 * reach + 0.0003 * (reach * reach);
            if (reach > path_max_extrapolation) path_uncertainty[index] = INFINITY;
        }
        for (Py_ssize_t i = 0; i < n; ++i) center[static_cast<size_t>(i)] = lateral[static_cast<size_t>(i)];
        const double* segments = envelope.doubles();
        const double low_edge = segments[0];
        const double high_edge = segments[4 * (s - 1) + 1];
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
    if (payload == nullptr) return nullptr;
    std::vector<PyObject*> owned;
    if (!accept(bytes_of(core.data(), core.size()), owned, "core")
            || !accept(bytes_of(context.data(), context.size()), owned, "context")
            || !accept(bytes_of(height.data(), height.size() * sizeof(double)), owned, "height")
            || !accept(bytes_of(observed.data(), observed.size()), owned, "observed")
            || !accept(bytes_of(overlap.data(), overlap.size()), owned, "nominal_overlap")
            || !accept(bytes_of(boundary.data(), boundary.size()), owned, "boundary")) {
        Py_DECREF(payload);
        for (PyObject* object : owned) Py_DECREF(object);
        return nullptr;
    }
    for (size_t index = 0; index < owned.size(); ++index) PyTuple_SET_ITEM(payload, index, owned[index]);
    return payload;
}

// Every patch's candidates in one call: the concatenated candidate indices, the
// per-patch slice offsets, and their unique union for the neighbour query.
PyObject* mask_candidates(PyObject*, PyObject* args) {
    PyObject *points_object, *planes_object, *bounds_object;
    double distance;
    if (!PyArg_ParseTuple(args, "OOOd", &points_object, &planes_object, &bounds_object, &distance)) return nullptr;
    Buffer points(points_object);
    Buffer planes(planes_object);
    Buffer bounds(bounds_object);
    if (!points.points() || !planes.matrix(4) || !bounds.matrix(2) || planes.rows() != bounds.rows()) {
        PyErr_SetString(PyExc_ValueError, "expected points (N,3), planes (P,4) and bounds (P,2)");
        return nullptr;
    }
    const double* data = points.doubles();
    const Py_ssize_t n = points.rows(), patches = planes.rows();
    const double* models = planes.doubles();
    const double* limits = bounds.doubles();
    auto& window = workspace.i0;
    auto& candidates = workspace.i1;
    auto& offsets = workspace.i2;
    auto& union_ids = workspace.i3;
    try {
        ReleaseGIL released;
        candidates.clear();
        offsets.assign(static_cast<size_t>(patches) + 1, 0);
        for (Py_ssize_t patch = 0; patch < patches; ++patch) {
            const double low = limits[2 * patch], high = limits[2 * patch + 1];
            window.clear();
            for (Py_ssize_t i = 0; i < n; ++i) {
                const double x = data[3 * i];
                if (x >= low && x <= high) window.push_back(static_cast<int64_t>(i));
            }
            band_candidates(data, window.data(), static_cast<Py_ssize_t>(window.size()), models + 4 * patch,
                            distance, candidates);
            offsets[static_cast<size_t>(patch) + 1] = static_cast<int64_t>(candidates.size());
        }
        union_ids.assign(candidates.begin(), candidates.end());
        std::sort(union_ids.begin(), union_ids.end());
        union_ids.erase(std::unique(union_ids.begin(), union_ids.end()), union_ids.end());
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    PyObject* payload = PyTuple_New(3);
    if (payload == nullptr) return nullptr;
    std::vector<PyObject*> owned;
    if (!accept(bytes_of(union_ids.data(), union_ids.size() * sizeof(int64_t)), owned, "union")
            || !accept(bytes_of(candidates.data(), candidates.size() * sizeof(int64_t)), owned, "candidates")
            || !accept(bytes_of(offsets.data(), offsets.size() * sizeof(int64_t)), owned, "candidate offsets")) {
        Py_DECREF(payload);
        for (PyObject* object : owned) Py_DECREF(object);
        return nullptr;
    }
    for (size_t index = 0; index < owned.size(); ++index) PyTuple_SET_ITEM(payload, index, owned[index]);
    return payload;
}

// The whole per-patch mask pass in one call: candidate filtering against the
// running mask, attachment-edge protection, strip membership, and the in-place
// background mark. `reliable` and `aligned` arrive already expanded to
// candidate order, exactly as the NumPy loop indexes them.
PyObject* mask_apply(PyObject*, PyObject* args) {
    PyObject *points_object, *protected_object, *background_object, *planes_object, *transverse_object,
        *strips_object, *strip_offsets_object, *candidates_object, *candidate_offsets_object,
        *reliable_object, *aligned_object, *sample_object, *normals_object, *normal_reliable_object;
    double margin, depth, radius, alignment;
    if (!PyArg_ParseTuple(args, "OOOOOOOOOOOOOO" "dddd", &points_object, &protected_object, &background_object,
                          &planes_object, &transverse_object, &strips_object, &strip_offsets_object,
                          &candidates_object, &candidate_offsets_object, &reliable_object, &aligned_object,
                          &sample_object, &normals_object, &normal_reliable_object,
                          &margin, &depth, &radius, &alignment)) return nullptr;
    Buffer points(points_object);
    Buffer protected_ids(protected_object);
    Buffer background(background_object, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS | PyBUF_WRITABLE);
    Buffer planes(planes_object);
    Buffer transverse(transverse_object);
    Buffer strips(strips_object);
    Buffer strip_offsets(strip_offsets_object);
    Buffer candidates(candidates_object);
    Buffer candidate_offsets(candidate_offsets_object);
    Buffer reliable(reliable_object);
    Buffer aligned(aligned_object);
    Buffer sample(sample_object);
    Buffer normals(normals_object);
    Buffer normal_reliable(normal_reliable_object);
    if (!points.points() || !protected_ids.flags() || !background.writable_flags() || !planes.matrix(4)
            || !transverse.integers() || !strips.matrix(4) || !strip_offsets.integers() || !candidates.integers()
            || !candidate_offsets.integers() || !reliable.flags() || !aligned.points()
            || !sample.points() || !normals.points() || !normal_reliable.flags()) {
        PyErr_SetString(PyExc_ValueError, "mask_apply received inconsistent buffers");
        return nullptr;
    }
    const Py_ssize_t patches = planes.rows();
    if (transverse.size() != patches || strip_offsets.size() != patches + 1
            || candidate_offsets.size() != patches + 1 || reliable.size() != candidates.size()
            || aligned.rows() != candidates.size()) {
        PyErr_SetString(PyExc_ValueError, "mask_apply expects per-patch offsets and per-candidate arrays");
        return nullptr;
    }
    const double* data = points.doubles();
    const double* models = planes.doubles();
    const bool* protected_flags = protected_ids.bools();
    bool* background_flags = background.writable_bools();
    const int64_t* transverse_values = transverse.int64s();
    const double* strip_boxes = strips.doubles();
    const int64_t* strip_bounds = strip_offsets.int64s();
    const int64_t* candidate_ids = candidates.int64s();
    const int64_t* candidate_bounds = candidate_offsets.int64s();
    const bool* reliable_flags = reliable.bools();
    const double* aligned_normals = aligned.doubles();
    const double* sample_points = sample.doubles();
    const double* normal_vectors = normals.doubles();
    const bool* sample_reliable = normal_reliable.bools();
    try {
        ReleaseGIL released;
        auto& near_ids = workspace.i0;
        auto& protrusions = workspace.i1;
        auto& cleared = workspace.i2;
        auto& inside = workspace.i3;
        for (Py_ssize_t patch = 0; patch < patches; ++patch) {
            const double* model = models + 4 * patch;
            const int64_t begin = candidate_bounds[patch], end = candidate_bounds[patch + 1];
            near_ids.clear();
            for (int64_t row = begin; row < end; ++row) {
                const int64_t point = candidate_ids[row];
                if (protected_flags[point] || background_flags[point]) continue;
                if (reliable_flags[row]) {
                    const double cosine = std::abs(aligned_normals[3 * row] * model[0]
                                                   + aligned_normals[3 * row + 1] * model[1]
                                                   + aligned_normals[3 * row + 2] * model[2]);
                    if (cosine < alignment) continue;
                }
                near_ids.push_back(point);
            }
            if (near_ids.empty()) continue;
            protrusions.clear();
            protrusion_samples(sample_points, normal_vectors, sample_reliable, sample.rows(), model,
                               depth, radius, alignment, protrusions);
            if (!protrusions.empty()) {
                // Distances are taken to the protruding samples only, exactly as
                // the reference gathers sample[protrusion] before querying.
                Arena& scratch = arena();
                auto& targets = scratch.d10;
                targets.resize(protrusions.size() * 3);
                for (size_t index = 0; index < protrusions.size(); ++index) {
                    const int64_t source = protrusions[index];
                    targets[3 * index] = sample_points[3 * source];
                    targets[3 * index + 1] = sample_points[3 * source + 1];
                    targets[3 * index + 2] = sample_points[3 * source + 2];
                }
                cleared.clear();
                clear_of_targets(data, near_ids.data(), static_cast<Py_ssize_t>(near_ids.size()),
                                 targets.data(), static_cast<Py_ssize_t>(protrusions.size()), radius, cleared);
                near_ids.swap(cleared);
            }
            if (near_ids.empty()) continue;
            inside.clear();
            const int64_t strip_begin = strip_bounds[patch], strip_end = strip_bounds[patch + 1];
            strips_of(data, near_ids.data(), static_cast<Py_ssize_t>(near_ids.size()),
                      strip_boxes + 4 * strip_begin, static_cast<Py_ssize_t>(strip_end - strip_begin),
                      margin, static_cast<long>(transverse_values[patch]), inside);
            for (int64_t point : inside) background_flags[point] = true;
        }
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    Py_RETURN_NONE;
}

// Longitudinal-window crop and voxel reduction in one call.
//
// The reduction itself is the same dedupe the standalone kernel performs, but
// the representatives are ordered without a global sort: the leading key axis is
// counted into its own buckets, and only the (few) representatives sharing one
// column are compared with each other. Since the axis is the most significant
// field, concatenating the columns in ascending order is exactly the
// lexicographic key order numpy.unique(axis=0) produced, and the retained
// measurement per voxel is still the first one in input order.
//
// The crop is applied while keys are computed, so the cropped copy of the cloud
// never exists.
PyObject* select_crop_voxels(PyObject*, PyObject* args) {
    PyObject* object;
    double min_forward, half_width, size;
    if (!PyArg_ParseTuple(args, "Oddd", &object, &min_forward, &half_width, &size)) return nullptr;
    if (!(std::isfinite(size) && size > 0)) {
        PyErr_SetString(PyExc_ValueError, "voxel size must be finite and positive");
        return nullptr;
    }
    Buffer buffer(object);
    if (!buffer.points()) {
        PyErr_SetString(PyExc_ValueError, "expected contiguous native float64 (N,3)");
        return nullptr;
    }
    const double* data = buffer.doubles();
    const Py_ssize_t n = buffer.rows();
    Arena& scratch = arena();
    auto& indices = scratch.i0;
    try {
        // The GIL is released only around the computation; the returned copy is
        // built afterwards, when the guard has been destroyed.
        {
            ReleaseGIL released;
            KeyTable& table = scratch.table;
            table.reset(static_cast<size_t>(n));
            int64_t lowest = 0, highest = 0;
            bool any = false;
            for (Py_ssize_t i = 0; i < n; ++i) {
                const double x = data[3 * i], y = data[3 * i + 1];
                if (!(x >= min_forward && std::abs(y) < half_width)) continue;
                Key key{};
                if (!voxel_key_of(data + 3 * i, size, key))
                    throw std::invalid_argument("nonfinite or out-of-range voxel coordinate");
                bool inserted = false;
                const size_t slot = table.slot_of(key, inserted);
                if (inserted) table.values[slot] = static_cast<int64_t>(i);
                if (!any || key[0] < lowest) lowest = key[0];
                if (!any || key[0] > highest) highest = key[0];
                any = true;
            }
            indices.clear();
            if (any) {
                const int64_t columns = highest - lowest + 1;
                auto& offsets = scratch.i1;
                auto& cursor = scratch.i2;
                auto& column_slots = scratch.i3;
                offsets.assign(static_cast<size_t>(columns) + 1, 0);
                for (size_t slot = 0; slot < table.keys.size(); ++slot)
                    if (table.used[slot]) ++offsets[static_cast<size_t>(table.keys[slot][0] - lowest) + 1];
                for (int64_t column = 0; column < columns; ++column) offsets[static_cast<size_t>(column) + 1] += offsets[static_cast<size_t>(column)];
                cursor.assign(offsets.begin(), offsets.end() - 1);
                column_slots.resize(table.count);
                for (size_t slot = 0; slot < table.keys.size(); ++slot)
                    if (table.used[slot])
                        column_slots[static_cast<size_t>(cursor[static_cast<size_t>(table.keys[slot][0] - lowest)]++)] = static_cast<int64_t>(slot);
                indices.reserve(table.count);
                const Key* keys = table.keys.data();
                for (int64_t column = 0; column < columns; ++column) {
                    const int64_t begin = offsets[static_cast<size_t>(column)];
                    const int64_t end = offsets[static_cast<size_t>(column) + 1];
                    if (end == begin) continue;
                    auto first = column_slots.begin() + begin;
                    auto last = column_slots.begin() + end;
                    // Only the representatives inside one column are ordered here.
                    std::sort(first, last, [keys](int64_t left, int64_t right) {
                        const Key& a = keys[left];
                        const Key& b = keys[right];
                        if (a[1] != b[1]) return a[1] < b[1];
                        return a[2] < b[2];
                    });
                    for (auto entry = first; entry != last; ++entry)
                        indices.push_back(table.values[static_cast<size_t>(*entry)]);
                }
            }
        }
        return bytes_of(indices.data(), indices.size() * sizeof(int64_t));
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    catch (const std::exception& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}

// Per-component statistics of the cluster cloud.
//
// One grouping pass by label (ascending, members ascending) followed by a single
// scan per component replaces a Python loop whose time went into thousands of
// small NumPy calls. Every value is the same expression the reference loop
// evaluates: min/max and their difference, element counts, and the first member
// achieving the minimum forward coordinate. Rejection happens in the same order,
// and the object dictionaries stay in Python, where they cost under a
// millisecond.
//
// Payload order: labels, offsets, members, reason codes, path relation, distance
// method, relation reason, bbox min, bbox max, centre, extent, height span,
// witness, distance, distance support points, cluster nearest x, supported
// nearest x, unresolved nearest x, support voxels, density-core voxels,
// in-envelope voxels, boundary voxels, immediate, interior density-core voxels,
// interior height span, intersection immediate.
PyObject* cluster_components(PyObject*, PyObject* args) {
    PyObject *cloud_object, *labels_object, *core_object, *boundary_object, *dense_object,
        *heights_object, *uncertain_object;
    int weak_min_voxels, immediate_min_voxels, envelope_support_mode;
    double min_extent, immediate_min_height;
    if (!PyArg_ParseTuple(args, "OOOOOOOiiddi", &cloud_object, &labels_object, &core_object,
                          &boundary_object, &dense_object, &heights_object, &uncertain_object,
                          &weak_min_voxels, &immediate_min_voxels, &min_extent,
                          &immediate_min_height, &envelope_support_mode)) return nullptr;
    Buffer cloud(cloud_object);
    Buffer labels(labels_object);
    Buffer core(core_object);
    Buffer boundary(boundary_object);
    Buffer dense(dense_object);
    Buffer heights(heights_object);
    Buffer uncertain(uncertain_object);
    if (!cloud.points() || !labels.integers() || !core.flags() || !boundary.flags() || !dense.flags()
            || !heights.vector() || !uncertain.flags()) {
        PyErr_SetString(PyExc_ValueError, "expected cloud (K,3), int64 labels, bool masks and heights");
        return nullptr;
    }
    const Py_ssize_t n = cloud.rows();
    if (labels.size() != n || core.size() != n || boundary.size() != n || dense.size() != n
            || heights.size() != n || uncertain.size() != n) {
        PyErr_SetString(PyExc_ValueError, "cloud, labels, masks and heights must agree in length");
        return nullptr;
    }
    const double* points = cloud.doubles();
    const int64_t* tag = labels.int64s();
    const bool* in_core = core.bools();
    const bool* on_boundary = boundary.bools();
    const bool* is_dense = dense.bools();
    const double* bed_height = heights.doubles();
    const bool* is_uncertain = uncertain.bools();
    Arena& scratch = arena();
    std::vector<int64_t> label_rows, offsets_out, member_rows, reason_codes, relations,
        distance_codes, relation_reasons, support_counts, dense_counts, envelope_counts,
        boundary_counts, support_points, interior_dense;
    std::vector<double> bbox_min, bbox_max, centres, extents, height_spans, witnesses, distances,
        nearest_cluster, nearest_supported, nearest_unresolved, interior_heights;
    std::vector<uint8_t> immediate_flags, intersection_flags;
    try {
        ReleaseGIL released;
        int64_t lowest = 0, highest = -1;
        for (Py_ssize_t i = 0; i < n; ++i) {
            const int64_t value = tag[i];
            if (value < 0) continue;
            if (highest < 0 || value < lowest) lowest = value;
            if (highest < 0 || value > highest) highest = value;
        }
        auto& counts = scratch.i0;
        auto& members = scratch.i1;
        auto& present = scratch.i2;
        if (highest >= 0) {
            const int64_t span = highest - lowest + 1;
            counts.assign(static_cast<size_t>(span) + 1, 0);
            for (Py_ssize_t i = 0; i < n; ++i)
                if (tag[i] >= 0) ++counts[static_cast<size_t>(tag[i] - lowest) + 1];
            for (int64_t index = 0; index < span; ++index)
                counts[static_cast<size_t>(index) + 1] += counts[static_cast<size_t>(index)];
            members.resize(static_cast<size_t>(counts[static_cast<size_t>(span)]));
            auto& cursor = scratch.i3;
            cursor.assign(counts.begin(), counts.end() - 1);
            for (Py_ssize_t i = 0; i < n; ++i) {
                if (tag[i] < 0) continue;
                members[static_cast<size_t>(cursor[static_cast<size_t>(tag[i] - lowest)]++)] = i;
            }
            // Absent labels contribute no members, so removing their empty runs
            // leaves the surviving offsets valid.
            offsets_out.assign(1, 0);
            for (int64_t index = 0; index < span; ++index) {
                if (counts[static_cast<size_t>(index) + 1] > counts[static_cast<size_t>(index)]) {
                    label_rows.push_back(lowest + index);
                    offsets_out.push_back(counts[static_cast<size_t>(index) + 1]);
                }
            }
            // The members live in the arena, which the next call reuses, so the
            // returned copy is taken here.
            member_rows.assign(members.begin(), members.end());
        }
        const size_t rows = label_rows.size();
        for (size_t row = 0; row < rows; ++row) {
            const int64_t begin = offsets_out[row], end = offsets_out[row + 1];
            const size_t length = static_cast<size_t>(end - begin);
            double low[3] = {HUGE_VAL, HUGE_VAL, HUGE_VAL};
            double high[3] = {-HUGE_VAL, -HUGE_VAL, -HUGE_VAL};
            double lowest_x = HUGE_VAL, height_low = HUGE_VAL, height_high = -HUGE_VAL;
            for (size_t slot = 0; slot < length; ++slot) {
                const int64_t i = members[static_cast<size_t>(begin) + slot];
                for (int axis = 0; axis < 3; ++axis) {
                    const double value = points[3 * i + axis];
                    if (value < low[axis]) low[axis] = value;
                    if (value > high[axis]) high[axis] = value;
                }
                if (points[3 * i] < lowest_x) lowest_x = points[3 * i];
                if (bed_height[i] < height_low) height_low = bed_height[i];
                if (bed_height[i] > height_high) height_high = bed_height[i];
            }
            bbox_min.insert(bbox_min.end(), low, low + 3);
            bbox_max.insert(bbox_max.end(), high, high + 3);
            for (int axis = 0; axis < 3; ++axis) centres.push_back(0.5 * (low[axis] + high[axis]));
            double largest = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                const double span_axis = high[axis] - low[axis];
                extents.push_back(span_axis);
                if (span_axis > largest) largest = span_axis;
            }
            height_spans.push_back(height_low);
            height_spans.push_back(height_high);
            nearest_cluster.push_back(lowest_x);
            int64_t inside = 0, uncertain_count = 0, dense_count = 0, boundary_count = 0;
            int64_t interior_dense_count = 0;
            double interior_low = HUGE_VAL, interior_high = -HUGE_VAL;
            double supported_x = HUGE_VAL, unresolved_x = HUGE_VAL;
            if (static_cast<int>(length) >= weak_min_voxels && largest >= min_extent) {
                for (size_t slot = 0; slot < length; ++slot) {
                    const int64_t i = members[static_cast<size_t>(begin) + slot];
                    if (in_core[i]) {
                        ++inside;
                        if (is_dense[i]) ++interior_dense_count;
                        // The interior span is the height spread of the support
                        // points themselves, not of the bed-relative heights.
                        if (points[3 * i + 2] < interior_low) interior_low = points[3 * i + 2];
                        if (points[3 * i + 2] > interior_high) interior_high = points[3 * i + 2];
                        if (points[3 * i] < supported_x) supported_x = points[3 * i];
                    }
                    if (is_uncertain[i]) {
                        ++uncertain_count;
                        if (points[3 * i] < unresolved_x) unresolved_x = points[3 * i];
                    }
                    if (is_dense[i]) ++dense_count;
                    if (on_boundary[i]) ++boundary_count;
                }
            }
            const bool intersects = inside >= weak_min_voxels;
            const bool unresolved = uncertain_count >= weak_min_voxels;
            int reason = 0;
            if (static_cast<int>(length) < weak_min_voxels) reason = 1;
            else if (largest < min_extent) reason = 2;
            else if (!intersects && !unresolved && dense_count == 0) reason = 3;
            reason_codes.push_back(reason);
            if (reason != 0) {
                relations.push_back(0);
                relation_reasons.push_back(0);
                distance_codes.push_back(0);
                distances.push_back(0.0);
                witnesses.insert(witnesses.end(), 3, 0.0);
                support_points.push_back(0);
                support_counts.push_back(static_cast<int64_t>(length));
                dense_counts.push_back(dense_count);
                envelope_counts.push_back(inside);
                boundary_counts.push_back(boundary_count);
                immediate_flags.push_back(0);
                interior_dense.push_back(0);
                interior_heights.push_back(0.0);
                intersection_flags.push_back(0);
                nearest_supported.push_back(NAN);
                nearest_unresolved.push_back(NAN);
                continue;
            }
            // The reported distance follows the configured support definition.
            int mode = 0;
            if (envelope_support_mode != 0) {
                if (intersects) mode = 1;
                else if (unresolved) mode = 2;
            }
            double witness_x = HUGE_VAL, witness_y = 0.0, witness_z = 0.0;
            int64_t considered = 0;
            for (size_t slot = 0; slot < length; ++slot) {
                const int64_t i = members[static_cast<size_t>(begin) + slot];
                const bool selected = mode == 0 ? true : (mode == 1 ? in_core[i] : is_uncertain[i]);
                if (!selected) continue;
                ++considered;
                if (points[3 * i] < witness_x) {
                    witness_x = points[3 * i];
                    witness_y = points[3 * i + 1];
                    witness_z = points[3 * i + 2];
                }
            }
            witnesses.push_back(witness_x);
            witnesses.push_back(witness_y);
            witnesses.push_back(witness_z);
            distances.push_back(witness_x);
            support_points.push_back(considered);
            support_counts.push_back(static_cast<int64_t>(length));
            dense_counts.push_back(dense_count);
            envelope_counts.push_back(inside);
            boundary_counts.push_back(boundary_count);
            immediate_flags.push_back(
                (dense_count >= immediate_min_voxels && extents[3 * row + 2] >= immediate_min_height) ? 1 : 0);
            interior_dense.push_back(interior_dense_count);
            const double interior_span = interior_low > interior_high ? 0.0 : interior_high - interior_low;
            interior_heights.push_back(interior_span);
            intersection_flags.push_back(
                (interior_dense_count >= immediate_min_voxels && interior_span >= immediate_min_height) ? 1 : 0);
            relations.push_back(intersects ? 1 : (unresolved ? 2 : 0));
            relation_reasons.push_back(intersects ? 0 : (unresolved ? (boundary_count > 0 ? 1 : 2) : 3));
            distance_codes.push_back(mode);
            nearest_supported.push_back(inside > 0 ? supported_x : NAN);
            nearest_unresolved.push_back(unresolved ? unresolved_x : NAN);
        }
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    catch (const std::exception& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
    const std::vector<std::pair<const void*, size_t>> blocks = {
        {label_rows.data(), label_rows.size() * sizeof(int64_t)},
        {offsets_out.data(), offsets_out.size() * sizeof(int64_t)},
        {member_rows.data(), member_rows.size() * sizeof(int64_t)},
        {reason_codes.data(), reason_codes.size() * sizeof(int64_t)},
        {relations.data(), relations.size() * sizeof(int64_t)},
        {distance_codes.data(), distance_codes.size() * sizeof(int64_t)},
        {relation_reasons.data(), relation_reasons.size() * sizeof(int64_t)},
        {bbox_min.data(), bbox_min.size() * sizeof(double)},
        {bbox_max.data(), bbox_max.size() * sizeof(double)},
        {centres.data(), centres.size() * sizeof(double)},
        {extents.data(), extents.size() * sizeof(double)},
        {height_spans.data(), height_spans.size() * sizeof(double)},
        {witnesses.data(), witnesses.size() * sizeof(double)},
        {distances.data(), distances.size() * sizeof(double)},
        {support_points.data(), support_points.size() * sizeof(int64_t)},
        {nearest_cluster.data(), nearest_cluster.size() * sizeof(double)},
        {nearest_supported.data(), nearest_supported.size() * sizeof(double)},
        {nearest_unresolved.data(), nearest_unresolved.size() * sizeof(double)},
        {support_counts.data(), support_counts.size() * sizeof(int64_t)},
        {dense_counts.data(), dense_counts.size() * sizeof(int64_t)},
        {envelope_counts.data(), envelope_counts.size() * sizeof(int64_t)},
        {boundary_counts.data(), boundary_counts.size() * sizeof(int64_t)},
        {immediate_flags.data(), immediate_flags.size()},
        {interior_dense.data(), interior_dense.size() * sizeof(int64_t)},
        {interior_heights.data(), interior_heights.size() * sizeof(double)},
        {intersection_flags.data(), intersection_flags.size()},
    };
    PyObject* payload = PyTuple_New(static_cast<Py_ssize_t>(blocks.size()));
    if (payload == nullptr) return nullptr;
    std::vector<PyObject*> owned;
    owned.reserve(blocks.size());
    for (size_t index = 0; index < blocks.size(); ++index) {
        PyObject* block_object = bytes_of(blocks[index].first, blocks[index].second);
        if (block_object == nullptr) {
            Py_DECREF(payload);
            for (PyObject* object : owned) Py_DECREF(object);
            return nullptr;
        }
        owned.push_back(block_object);
        PyTuple_SET_ITEM(payload, static_cast<Py_ssize_t>(index), block_object);
    }
    return payload;
}

// Neighbour counts and mean-centred covariances of a point cloud.
//
// This reproduces what the planner previously obtained from two separate Open3D
// traversals (estimate_normals and estimate_covariances), measured against the
// real library rather than assumed:
//
//   * neighbours are the points within `radius`, capped to the `max_nn` nearest;
//   * the covariance is sum((x - mean) (x - mean)^T) / n over that capped set;
//   * the eigen-decomposition stays with NumPy, so the eigenvalues and the
//     smallest eigenvector are produced by the same solver as before.
//
// One grid pass therefore replaces two KD-tree traversals per point and yields
// the uncapped count as well, which the planarity gate needs. Accumulation order
// differs from Open3D's internal order, so covariances agree to round-off rather
// than bitwise; the gate thresholds are far from that margin.
PyObject* normal_covariances(PyObject*, PyObject* args) {
    PyObject* object;
    double radius;
    int max_nn;
    if (!PyArg_ParseTuple(args, "Odi", &object, &radius, &max_nn)) return nullptr;
    if (!(std::isfinite(radius) && radius > 0) || max_nn < 1) {
        PyErr_SetString(PyExc_ValueError, "radius must be positive and max_nn at least one");
        return nullptr;
    }
    Buffer buffer(object);
    if (!buffer.points()) {
        PyErr_SetString(PyExc_ValueError, "expected contiguous native float64 (N,3)");
        return nullptr;
    }
    const double* data = buffer.doubles();
    const Py_ssize_t n = buffer.rows();
    Arena& scratch = arena();
    auto& counts = scratch.i0;
    auto& covariances = scratch.d0;
    auto& eigenvalues = scratch.d1;
    auto& normals_out = scratch.d2;
    try {
        ReleaseGIL released;
        build_cells(data, n, radius);
        Arena& cells = arena();
        KeyTable& table = cells.table;
        counts.resize(static_cast<size_t>(n));
        covariances.resize(static_cast<size_t>(n) * 6);
        eigenvalues.resize(static_cast<size_t>(n) * 3);
        normals_out.resize(static_cast<size_t>(n) * 3);
        const unsigned workers = worker_count(n);
        // Per-thread scratch: the membership and selection buffers are mutated
        // inside the parallel region, so they cannot live in the shared arena.
        auto process = [&](Py_ssize_t start, Py_ssize_t stop) {
            std::vector<int64_t> members;
            std::vector<int64_t> picked;
            for (Py_ssize_t i = start; i < stop; ++i) {
                const double xi = data[3 * i], yi = data[3 * i + 1], zi = data[3 * i + 2];
                Key low{}, high{};
                for (int axis = 0; axis < 3; ++axis) {
                    const double value = data[3 * i + axis];
                    low[axis] = static_cast<int64_t>(std::floor((value - radius) / radius));
                    high[axis] = static_cast<int64_t>(std::floor((value + radius) / radius));
                }
                // One pass: membership is both the uncapped count and the pool the
                // nearest max_nn are selected from.
                members.clear();
                for (int64_t cx = low[0]; cx <= high[0]; ++cx)
                    for (int64_t cy = low[1]; cy <= high[1]; ++cy)
                        for (int64_t cz = low[2]; cz <= high[2]; ++cz) {
                            const size_t slot = table.find(Key{cx, cy, cz});
                            if (slot == static_cast<size_t>(-1)) continue;
                            const int64_t* begin = cells.cell_items.data() + cells.cell_start[slot];
                            const int64_t* end = cells.cell_items.data() + cells.cell_start[slot + 1];
                            for (const int64_t* item = begin; item != end; ++item) {
                                const int64_t j = *item;
                                if (distance_squared(data, j, xi, yi, zi) <= radius * radius) members.push_back(j);
                            }
                        }
                counts[static_cast<size_t>(i)] = static_cast<int64_t>(members.size());
                const Py_ssize_t total = static_cast<Py_ssize_t>(members.size());
                const Py_ssize_t used = total > max_nn ? max_nn : total;
                picked.resize(static_cast<size_t>(total));
                for (Py_ssize_t slot = 0; slot < total; ++slot) picked[static_cast<size_t>(slot)] = slot;
                if (total > max_nn) {
                    std::partial_sort(picked.begin(), picked.begin() + max_nn, picked.end(),
                                      [&](int64_t left, int64_t right) {
                                          const double dl = distance_squared(data, members[static_cast<size_t>(left)], xi, yi, zi);
                                          const double dr = distance_squared(data, members[static_cast<size_t>(right)], xi, yi, zi);
                                          if (dl != dr) return dl < dr;
                                          return members[static_cast<size_t>(left)] < members[static_cast<size_t>(right)];
                                      });
                }
                double mean_x = 0.0, mean_y = 0.0, mean_z = 0.0;
                for (Py_ssize_t slot = 0; slot < used; ++slot) {
                    const int64_t j = members[static_cast<size_t>(picked[static_cast<size_t>(slot)])];
                    mean_x += data[3 * j];
                    mean_y += data[3 * j + 1];
                    mean_z += data[3 * j + 2];
                }
                mean_x /= static_cast<double>(used);
                mean_y /= static_cast<double>(used);
                mean_z /= static_cast<double>(used);
                double xx = 0.0, xy = 0.0, xz = 0.0, yy = 0.0, yz = 0.0, zz = 0.0;
                for (Py_ssize_t slot = 0; slot < used; ++slot) {
                    const int64_t j = members[static_cast<size_t>(picked[static_cast<size_t>(slot)])];
                    const double dx = data[3 * j] - mean_x;
                    const double dy = data[3 * j + 1] - mean_y;
                    const double dz = data[3 * j + 2] - mean_z;
                    xx += dx * dx;
                    xy += dx * dy;
                    xz += dx * dz;
                    yy += dy * dy;
                    yz += dy * dz;
                    zz += dz * dz;
                }
                const double scale = 1.0 / static_cast<double>(used);
                const size_t base = 6 * static_cast<size_t>(i);
                covariances[base] = xx * scale;
                covariances[base + 1] = xy * scale;
                covariances[base + 2] = xz * scale;
                covariances[base + 3] = yy * scale;
                covariances[base + 4] = yz * scale;
                covariances[base + 5] = zz * scale;
                double values[3], direction[3];
                symmetric_eigen3(xx * scale, xy * scale, xz * scale, yy * scale, yz * scale, zz * scale,
                                 values, direction);
                for (int axis = 0; axis < 3; ++axis) {
                    eigenvalues[3 * static_cast<size_t>(i) + axis] = values[axis];
                    normals_out[3 * static_cast<size_t>(i) + axis] = direction[axis];
                }
            }
        };
        if (workers <= 1) {
            process(0, n);
        } else {
            std::vector<std::thread> pool;
            const Py_ssize_t chunk = (n + static_cast<Py_ssize_t>(workers) - 1) / static_cast<Py_ssize_t>(workers);
            for (unsigned worker = 0; worker < workers; ++worker) {
                const Py_ssize_t start = static_cast<Py_ssize_t>(worker) * chunk;
                const Py_ssize_t stop = std::min(n, start + chunk);
                if (start >= stop) break;
                pool.emplace_back([&process, start, stop] { process(start, stop); });
            }
            for (auto& thread : pool) thread.join();
        }
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
    catch (const std::exception& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
    PyObject* payload = PyTuple_New(4);
    if (payload == nullptr) return nullptr;
    PyObject* count_bytes = bytes_of(counts.data(), counts.size() * sizeof(int64_t));
    PyObject* covariance_bytes = bytes_of(covariances.data(), covariances.size() * sizeof(double));
    PyObject* eigenvalue_bytes = bytes_of(eigenvalues.data(), eigenvalues.size() * sizeof(double));
    PyObject* normal_bytes = bytes_of(normals_out.data(), normals_out.size() * sizeof(double));
    if (count_bytes == nullptr || covariance_bytes == nullptr || eigenvalue_bytes == nullptr
            || normal_bytes == nullptr) {
        Py_XDECREF(count_bytes);
        Py_XDECREF(covariance_bytes);
        Py_XDECREF(eigenvalue_bytes);
        Py_XDECREF(normal_bytes);
        Py_DECREF(payload);
        return nullptr;
    }
    PyTuple_SET_ITEM(payload, 0, count_bytes);
    PyTuple_SET_ITEM(payload, 1, covariance_bytes);
    PyTuple_SET_ITEM(payload, 2, eigenvalue_bytes);
    PyTuple_SET_ITEM(payload, 3, normal_bytes);
    return payload;
}

static PyMethodDef methods[] = {
    {"voxel_indices", voxel_indices, METH_VARARGS,
     "First measurement indices, lexicographic voxel order."},
    {"voxel_count", voxel_count, METH_VARARGS,
     "Number of distinct voxels for the same keys."},
    {"voxel_counts", voxel_counts, METH_VARARGS,
     "Distinct voxel count for each stack in one call, reusing one arena."},
    {"normal_covariances", normal_covariances, METH_VARARGS,
     "Neighbour counts and mean-centred covariances in one grid pass."},
    {"cluster_components", cluster_components, METH_VARARGS,
     "Component grouping and statistics of the cluster cloud in one pass."},
    {"select_crop_voxels", select_crop_voxels, METH_VARARGS,
     "Longitudinal-window crop and voxel reduction in one slice-partitioned pass."},
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
    {"within_radius", within_radius, METH_VARARGS,
     "Per-query flag: a target lies within the Euclidean protection radius."},
    {"ground_values", ground_values, METH_VARARGS,
     "Track-bed reference height and extrapolation uncertainty for the current scan."},
    {"classify_geometry", classify_geometry, METH_VARARGS,
     "Envelope, rail-relative and ground support classification of the current scan."},
    {"mask_candidates", mask_candidates, METH_VARARGS,
     "All patch candidates, their slice offsets and their unique union in one call."},
    {"mask_apply", mask_apply, METH_VARARGS,
     "Full per-patch background mask pass, marking the caller's mask in place."},
    {nullptr, nullptr, 0, nullptr}
};

static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native", nullptr, -1, methods};

PyMODINIT_FUNC PyInit__native() { return PyModule_Create(&module); }
