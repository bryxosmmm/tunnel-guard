// CUDA backend for the detector's native kernels.
//
// Design rules, in the order they were decided:
//
// 1. Drop-in. This module exposes every entry point `tunnel_guard._native` exposes.
//    What is ported here runs on the device; everything else is forwarded to the
//    CPU extension by name, so a caller that swaps the module cannot silently lose
//    a kernel. `cuda_entry_points()` lists what actually runs on the device.
//
// 2. Bit-identical where ported, or not ported at all. The CPU kernels reproduce
//    their NumPy expressions value for value; the device code reproduces the CPU
//    expressions the same way - same order, no reassociation, `fma()` exactly where
//    the CPU calls std::fma, no fast-math, and `--fmad=false` at compile time so a
//    product-sum never contracts on one side only.
//
// 3. One device slab, bumped per call and reset at its start: the device
//    counterpart of the CPU arena. No cudaMalloc in steady state, no per-kernel
//    synchronisation beyond the one copy that returns the result.
//
// 4. Small calls stay on the CPU. Under the thresholds below the transfer costs
//    more than the work, so the entry point forwards to the CPU kernel. That is a
//    documented policy, not a silent fallback, and it is listed in the module's
//    own report.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <random>
#include <thread>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kThreads = 256;
constexpr Py_ssize_t kClassifyMinimum = 4096;
constexpr Py_ssize_t kFrontMinimum = 4096;
constexpr Py_ssize_t kRangeMinimum = 1 << 16;
// Below this many points a plane proposal is evaluated on the host: the device round
// trip costs more than the sequential scan, and both paths are the same algorithm.
constexpr Py_ssize_t kPlaneDeviceMinimum = 4096;
constexpr size_t kPoolBytes = size_t(512) << 20;
constexpr int64_t kKeyBias = 1 << 20;

[[noreturn]] void fail(const std::string& what) { throw std::runtime_error(what); }

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(status));
}

struct Pool {
    char* base = nullptr;
    size_t capacity = 0;
    size_t used = 0;
    std::vector<void*> extra;

    ~Pool() {
        for (void* block : extra) cudaFree(block);
        if (base) cudaFree(base);
    }

    void* take(size_t bytes, size_t alignment = 256) {
        if (bytes == 0) bytes = 1;
        const size_t aligned = (used + alignment - 1) / alignment * alignment;
        if (base && aligned + bytes <= capacity) {
            void* block = base + aligned;
            used = aligned + bytes;
            return block;
        }
        void* block = nullptr;
        check(cudaMalloc(&block, bytes), "cudaMalloc");
        extra.push_back(block);
        return block;
    }

    // Pageable host memory copies at a fraction of the link speed, so transfers go
    // through one pinned staging buffer that grows and is never freed in steady
    // state. The extra memcpy on the host is cheaper than the difference.
    void* pinned = nullptr;
    size_t pinned_bytes = 0;

    void* stage(size_t bytes) {
        if (bytes <= pinned_bytes) return pinned;
        if (pinned) cudaFreeHost(pinned);
        size_t capacity_target = 1 << 16;
        while (capacity_target < bytes) capacity_target <<= 1;
        check(cudaMallocHost(&pinned, capacity_target), "cudaMallocHost");
        pinned_bytes = capacity_target;
        return pinned;
    }

    // Device arrays are allocated through the slab in the same order on every call,
    // so a frame that fits the slab allocates nothing.
    void ensure() {
        if (base) return;
        check(cudaMalloc(&base, kPoolBytes), "cudaMalloc pool");
        capacity = kPoolBytes;
    }

    void reset() {
        for (void* block : extra) cudaFree(block);
        extra.clear();
        used = 0;
    }
};

Pool& pool() {
    static Pool instance;
    return instance;
}

struct ReleaseGIL {
    PyThreadState* state = PyEval_SaveThread();
    ~ReleaseGIL() { PyEval_RestoreThread(state); }
};

// Entry points that are not ported yet, and calls below the size thresholds, are
// served by the CPU extension. Declared here because the CUDA entry points use it.
PyObject* delegate_to_cpu(const char* name, PyObject* args);

// ------------------------------------------------------------------ device code

__device__ inline double interp_of(double value, const double* xp, const double* fp, int m) {
    if (m == 1) return fp[0];
    if (isnan(value)) return value;
    if (value <= xp[0]) return fp[0];
    if (value >= xp[m - 1]) return fp[m - 1];
    int low = 0, high = m - 1;
    while (high - low > 1) {
        const int middle = (low + high) / 2;
        if (xp[middle] <= value) low = middle; else high = middle;
    }
    const double slope = (fp[low + 1] - fp[low]) / (xp[low + 1] - xp[low]);
    return fma(slope, value - xp[low], fp[low]);
}

__device__ inline double nearest_of(double value, const double* anchor_x, int m) {
    int position = m;
    if (!isnan(value)) {
        int low = 0, high = m;
        while (low < high) {
            const int middle = (low + high) / 2;
            if (anchor_x[middle] < value) low = middle + 1; else high = middle;
        }
        position = low;
    }
    if (position < 1) position = 1;
    if (position > m - 1) position = m - 1;
    const double left = fabs(value - anchor_x[position - 1]);
    const double right = fabs(anchor_x[position] - value);
    return left < right ? left : right;
}

__device__ inline int segment_of(const double* edges, int m, double value) {
    int low = 0, high = m;
    while (low < high) {
        const int middle = (low + high) / 2;
        if (edges[middle] <= value) low = middle + 1; else high = middle;
    }
    int index = low - 1;
    if (index < 0) index = 0;
    if (index > m - 1) index = m - 1;
    return index;
}

struct ClassifyArgs {
    const double* points;
    const double* plane;
    int plane_size;
    const double* ground_x;
    const double* ground_shift;
    const double* ground_spread;
    int ground_count;
    const double* rail_x;
    const double* rail_center;
    const double* rail_gauge;
    int rail_count;
    const double* segments;
    const double* segment_edges;
    int segment_count;
    double edge_x[2], edge_y[2], edge_slope[2], edge_curvature[2];
    double edge_slope_sigma[2], edge_curvature_sigma[2], edge_covariance[2];
    double rail_head, ground_max_uncertainty, path_max_uncertainty, ground_max_extrapolation,
        path_max_extrapolation, rail_half_width, rail_vertical_margin, min_running_height,
        cluster_context_margin, segmentation_half_width, envelope_margin, nominal_gauge,
        normal_scale, lateral_scale, slope_plane, low_edge, high_edge;
    uint8_t* core;
    uint8_t* context;
    uint8_t* observed;
    uint8_t* overlap;
    uint8_t* boundary;
    double* height;
    double* lateral;
    double* running;
    double* gauge;
};

// One thread per point, the entire classifier. The CPU version needs ten temporary
// arrays of size n and eleven passes over them; here each point computes its own
// interpolation, continuation and masks and writes the nine results once.
__global__ void classify_kernel(ClassifyArgs p, Py_ssize_t n) {
    const Py_ssize_t i = static_cast<Py_ssize_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double* point = p.points + 3 * i;
    const double x = point[0], y = point[1], z_point = point[2];

    double ground_uncertainty = INFINITY;
    double height = NAN;
    if (!(p.plane_size == 0 || p.ground_count == 0)) {
        const double shift = interp_of(x, p.ground_x, p.ground_shift, p.ground_count);
        double spread = interp_of(x, p.ground_x, p.ground_spread, p.ground_count);
        const double reach = nearest_of(x, p.ground_x, p.ground_count);
        spread = spread + reach * 0.008;
        if (reach > p.ground_max_extrapolation) spread = INFINITY;
        ground_uncertainty = spread;
        const double plane_z = x * p.plane[0] + y * p.plane[1] + p.plane[2] + shift;
        height = z_point - plane_z;
    }
    p.height[i] = height;

    double centre = 0.0;
    double gauge_value = p.nominal_gauge;
    double path_uncertainty = INFINITY;
    if (p.rail_count >= 2) {
        double rail_centre = interp_of(x, p.rail_x, p.rail_center, p.rail_count);
        gauge_value = interp_of(x, p.rail_x, p.rail_gauge, p.rail_count);
        const double reach = nearest_of(x, p.rail_x, p.rail_count);
        path_uncertainty = 0.06 + 0.008 * reach + 0.0003 * (reach * reach);
        // The two continuation edges, in the CPU's order. They cover disjoint
        // intervals, so a point outside the anchor span is handled by exactly one.
        for (int edge = 0; edge < 2; ++edge) {
            const bool below = edge == 0;
            const bool outside = below ? (x < p.rail_x[0]) : (x > p.rail_x[p.rail_count - 1]);
            if (!outside) continue;
            const double distance = x - p.edge_x[edge];
            rail_centre = p.edge_y[edge] + p.edge_slope[edge] * distance
                + p.edge_curvature[edge] * distance * distance;
            const double combined =
                (distance * p.edge_slope_sigma[edge]) * (distance * p.edge_slope_sigma[edge]) +
                ((distance * distance) * p.edge_curvature_sigma[edge]) *
                    ((distance * distance) * p.edge_curvature_sigma[edge]) +
                2.0 * ((distance * distance) * distance) * p.edge_covariance[edge];
            const double extension = sqrt(fmax(combined, 0.0));
            path_uncertainty = sqrt(path_uncertainty * path_uncertainty + extension * extension);
        }
        if (reach > p.path_max_extrapolation) path_uncertainty = INFINITY;
        centre = rail_centre;
    }
    p.gauge[i] = gauge_value;

    const double relative = height - p.rail_head;
    const double running_height = relative / p.normal_scale;
    p.running[i] = running_height;
    const double dy = y - centre;
    const double lateral_value = (dy + p.slope_plane * (relative + p.slope_plane * dy)) / p.lateral_scale;
    p.lateral[i] = lateral_value;
    const int segment = segment_of(p.segment_edges, p.segment_count, running_height);
    const double* box = p.segments + 4 * segment;
    double fraction = (running_height - box[0]) / (box[1] - box[0]);
    fraction = fmax(0.0, fmin(1.0, fraction));
    const double half_width = box[2] + fraction * (box[3] - box[2]) + p.envelope_margin;
    const bool ground_ok = ground_uncertainty <= p.ground_max_uncertainty;
    const bool path_ok = path_uncertainty <= p.path_max_uncertainty;
    p.observed[i] = (ground_ok && path_ok) ? 1 : 0;
    const bool on_rail = p.observed[i]
        && (fabs(fabs(lateral_value) - gauge_value / 2.0) < p.rail_half_width)
        && (running_height <= p.rail_vertical_margin);
    const double bed_error = p.observed[i] ? ground_uncertainty : 0.0;
    const double height_error = bed_error / p.normal_scale;
    const double low_height = running_height - height_error;
    const double high_height = running_height + height_error;
    double min_width = INFINITY, max_width = -INFINITY;
    for (int seg = 0; seg < p.segment_count; ++seg) {
        const double* limits = p.segments + 4 * seg;
        const double bottom = limits[0], top = limits[1];
        const double first = fmax(low_height, bottom);
        const double second = fmin(high_height, top);
        if (!(first <= second)) continue;
        const double at_first = limits[2] + (limits[3] - limits[2]) * (first - bottom) / (top - bottom)
            + p.envelope_margin;
        const double at_second = limits[2] + (limits[3] - limits[2]) * (second - bottom) / (top - bottom)
            + p.envelope_margin;
        min_width = fmin(min_width, fmin(at_first, at_second));
        max_width = fmax(max_width, fmax(at_first, at_second));
    }
    const double lateral_uncertainty = (p.observed[i] ? path_uncertainty : 0.0) * p.lateral_scale
        + fabs(p.slope_plane) * bed_error / p.lateral_scale;
    const bool vertical = (low_height >= p.low_edge) && (high_height <= p.high_edge);
    p.core[i] = (p.observed[i] && !on_rail && vertical
                 && (fabs(lateral_value) + lateral_uncertainty <= min_width)) ? 1 : 0;
    const bool possible = (high_height >= p.low_edge) && (low_height <= p.high_edge)
        && (fabs(lateral_value) - lateral_uncertainty <= max_width);
    p.boundary[i] = (p.observed[i] && !on_rail && p.core[i] == 0 && possible) ? 1 : 0;
    const bool segmentation = (running_height >= p.min_running_height)
        && (running_height <= p.high_edge + p.cluster_context_margin);
    p.context[i] = (segmentation && !on_rail
                    && (fabs(lateral_value) <= p.segmentation_half_width || !path_ok)) ? 1 : 0;
    p.overlap[i] = ((running_height >= p.low_edge) && (running_height <= p.high_edge) && !on_rail
                    && (fabs(lateral_value) <= half_width)) ? 1 : 0;
    if (p.plane_size == 0 || p.ground_count == 0 || p.rail_count < 2) {
        const bool vertical_measured = ground_ok && isfinite(p.rail_head);
        p.context[i] = (!vertical_measured || segmentation) ? 1 : 0;
        p.core[i] = 0;
        p.observed[i] = 0;
        p.overlap[i] = 0;
        p.boundary[i] = 0;
        p.lateral[i] = NAN;
        p.running[i] = NAN;
    }
}

__global__ void range_flag_kernel(const double* points, Py_ssize_t n, double low, double high,
                                  uint8_t* flags) {
    const Py_ssize_t i = static_cast<Py_ssize_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double x = points[3 * i], y = points[3 * i + 1], z = points[3 * i + 2];
    const double radius = sqrt(x * x + y * y + z * z);
    flags[i] = (radius >= low && radius <= high) ? 1 : 0;
}

__global__ void range_scatter_kernel(const int* offsets, const uint8_t* flags, Py_ssize_t n,
                                     int64_t* out) {
    const Py_ssize_t i = static_cast<Py_ssize_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n || !flags[i]) return;
    out[offsets[i]] = static_cast<int64_t>(i);
}

// Crop predicate and packed voxel key in one pass. A point outside the crop is
// never keyed, exactly as the CPU kernel skips it before deriving the key.
__global__ void crop_key_kernel(const double* points, Py_ssize_t n, double min_forward,
                                double half_width, double size, uint8_t* pass, uint64_t* key,
                                int* out_of_range) {
    const Py_ssize_t i = static_cast<Py_ssize_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double x = points[3 * i], y = points[3 * i + 1];
    if (!(x >= min_forward && fabs(y) < half_width)) {
        pass[i] = 0;
        return;
    }
    pass[i] = 1;
    const double values[3] = {x, y, points[3 * i + 2]};
    uint64_t word = 0;
    for (int axis = 0; axis < 3; ++axis) {
        const double scaled = floor(values[axis] / size);
        const int64_t k = static_cast<int64_t>(scaled);
        // The packed ordering is only the lexicographic one while every key fits
        // the biased field the CPU checks for; outside it the caller is served by
        // the CPU kernel instead of by a silently wrong order.
        if (k < -kKeyBias || k >= kKeyBias) {
            atomicExch(out_of_range, 1);
            return;
        }
        const uint64_t biased = static_cast<uint64_t>(k + kKeyBias);
        word |= biased << (42 - 21 * axis);
    }
    key[i] = word;
}

__global__ void scatter_pairs_kernel(const int* offsets, const uint8_t* pass, Py_ssize_t n,
                                     const uint64_t* key, uint64_t* out_key, int64_t* out_value) {
    const Py_ssize_t i = static_cast<Py_ssize_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n || !pass[i]) return;
    const int slot = offsets[i];
    out_key[slot] = key[i];
    out_value[slot] = static_cast<int64_t>(i);
}

__global__ void run_head_kernel(const uint64_t* key, int64_t count, uint8_t* head) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    head[i] = (i == 0 || key[i] != key[i - 1]) ? 1 : 0;
}

__global__ void gather_runs_kernel(const int* offsets, const uint8_t* head, int64_t count,
                                   const int64_t* value, int64_t* out) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count || !head[i]) return;
    out[offsets[i]] = value[i];
}

// The CPU summary evaluates every bin against every row. The counts are integers
// and each bin owns three slots, so the device version assigns each row to its bin
// and accumulates: the same integers, accumulated in a different order.
__global__ void summary_geometry_kernel(const double* clouds, const uint8_t* supported,
                                        Py_ssize_t rows, const double* edges, int bins,
                                        unsigned long long* counters) {
    const Py_ssize_t i = static_cast<Py_ssize_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    const double x = clouds[3 * i];
    for (int bin = 0; bin < bins; ++bin) {
        if (x >= edges[2 * bin] && x < edges[2 * bin + 1]) {
            atomicAdd(&counters[3 * bin], 1ULL);
            if (supported[i]) atomicAdd(&counters[3 * bin + 2], 1ULL);
            return;
        }
    }
}

__global__ void summary_raw_kernel(const double* raw, const uint8_t* keep, Py_ssize_t rows,
                                   const double* edges, int bins, unsigned long long* counters) {
    const Py_ssize_t i = static_cast<Py_ssize_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows || !keep[i]) return;
    const double x = raw[3 * i];
    for (int bin = 0; bin < bins; ++bin) {
        if (x >= edges[2 * bin] && x < edges[2 * bin + 1]) {
            atomicAdd(&counters[3 * bin + 1], 1ULL);
            return;
        }
    }
}

inline unsigned grid_for(Py_ssize_t rows) {
    return static_cast<unsigned>((rows + kThreads - 1) / kThreads);
}

// -------------------------------------------------------------- host-side helpers

struct HostBuffer {
    Py_buffer view{};
    bool ok = false;

    HostBuffer(PyObject* object, int flags = PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) {
        ok = PyObject_GetBuffer(object, &view, flags) == 0;
    }
    ~HostBuffer() { if (ok) PyBuffer_Release(&view); }
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;

    bool points() const {
        return ok && view.ndim == 2 && view.shape[1] == 3 && view.itemsize == sizeof(double)
            && view.format && strcmp(view.format, "d") == 0;
    }
    bool matrix(Py_ssize_t cols) const {
        return ok && view.ndim == 2 && view.shape[1] == cols && view.itemsize == sizeof(double)
            && view.format && strcmp(view.format, "d") == 0;
    }
    bool vector() const {
        return ok && view.ndim == 1 && view.itemsize == sizeof(double)
            && view.format && strcmp(view.format, "d") == 0;
    }
    bool flags() const {
        return ok && view.ndim == 1 && view.itemsize == sizeof(bool)
            && view.format && strcmp(view.format, "?") == 0;
    }
    Py_ssize_t rows() const { return view.ndim >= 1 ? view.shape[0] : 0; }
    Py_ssize_t size() const {
        const Py_ssize_t itemsize = static_cast<Py_ssize_t>(view.itemsize);
        return itemsize ? view.len / itemsize : 0;
    }
    const double* doubles() const { return static_cast<const double*>(view.buf); }
    const uint8_t* bytes() const { return static_cast<const uint8_t*>(view.buf); }
};

template <typename T>
T* device_copy(const T* host, size_t count, const char* what) {
    if (count == 0) return nullptr;
    const size_t bytes = count * sizeof(T);
    T* device = static_cast<T*>(pool().take(bytes));
    void* staging = pool().stage(bytes);
    std::memcpy(staging, host, bytes);
    check(cudaMemcpy(device, staging, bytes, cudaMemcpyHostToDevice), what);
    return device;
}

template <typename T>
std::vector<T> host_copy(const T* device, size_t count, const char* what) {
    std::vector<T> host(count);
    if (!count) return host;
    const size_t bytes = count * sizeof(T);
    void* staging = pool().stage(bytes);
    check(cudaMemcpy(staging, device, bytes, cudaMemcpyDeviceToHost), what);
    std::memcpy(host.data(), staging, bytes);
    return host;
}

// One element, so a total can be read without bringing the whole array back.
template <typename T>
T tail_value(const T* device, int64_t count, const char* what) {
    T value{};
    void* staging = pool().stage(sizeof(T));
    check(cudaMemcpy(staging, device + (count - 1), sizeof(T), cudaMemcpyDeviceToHost), what);
    std::memcpy(&value, staging, sizeof(T));
    return value;
}

PyObject* bytes_of(const void* data, size_t bytes) {
    return PyBytes_FromStringAndSize(static_cast<const char*>(data), static_cast<Py_ssize_t>(bytes));
}

// Compact the flagged entries of `count` rows into a dense array, preserving order.
// A scan plus a scatter: the rank of a flagged row is the number of flagged rows
// before it, which is what makes the result the serial order again.
int* compact_offsets(const uint8_t* flags, int64_t count) {
    int* offsets = static_cast<int*>(pool().take(static_cast<size_t>(count + 1) * sizeof(int)));
    uint8_t* input = static_cast<uint8_t*>(pool().take(static_cast<size_t>(count) * sizeof(uint8_t)));
    check(cudaMemcpy(input, flags, static_cast<size_t>(count) * sizeof(uint8_t),
                     cudaMemcpyDeviceToDevice), "flags copy");
    void* temporary = nullptr;
    size_t bytes = 0;
    cub::DeviceScan::ExclusiveSum(temporary, bytes, input, offsets, static_cast<int>(count));
    temporary = pool().take(bytes);
    cub::DeviceScan::ExclusiveSum(temporary, bytes, input, offsets, static_cast<int>(count));
    return offsets;
}

// ------------------------------------ exact port of Open3D 0.19 SegmentPlane

// Open3D's plane proposals are a third-party kernel this project depends on, and
// they were 280 ms of a 714 ms frame. Porting them is only legitimate if the port
// returns what Open3D returns, value for value, so this is a transcription of
// cpp/open3d/geometry/PointCloudSegmentation.cpp at v0.19.0 plus the arithmetic
// facts that file does not state: Eigen evaluates the 4-element dot product, the
// 3-element norm and the refit sums in sequential order, and the wheel's build does
// not contract products into FMAs (measured: scripts/cuda_ransac_probe.py matched
// Open3D's plane bit for bit and its inlier set exactly with sequential order and
// no FMA, and the packet orders did not).
//
// Structure of the port, and why it is parallel: Open3D draws every iteration's
// three sample indices BEFORE evaluating anything, and its evaluation loop only
// accumulates a best result under a critical section. So each iteration is
// independent, and running one thread per iteration - each scanning the cloud in
// the original order, so its inlier count, its squared error and its rmse are the
// sequential values - reproduces the loop. The best-result update and the
// break_iteration stopping rule depend on the iteration ORDER, so they are replayed
// afterwards on the host, where std::log and std::pow are the same libm calls
// Open3D used.
// --------------------------------------------------------------- entry points

PyObject* cuda_classify_geometry(PyObject*, PyObject* args) {
    PyObject *points_object, *plane_object, *ground_object, *rail_object, *envelope_object;
    double rail_head, ground_max_uncertainty, path_max_uncertainty, ground_max_extrapolation,
        path_max_extrapolation, rail_half_width, rail_vertical_margin, min_running_height,
        cluster_context_margin, segmentation_half_width, envelope_margin, rail_max_heading,
        path_curve_window, path_curvature_significance, nominal_gauge;
    if (!PyArg_ParseTuple(args, "OOOOO" "ddddddddddddddd", &points_object, &plane_object, &ground_object,
                          &rail_object, &envelope_object, &rail_head, &ground_max_uncertainty,
                          &path_max_uncertainty, &ground_max_extrapolation, &path_max_extrapolation,
                          &rail_half_width, &rail_vertical_margin, &min_running_height,
                          &cluster_context_margin, &segmentation_half_width, &envelope_margin,
                          &rail_max_heading, &path_curve_window, &path_curvature_significance, &nominal_gauge))
        return nullptr;
    HostBuffer points(points_object);
    HostBuffer plane(plane_object);
    HostBuffer ground(ground_object);
    HostBuffer rail(rail_object);
    HostBuffer envelope(envelope_object);
    if (!points.points() || !plane.vector() || (plane.size() != 0 && plane.size() != 3) || !ground.matrix(3)
            || !rail.matrix(4) || !envelope.matrix(4)) {
        PyErr_SetString(PyExc_ValueError,
                        "expected points (N,3), plane (0 or 3), ground anchors (G,3), rail anchors (R,4), envelope (S,4)");
        return nullptr;
    }
    const Py_ssize_t n = points.rows();
    const Py_ssize_t g = ground.rows(), r = rail.rows(), s = envelope.rows();
    if (s < 1) {
        PyErr_SetString(PyExc_ValueError, "classification requires a nonempty envelope");
        return nullptr;
    }
    if (n < kClassifyMinimum || n > INT_MAX) {
        return delegate_to_cpu("classify_geometry", args);
    }
    try {
        Pool& memory = pool();
        memory.ensure();
        memory.reset();
        const double* host_points = points.doubles();
        const double* ground_data = ground.doubles();
        const double* rail_data = rail.doubles();
        const double* segments = envelope.doubles();

        std::vector<double> plane_values;
        std::vector<double> ground_x(static_cast<size_t>(g)), ground_shift(static_cast<size_t>(g)),
            ground_spread(static_cast<size_t>(g));
        std::vector<double> rail_x(static_cast<size_t>(r)), rail_center(static_cast<size_t>(r)),
            rail_gauge_values(static_cast<size_t>(r));
        std::vector<double> segment_edges(static_cast<size_t>(s));
        for (Py_ssize_t i = 0; i < g; ++i) {
            ground_x[static_cast<size_t>(i)] = ground_data[3 * i];
            ground_shift[static_cast<size_t>(i)] = ground_data[3 * i + 1];
            ground_spread[static_cast<size_t>(i)] = ground_data[3 * i + 2];
        }
        for (Py_ssize_t i = 0; i < r; ++i) {
            rail_x[static_cast<size_t>(i)] = rail_data[4 * i];
            rail_center[static_cast<size_t>(i)] = rail_data[4 * i + 1];
            rail_gauge_values[static_cast<size_t>(i)] = rail_data[4 * i + 2];
        }
        for (Py_ssize_t i = 0; i < s; ++i) segment_edges[static_cast<size_t>(i)] = segments[4 * i];
        if (plane.size() == 3) plane_values.assign(plane.doubles(), plane.doubles() + 3);

        ClassifyArgs arguments{};
        arguments.plane_size = static_cast<int>(plane.size());
        arguments.ground_count = static_cast<int>(g);
        arguments.rail_count = static_cast<int>(r);
        arguments.segment_count = static_cast<int>(s);
        arguments.rail_head = rail_head;
        arguments.ground_max_uncertainty = ground_max_uncertainty;
        arguments.path_max_uncertainty = path_max_uncertainty;
        arguments.ground_max_extrapolation = ground_max_extrapolation;
        arguments.path_max_extrapolation = path_max_extrapolation;
        arguments.rail_half_width = rail_half_width;
        arguments.rail_vertical_margin = rail_vertical_margin;
        arguments.min_running_height = min_running_height;
        arguments.cluster_context_margin = cluster_context_margin;
        arguments.segmentation_half_width = segmentation_half_width;
        arguments.envelope_margin = envelope_margin;
        arguments.nominal_gauge = nominal_gauge;
        arguments.low_edge = segments[0];
        arguments.high_edge = segments[4 * (s - 1) + 1];
        const double slope_plane = plane.size() == 3 ? plane_values[1] : 0.0;
        arguments.slope_plane = slope_plane;
        arguments.normal_scale = plane.size() == 3
            ? std::sqrt(1.0 + (plane_values[0] * plane_values[0] + plane_values[1] * plane_values[1])) : 1.0;
        arguments.lateral_scale = std::sqrt(1.0 + slope_plane * slope_plane);

        // Continuation parameters, computed here exactly as the CPU computes them:
        // the same scalar order over the anchors, then handed to the device as
        // scalars so the per-point work stays element-wise.
        if (r >= 2) {
            const std::array<int, 2> edges = {{0, -1}};
            for (int slot = 0; slot < 2; ++slot) {
                const int edge = edges[static_cast<size_t>(slot)];
                const int index_here = edge < 0 ? static_cast<int>(r) - 1 : 0;
                const double x_edge = rail_x[static_cast<size_t>(index_here)];
                const double y_edge = rail_center[static_cast<size_t>(index_here)];
                double slope = 0.0, curvature = 0.0, slope_sigma = 0.0, curvature_sigma = 0.0, covariance = 0.0;
                bool fitted = false;
                const bool below = edge == 0;
                if (r >= 3 && path_curve_window > 0) {
                    int count = 0;
                    double s11 = 0, s12 = 0, s22 = 0, b1 = 0, b2 = 0;
                    for (Py_ssize_t i = 0; i < r; ++i) {
                        const double ax = rail_x[static_cast<size_t>(i)];
                        if (below ? (ax > x_edge + path_curve_window) : (ax < x_edge - path_curve_window)) continue;
                        const double t = (ax - x_edge) / path_curve_window;
                        const double v = rail_center[static_cast<size_t>(i)] - y_edge;
                        const double tt = t * t;
                        s11 += tt; s12 += tt * t; s22 += tt * tt; b1 += t * v; b2 += tt * v;
                        ++count;
                    }
                    if (count >= 3) {
                        const double determinant = s11 * s22 - s12 * s12;
                        if (determinant > 0) {
                            const double slope_scaled = (b1 * s22 - b2 * s12) / determinant;
                            const double curvature_scaled = (s11 * b2 - s12 * b1) / determinant;
                            double residual_square = 0;
                            for (Py_ssize_t i = 0; i < r; ++i) {
                                const double ax = rail_x[static_cast<size_t>(i)];
                                if (below ? (ax > x_edge + path_curve_window) : (ax < x_edge - path_curve_window)) continue;
                                const double t = (ax - x_edge) / path_curve_window;
                                const double v = rail_center[static_cast<size_t>(i)] - y_edge;
                                const double residual = v - (slope_scaled * t + curvature_scaled * t * t);
                                residual_square += residual * residual;
                            }
                            const int dof = count - 2;
                            const double variance = dof > 0 ? residual_square / dof : 0.0;
                            slope_sigma = std::sqrt(std::max(variance * s22 / determinant, 0.0)) / path_curve_window;
                            curvature_sigma = std::sqrt(std::max(variance * s11 / determinant, 0.0)) /
                                              (path_curve_window * path_curve_window);
                            covariance = -variance * s12 /
                                         (determinant * path_curve_window * path_curve_window * path_curve_window);
                            double scaled = curvature_scaled;
                            if (std::abs(scaled) < path_curvature_significance * curvature_sigma *
                                                       path_curve_window * path_curve_window)
                                scaled = 0.0;
                            slope = slope_scaled / path_curve_window;
                            slope = std::max(-rail_max_heading, std::min(rail_max_heading, slope));
                            curvature = scaled / (path_curve_window * path_curve_window);
                            fitted = true;
                        }
                    }
                }
                if (!fitted) {
                    const int index_other = below ? 1 : static_cast<int>(r) - 2;
                    const double raw = (rail_center[static_cast<size_t>(index_here)] -
                                        rail_center[static_cast<size_t>(index_other)]) /
                                       (rail_x[static_cast<size_t>(index_here)] -
                                        rail_x[static_cast<size_t>(index_other)]);
                    slope = std::max(-rail_max_heading, std::min(rail_max_heading, raw));
                    curvature_sigma = 0.0;
                }
                arguments.edge_x[slot] = x_edge;
                arguments.edge_y[slot] = y_edge;
                arguments.edge_slope[slot] = slope;
                arguments.edge_curvature[slot] = curvature;
                arguments.edge_slope_sigma[slot] = slope_sigma;
                arguments.edge_curvature_sigma[slot] = curvature_sigma;
                arguments.edge_covariance[slot] = covariance;
            }
        }

        const auto size = static_cast<size_t>(n);
        uint8_t* core = static_cast<uint8_t*>(memory.take(size));
        uint8_t* context = static_cast<uint8_t*>(memory.take(size));
        uint8_t* observed = static_cast<uint8_t*>(memory.take(size));
        uint8_t* overlap = static_cast<uint8_t*>(memory.take(size));
        uint8_t* boundary = static_cast<uint8_t*>(memory.take(size));
        double* height = static_cast<double*>(memory.take(size * sizeof(double)));
        double* lateral = static_cast<double*>(memory.take(size * sizeof(double)));
        double* running = static_cast<double*>(memory.take(size * sizeof(double)));
        double* gauge_out = static_cast<double*>(memory.take(size * sizeof(double)));

        arguments.points = device_copy(host_points, size * 3, "points");
        arguments.plane = plane_values.empty() ? nullptr : device_copy(plane_values.data(), 3, "plane");
        arguments.ground_x = device_copy(ground_x.data(), static_cast<size_t>(g), "ground x");
        arguments.ground_shift = device_copy(ground_shift.data(), static_cast<size_t>(g), "ground shift");
        arguments.ground_spread = device_copy(ground_spread.data(), static_cast<size_t>(g), "ground spread");
        arguments.rail_x = device_copy(rail_x.data(), static_cast<size_t>(r), "rail x");
        arguments.rail_center = device_copy(rail_center.data(), static_cast<size_t>(r), "rail centre");
        arguments.rail_gauge = device_copy(rail_gauge_values.data(), static_cast<size_t>(r), "rail gauge");
        arguments.segments = device_copy(segments, static_cast<size_t>(4 * s), "envelope");
        arguments.segment_edges = device_copy(segment_edges.data(), static_cast<size_t>(s), "segment edges");
        arguments.core = core;
        arguments.context = context;
        arguments.observed = observed;
        arguments.overlap = overlap;
        arguments.boundary = boundary;
        arguments.height = height;
        arguments.lateral = lateral;
        arguments.running = running;
        arguments.gauge = gauge_out;

        {
            ReleaseGIL released;
            classify_kernel<<<grid_for(n), kThreads>>>(arguments, n);
            check(cudaGetLastError(), "classify launch");
            check(cudaDeviceSynchronize(), "classify");
        }

        const std::vector<uint8_t> core_host = host_copy(core, size, "core");
        const std::vector<uint8_t> context_host = host_copy(context, size, "context");
        const std::vector<uint8_t> observed_host = host_copy(observed, size, "observed");
        const std::vector<uint8_t> overlap_host = host_copy(overlap, size, "overlap");
        const std::vector<uint8_t> boundary_host = host_copy(boundary, size, "boundary");
        const std::vector<double> height_host = host_copy(height, size, "height");
        const std::vector<double> lateral_host = host_copy(lateral, size, "lateral");
        const std::vector<double> running_host = host_copy(running, size, "running");
        const std::vector<double> gauge_host = host_copy(gauge_out, size, "gauge");

        PyObject* payload = PyTuple_New(9);
        if (!payload) return nullptr;
        PyObject* items[9] = {
            bytes_of(core_host.data(), size), bytes_of(context_host.data(), size),
            bytes_of(height_host.data(), size * sizeof(double)), bytes_of(observed_host.data(), size),
            bytes_of(overlap_host.data(), size), bytes_of(boundary_host.data(), size),
            bytes_of(lateral_host.data(), size * sizeof(double)),
            bytes_of(running_host.data(), size * sizeof(double)),
            bytes_of(gauge_host.data(), size * sizeof(double))};
        for (int index = 0; index < 9; ++index) {
            if (items[index] == nullptr) {
                for (int other = 0; other < index; ++other) Py_DECREF(items[other]);
                Py_DECREF(payload);
                return nullptr;
            }
            PyTuple_SET_ITEM(payload, index, items[index]);
        }
        return payload;
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyObject* cuda_range_indices(PyObject*, PyObject* args) {
    PyObject* object;
    double minimum, maximum;
    if (!PyArg_ParseTuple(args, "Odd", &object, &minimum, &maximum)) return nullptr;
    HostBuffer points(object);
    if (!points.points()) {
        PyErr_SetString(PyExc_ValueError, "expected contiguous native float64 (N,3)");
        return nullptr;
    }
    const Py_ssize_t n = points.rows();
    if (n < kRangeMinimum || n > INT_MAX) return delegate_to_cpu("range_indices", args);
    try {
        Pool& memory = pool();
        memory.ensure();
        memory.reset();
        int64_t* indices = nullptr;
        Py_ssize_t kept = 0;
        {
            // Every Python call happens after this scope: the GIL is released only
            // around the device work, never across the construction of the result.
            ReleaseGIL released;
            const double* device_points = device_copy(points.doubles(), static_cast<size_t>(n) * 3, "points");
            uint8_t* flags = static_cast<uint8_t*>(memory.take(static_cast<size_t>(n)));
            range_flag_kernel<<<grid_for(n), kThreads>>>(device_points, n, minimum, maximum, flags);
            check(cudaGetLastError(), "range flag launch");
            check(cudaDeviceSynchronize(), "range flags");
            const uint8_t last = tail_value(flags, n, "last flag");
            const int* offsets = compact_offsets(flags, n);
            kept = static_cast<Py_ssize_t>(tail_value(offsets, n, "offsets")) + last;
            indices = static_cast<int64_t*>(memory.take(static_cast<size_t>(kept) * sizeof(int64_t)));
            range_scatter_kernel<<<grid_for(n), kThreads>>>(offsets, flags, n, indices);
            check(cudaGetLastError(), "range scatter launch");
            check(cudaDeviceSynchronize(), "range indices");
        }
        const std::vector<int64_t> host = host_copy(indices, static_cast<size_t>(kept), "indices");
        return bytes_of(host.data(), host.size() * sizeof(int64_t));
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyObject* cuda_select_crop_voxels(PyObject*, PyObject* args) {
    PyObject* object;
    double min_forward, half_width, size;
    if (!PyArg_ParseTuple(args, "Oddd", &object, &min_forward, &half_width, &size)) return nullptr;
    if (!(std::isfinite(size) && size > 0)) {
        PyErr_SetString(PyExc_ValueError, "voxel size must be finite and positive");
        return nullptr;
    }
    HostBuffer buffer(object);
    if (!buffer.points()) {
        PyErr_SetString(PyExc_ValueError, "expected contiguous native float64 (N,3)");
        return nullptr;
    }
    const Py_ssize_t n = buffer.rows();
    if (n < kFrontMinimum || n > INT_MAX) return delegate_to_cpu("select_crop_voxels", args);
    try {
        Pool& memory = pool();
        memory.ensure();
        memory.reset();
        int64_t* result = nullptr;
        Py_ssize_t voxels = 0;
        bool use_cpu = false;
        {
            ReleaseGIL released;
            const double* device_points = device_copy(buffer.doubles(), static_cast<size_t>(n) * 3, "points");
            uint8_t* pass = static_cast<uint8_t*>(memory.take(static_cast<size_t>(n)));
            uint64_t* keys = static_cast<uint64_t*>(memory.take(static_cast<size_t>(n) * sizeof(uint64_t)));
            int* out_of_range = static_cast<int*>(memory.take(sizeof(int)));
            check(cudaMemset(out_of_range, 0, sizeof(int)), "clear range flag");
            crop_key_kernel<<<grid_for(n), kThreads>>>(device_points, n, min_forward, half_width, size,
                                                      pass, keys, out_of_range);
            check(cudaGetLastError(), "crop key launch");
            check(cudaDeviceSynchronize(), "crop keys");
            const std::vector<int> range_flag = host_copy(out_of_range, size_t(1), "range flag");
            if (range_flag[0]) {
                // A key outside the packed field: the CPU kernel holds the exact ordering
                // for that case. Decided here, served after the GIL is back.
                use_cpu = true;
            } else {
                const uint8_t last = tail_value(pass, n, "last pass");
                const int* offsets = compact_offsets(pass, n);
                const Py_ssize_t count = static_cast<Py_ssize_t>(tail_value(offsets, n, "offsets")) + last;
                if (count > INT_MAX) {
                    use_cpu = true;
                } else if (count > 0) {
                    uint64_t* packed_key = static_cast<uint64_t*>(
                        memory.take(static_cast<size_t>(count) * sizeof(uint64_t)));
                    int64_t* packed_value = static_cast<int64_t*>(
                        memory.take(static_cast<size_t>(count) * sizeof(int64_t)));
                    scatter_pairs_kernel<<<grid_for(n), kThreads>>>(offsets, pass, n, keys, packed_key, packed_value);
                    check(cudaGetLastError(), "pair scatter launch");
                    uint64_t* sorted_key = static_cast<uint64_t*>(
                        memory.take(static_cast<size_t>(count) * sizeof(uint64_t)));
                    int64_t* sorted_value = static_cast<int64_t*>(
                        memory.take(static_cast<size_t>(count) * sizeof(int64_t)));
                    void* temporary = nullptr;
                    size_t bytes = 0;
                    cub::DeviceRadixSort::SortPairs(temporary, bytes, packed_key, sorted_key, packed_value,
                                                    sorted_value, static_cast<int>(count));
                    temporary = memory.take(bytes);
                    // A stable radix sort keeps equal keys in input order, so the first
                    // entry of every run is the first measurement of that voxel, which is
                    // the representative numpy.unique(return_index=True) selects.
                    cub::DeviceRadixSort::SortPairs(temporary, bytes, packed_key, sorted_key, packed_value,
                                                    sorted_value, static_cast<int>(count));
                    uint8_t* heads = static_cast<uint8_t*>(memory.take(static_cast<size_t>(count)));
                    run_head_kernel<<<grid_for(count), kThreads>>>(sorted_key, count, heads);
                    check(cudaGetLastError(), "run head launch");
                    const uint8_t last_head = tail_value(heads, count, "last head");
                    const int* head_offsets = compact_offsets(heads, count);
                    voxels = static_cast<Py_ssize_t>(tail_value(head_offsets, count, "head offsets")) + last_head;
                    result = static_cast<int64_t*>(memory.take(static_cast<size_t>(voxels) * sizeof(int64_t)));
                    gather_runs_kernel<<<grid_for(count), kThreads>>>(head_offsets, heads, count, sorted_value, result);
                    check(cudaGetLastError(), "gather launch");
                    check(cudaDeviceSynchronize(), "crop voxels");
                }
            }
        }
        if (use_cpu) return delegate_to_cpu("select_crop_voxels", args);
        const std::vector<int64_t> host = host_copy(result, static_cast<size_t>(voxels), "voxel rows");
        return bytes_of(host.data(), host.size() * sizeof(int64_t));
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyObject* cuda_range_summary(PyObject*, PyObject* args) {
    PyObject *reduced_object, *frame_object, *crop_object, *observed_object, *bins_object;
    if (!PyArg_ParseTuple(args, "OOOOO", &reduced_object, &frame_object, &crop_object, &observed_object,
                          &bins_object)) return nullptr;
    HostBuffer reduced(reduced_object);
    HostBuffer frame(frame_object);
    HostBuffer crop(crop_object);
    HostBuffer observed(observed_object);
    HostBuffer bins(bins_object);
    if (!reduced.points() || !frame.points() || !crop.flags() || !observed.flags() || !bins.matrix(2)) {
        PyErr_SetString(PyExc_ValueError, "expected two point clouds, two bool masks and (B,2) bin edges");
        return nullptr;
    }
    if (crop.size() != frame.rows() || observed.size() != reduced.rows()) {
        PyErr_SetString(PyExc_ValueError, "masks must match the clouds they describe");
        return nullptr;
    }
    const Py_ssize_t rows = reduced.rows(), raw_rows = frame.rows(), count = bins.rows();
    if (rows < kFrontMinimum) return delegate_to_cpu("range_summary", args);
    try {
        Pool& memory = pool();
        memory.ensure();
        memory.reset();
        const double* device_edges = device_copy(bins.doubles(), static_cast<size_t>(2 * count), "bin edges");
        unsigned long long* counters = static_cast<unsigned long long*>(
            memory.take(static_cast<size_t>(3 * count) * sizeof(unsigned long long)));
        {
            ReleaseGIL released;
            check(cudaMemset(counters, 0, static_cast<size_t>(3 * count) * sizeof(unsigned long long)), "clear");
            const double* device_cloud = device_copy(reduced.doubles(), static_cast<size_t>(rows) * 3, "reduced");
            const uint8_t* device_supported = device_copy(
                reinterpret_cast<const uint8_t*>(observed.view.buf), static_cast<size_t>(observed.size()), "observed");
            summary_geometry_kernel<<<grid_for(rows), kThreads>>>(device_cloud, device_supported, rows,
                                                                 device_edges, static_cast<int>(count), counters);
            check(cudaGetLastError(), "summary geometry launch");
            const double* device_raw = device_copy(frame.doubles(), static_cast<size_t>(raw_rows) * 3, "frame");
            const uint8_t* device_keep = device_copy(
                reinterpret_cast<const uint8_t*>(crop.view.buf), static_cast<size_t>(crop.size()), "crop");
            summary_raw_kernel<<<grid_for(raw_rows), kThreads>>>(device_raw, device_keep, raw_rows,
                                                                device_edges, static_cast<int>(count), counters);
            check(cudaGetLastError(), "summary raw launch");
            check(cudaDeviceSynchronize(), "range summary");
        }
        const std::vector<unsigned long long> host = host_copy(counters, static_cast<size_t>(3 * count), "counters");
        std::vector<int64_t> result(static_cast<size_t>(3 * count));
        for (size_t index = 0; index < result.size(); ++index)
            result[index] = static_cast<int64_t>(host[index]);
        return bytes_of(result.data(), result.size() * sizeof(int64_t));
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

}  // namespace

// ------------------------------------------------------------- delegation to CPU

namespace {

PyObject* cpu_module() {
    static PyObject* module = nullptr;
    if (module == nullptr) {
        module = PyImport_ImportModule("tunnel_guard._native");
        if (module == nullptr) PyErr_Clear();
    }
    return module;
}

}  // namespace

namespace {
PyObject* delegate_to_cpu(const char* name, PyObject* args) {
    PyObject* module = cpu_module();
    if (module == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "tunnel_guard._native is required as the CPU fallback");
        return nullptr;
    }
    PyObject* function = PyObject_GetAttrString(module, name);
    if (function == nullptr) return nullptr;
    PyObject* result = PyObject_CallObject(function, args);
    Py_DECREF(function);
    return result;
}

#define TG_FORWARD(NAME)                                                       \
    PyObject* cuda_forward_##NAME(PyObject*, PyObject* args) {                 \
        return delegate_to_cpu(#NAME, args);                                   \
    }

TG_FORWARD(mutual_graph)
TG_FORWARD(voxel_counts)
TG_FORWARD(patch_candidates)
TG_FORWARD(strip_inside)
TG_FORWARD(protrusion_ids)
TG_FORWARD(within_radius)
TG_FORWARD(ground_values)
TG_FORWARD(mask_candidates)
TG_FORWARD(mask_apply)
TG_FORWARD(cluster_components)
TG_FORWARD(normal_covariances)
TG_FORWARD(component_labels)
TG_FORWARD(window_indices)
TG_FORWARD(remove_rows)
TG_FORWARD(support_strips)
TG_FORWARD(ground_profile)
TG_FORWARD(voxel_indices)
TG_FORWARD(voxel_count)

// The global engine mirrors open3d::utility::random: one stream, seeded once per
// frame, consumed by every proposal in order. Replicating the stream is what makes
// the sampled clouds identical, and the draws are data-independent, so the number
// consumed by a call is fixed by the point count alone.
std::mt19937& plane_engine() {
    static std::mt19937 engine;
    return engine;
}

std::vector<int64_t> draw_samples(size_t num_points, int iterations, int ransac_n) {
    std::vector<int64_t> samples;
    samples.reserve(static_cast<size_t>(iterations) * static_cast<size_t>(ransac_n));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        std::vector<int64_t> picked;
        picked.reserve(static_cast<size_t>(ransac_n));
        while (static_cast<int>(picked.size()) < ransac_n) {
            const int64_t index = static_cast<int64_t>(plane_engine()() % static_cast<uint32_t>(num_points));
            if (std::find(picked.begin(), picked.end(), index) == picked.end()) picked.push_back(index);
        }
        samples.insert(samples.end(), picked.begin(), picked.end());
    }
    return samples;
}

// GetPlaneFromPoints: the plane minimising the summed squared distance to the inlier
// set, with the accumulation order of the original.
void fit_plane(const double* points, const std::vector<int64_t>& inliers, double* result) {
    const size_t count = inliers.size();
    double centroid[3] = {0.0, 0.0, 0.0};
    for (int64_t index : inliers) {
        const double* point = points + 3 * index;
        centroid[0] += point[0];
        centroid[1] += point[1];
        centroid[2] += point[2];
    }
    for (int axis = 0; axis < 3; ++axis) centroid[axis] = centroid[axis] / static_cast<double>(count);
    double xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
    for (int64_t index : inliers) {
        const double* point = points + 3 * index;
        const double r0 = point[0] - centroid[0];
        const double r1 = point[1] - centroid[1];
        const double r2 = point[2] - centroid[2];
        xx += r0 * r0;
        xy += r0 * r1;
        xz += r0 * r2;
        yy += r1 * r1;
        yz += r1 * r2;
        zz += r2 * r2;
    }
    const double det_x = yy * zz - yz * yz;
    const double det_y = xx * zz - xz * xz;
    const double det_z = xx * yy - xy * xy;
    double abc[3];
    if (det_x > det_y && det_x > det_z) {
        abc[0] = det_x; abc[1] = xz * yz - xy * zz; abc[2] = xy * yz - xz * yy;
    } else if (det_y > det_z) {
        abc[0] = xz * yz - xy * zz; abc[1] = det_y; abc[2] = xy * xz - yz * xx;
    } else {
        abc[0] = xy * yz - xz * yy; abc[1] = xy * xz - yz * xx; abc[2] = det_z;
    }
    const double norm = std::sqrt((abc[0] * abc[0] + abc[1] * abc[1]) + abc[2] * abc[2]);
    if (norm == 0.0) {
        result[0] = result[1] = result[2] = result[3] = 0.0;
        return;
    }
    for (int axis = 0; axis < 3; ++axis) abc[axis] = abc[axis] / norm;
    result[0] = abc[0];
    result[1] = abc[1];
    result[2] = abc[2];
    result[3] = -((abc[0] * centroid[0] + abc[1] * centroid[1]) + abc[2] * centroid[2]);
}

PyObject* cuda_segment_plane_seed(PyObject*, PyObject* args) {
    int seed = 0;
    if (!PyArg_ParseTuple(args, "i", &seed)) return nullptr;
    plane_engine().seed(static_cast<std::mt19937::result_type>(seed));
    Py_RETURN_NONE;
}

// A small persistent pool for iteration-parallel work. A task is a set of independent
// indices; workers pull them with an atomic counter, so which worker computes which
// index does not matter, and the caller works too. Each index's own work is sequential,
// which is what keeps the plane scans identical to Open3D's.
class WorkerPool {
public:
    static WorkerPool& instance() {
        static WorkerPool single;
        return single;
    }

    template <typename Body>
    void run(int count, Body body) {
        if (count <= 0) return;
        start();
        if (threads_.empty()) {
            for (int index = 0; index < count; ++index) body(index);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            body_ = [&body](int index) { body(index); };
            count_ = count;
            next_.store(0, std::memory_order_relaxed);
            active_ = static_cast<int>(threads_.size());
            ++generation_;
        }
        work_.notify_all();
        drain();
        std::unique_lock<std::mutex> lock(mutex_);
        finished_.wait(lock, [this] { return active_ == 0; });
        body_ = nullptr;
    }

    ~WorkerPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        work_.notify_all();
        for (auto& thread : threads_) thread.join();
    }

private:
    WorkerPool() = default;

    void start() {
        if (started_) return;
        started_ = true;
        const unsigned hardware = std::thread::hardware_concurrency();
        const unsigned workers = std::max(1u, std::min(hardware ? hardware : 1u, 8u));
        seen_.assign(workers > 0 ? workers - 1 : 0, 0);
        for (unsigned worker = 1; worker < workers; ++worker) {
            threads_.emplace_back([this, worker] { loop(worker); });
        }
    }

    void loop(unsigned worker) {
        unsigned& seen = seen_[worker - 1];
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                work_.wait(lock, [this, &seen] { return stopping_ || generation_ != seen; });
                seen = generation_;
                if (stopping_) return;
            }
            drain();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (--active_ == 0) finished_.notify_all();
            }
        }
    }

    void drain() {
        for (;;) {
            const int index = next_.fetch_add(1, std::memory_order_relaxed);
            if (index >= count_) return;
            body_(index);
        }
    }

    std::mutex mutex_;
    std::condition_variable work_;
    std::condition_variable finished_;
    std::vector<std::thread> threads_;
    std::vector<unsigned> seen_;
    std::function<void(int)> body_;
    std::atomic<int> next_{0};
    int count_ = 0;
    int active_ = 0;
    unsigned generation_ = 0;
    bool stopping_ = false;
    bool started_ = false;
};

template <typename Body>
void run_iterations(int count, Body body) {
    WorkerPool::instance().run(count, body);
}

// The sequential walk, on the host. Used for small windows, where a device round trip
// costs more than the work, and it is the same algorithm in the same order as the
// kernel: the two must agree, and the equivalence harness compares both against
// Open3D.
struct RansacModel {
    double plane[4];
    double fitness;
    double rmse;
    bool degenerate;
};

RansacModel evaluate_iteration(const double* points, int64_t n, const int64_t* sample, int ransac_n,
                               double threshold) {
    RansacModel model{};
    if (ransac_n != 3) {
        model.degenerate = true;
        return model;
    }
    const double* p0 = points + 3 * sample[0];
    const double* p1 = points + 3 * sample[1];
    const double* p2 = points + 3 * sample[2];
    const double e0[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
    const double e1[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
    double abc[3] = {e0[1] * e1[2] - e0[2] * e1[1],
                     e0[2] * e1[0] - e0[0] * e1[2],
                     e0[0] * e1[1] - e0[1] * e1[0]};
    const double norm = std::sqrt((abc[0] * abc[0] + abc[1] * abc[1]) + abc[2] * abc[2]);
    if (norm == 0.0) {
        model.degenerate = true;
        return model;
    }
    for (int axis = 0; axis < 3; ++axis) abc[axis] = abc[axis] / norm;
    model.plane[0] = abc[0];
    model.plane[1] = abc[1];
    model.plane[2] = abc[2];
    model.plane[3] = -((abc[0] * p0[0] + abc[1] * p0[1]) + abc[2] * p0[2]);
    double error = 0.0;
    int64_t count = 0;
    int64_t index = 0;
    for (; index + 4 <= n; index += 4) {
        const double* p0 = points + 3 * (index + 0);
        const double* p1 = points + 3 * (index + 1);
        const double* p2 = points + 3 * (index + 2);
        const double* p3 = points + 3 * (index + 3);
        const double d0 = std::fabs(((model.plane[0] * p0[0] + model.plane[1] * p0[1]) + model.plane[2] * p0[2]) + model.plane[3]);
        const double d1 = std::fabs(((model.plane[0] * p1[0] + model.plane[1] * p1[1]) + model.plane[2] * p1[2]) + model.plane[3]);
        const double d2 = std::fabs(((model.plane[0] * p2[0] + model.plane[1] * p2[1]) + model.plane[2] * p2[2]) + model.plane[3]);
        const double d3 = std::fabs(((model.plane[0] * p3[0] + model.plane[1] * p3[1]) + model.plane[2] * p3[2]) + model.plane[3]);
        if (d0 < threshold) { error += d0 * d0; ++count; }
        if (d1 < threshold) { error += d1 * d1; ++count; }
        if (d2 < threshold) { error += d2 * d2; ++count; }
        if (d3 < threshold) { error += d3 * d3; ++count; }
    }
    for (; index < n; ++index) {
        const double* point = points + 3 * index;
        const double distance = std::fabs(((model.plane[0] * point[0] + model.plane[1] * point[1])
                                           + model.plane[2] * point[2]) + model.plane[3]);
        if (distance < threshold) {
            error += distance * distance;
            ++count;
        }
    }
    model.fitness = static_cast<double>(count) / static_cast<double>(n);
    model.rmse = count == 0 ? 0.0 : std::sqrt(error / static_cast<double>(count));
    return model;
}

// The best-result update and the stopping rule, in iteration order, with the same
// libm Open3D used. Returns the winning iteration or -1.
int replay_best_iteration(const std::vector<RansacModel>& models, int iterations, int ransac_n,
                         double probability) {
    double best_fitness = 0.0, best_rmse = 0.0;
    int best_iteration = -1;
    int iteration_count = 0;
    size_t break_iteration = std::numeric_limits<size_t>::max();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        if (static_cast<size_t>(iteration_count) > break_iteration) continue;
        if (models[static_cast<size_t>(iteration)].degenerate) continue;
        const double fitness = models[static_cast<size_t>(iteration)].fitness;
        const double rmse = models[static_cast<size_t>(iteration)].rmse;
        if (fitness > best_fitness || (fitness == best_fitness && rmse < best_rmse)) {
            best_fitness = fitness;
            best_rmse = rmse;
            best_iteration = iteration;
            if (best_fitness < 1.0) {
                const double required = std::log(1.0 - probability)
                    / std::log(1.0 - std::pow(best_fitness, ransac_n));
                break_iteration = static_cast<size_t>(std::min(required, static_cast<double>(iterations)));
            } else {
                break_iteration = 0;
            }
        }
        ++iteration_count;
    }
    return best_iteration;
}

// Final inliers with the winning model, in ascending index order, then the refit.
void finish_plane(const double* points, int64_t n, const double* plane, double threshold, double* plane_out,
                  std::vector<int64_t>& inliers_out) {
    inliers_out.clear();
    for (int64_t index = 0; index < n; ++index) {
        const double* point = points + 3 * index;
        const double distance = std::fabs(((plane[0] * point[0] + plane[1] * point[1])
                                           + plane[2] * point[2]) + plane[3]);
        if (distance < threshold) inliers_out.push_back(index);
    }
    fit_plane(points, inliers_out, plane_out);
}

PyObject* segment_plane_result(const double* plane, const std::vector<int64_t>& inliers) {
    PyObject* plane_bytes = bytes_of(plane, 4 * sizeof(double));
    PyObject* index_bytes = bytes_of(inliers.data(), inliers.size() * sizeof(int64_t));
    if (!plane_bytes || !index_bytes) {
        Py_XDECREF(plane_bytes);
        Py_XDECREF(index_bytes);
        return nullptr;
    }
    PyObject* payload = PyTuple_New(2);
    PyTuple_SET_ITEM(payload, 0, plane_bytes);
    PyTuple_SET_ITEM(payload, 1, index_bytes);
    return payload;
}

PyObject* cuda_segment_plane(PyObject*, PyObject* args) {
    PyObject* object;
    double threshold, probability;
    int ransac_n, iterations;
    if (!PyArg_ParseTuple(args, "Odiid", &object, &threshold, &ransac_n, &iterations, &probability))
        return nullptr;
    HostBuffer buffer(object);
    if (!buffer.points()) {
        PyErr_SetString(PyExc_ValueError, "expected contiguous native float64 (N,3)");
        return nullptr;
    }
    const Py_ssize_t n = buffer.rows();
    if (!(probability > 0.0 && probability <= 1.0) || ransac_n < 3 || iterations < 1) {
        PyErr_SetString(PyExc_ValueError, "probability must be in (0, 1], ransac_n >= 3 and iterations >= 1");
        return nullptr;
    }
    if (static_cast<size_t>(n) < static_cast<size_t>(ransac_n) || n == 0) {
        // Open3D returns a zero plane and an empty index list for these inputs.
        const double zero[4] = {0.0, 0.0, 0.0, 0.0};
        return segment_plane_result(zero, std::vector<int64_t>{});
    }
    try {
        const double* points = buffer.doubles();
        const std::vector<int64_t> samples = draw_samples(static_cast<size_t>(n), iterations, ransac_n);
        double plane_host[4] = {0.0, 0.0, 0.0, 0.0};
        std::vector<int64_t> inliers_host;
        {
            ReleaseGIL released;
            if (n < kPlaneDeviceMinimum) {
                // Small window: the device round trip costs more than the scan.
                std::vector<RansacModel> models(static_cast<size_t>(iterations));
                for (int iteration = 0; iteration < iterations; ++iteration) {
                    models[static_cast<size_t>(iteration)] = evaluate_iteration(
                        points, n, samples.data() + static_cast<size_t>(iteration) * ransac_n, ransac_n, threshold);
                }
                const int winner = replay_best_iteration(models, iterations, ransac_n, probability);
                if (winner >= 0) {
                    finish_plane(points, n, models[static_cast<size_t>(winner)].plane, threshold, plane_host,
                                 inliers_host);
                }
            } else {
                // Parallel over iterations, sequential inside each: the scan order is
                // part of the result, and an iteration's scan is a serial dependency
                // chain, which cores run far better than a GPU does (measured: one
                // thread per iteration on the device cost 8.8 ms for a 31 000-point
                // window; the same iterations split over the CPU cores cost a few
                // hundred microseconds).
                std::vector<RansacModel> models(static_cast<size_t>(iterations));
                run_iterations(iterations, [&](int iteration) {
                    models[static_cast<size_t>(iteration)] = evaluate_iteration(
                        points, n, samples.data() + static_cast<size_t>(iteration) * ransac_n, ransac_n, threshold);
                });
                const int winner = replay_best_iteration(models, iterations, ransac_n, probability);
                if (winner >= 0) {
                    // The final pass and the refit are sequential in the original, so
                    // they run here rather than as two more device round trips.
                    finish_plane(points, n, models[static_cast<size_t>(winner)].plane, threshold, plane_host,
                                 inliers_host);
                }
            }
        }
        return segment_plane_result(plane_host, inliers_host);
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
        return nullptr;
    }
}

PyObject* cuda_entry_points(PyObject*, PyObject*) {
    static const char* names[] = {"classify_geometry", "range_indices", "select_crop_voxels", "range_summary",
                                  "segment_plane"};
    const int count = 5;
    PyObject* list = PyList_New(count);
    if (list == nullptr) return nullptr;
    for (int index = 0; index < count; ++index) {
        PyObject* name = PyUnicode_FromString(names[index]);
        if (name == nullptr) { Py_DECREF(list); return nullptr; }
        PyList_SET_ITEM(list, index, name);
    }
    return list;
}

PyObject* cuda_device_info(PyObject*, PyObject*) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) return Py_BuildValue("{s:i}", "devices", 0);
    if (count == 0) return Py_BuildValue("{s:i}", "devices", 0);
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess) return Py_BuildValue("{s:i}", "devices", 0);
    return Py_BuildValue("{s:i,s:s,s:i,s:i,s:L}", "devices", count, "name", properties.name,
                         "major", properties.major, "minor", properties.minor,
                         "memory_bytes", static_cast<long long>(properties.totalGlobalMem));
}

PyMethodDef methods[] = {
    {"classify_geometry", cuda_classify_geometry, METH_VARARGS, "CUDA classifier."},
    {"range_indices", cuda_range_indices, METH_VARARGS, "CUDA radial band filter."},
    {"select_crop_voxels", cuda_select_crop_voxels, METH_VARARGS, "CUDA crop and voxel reduction."},
    {"range_summary", cuda_range_summary, METH_VARARGS, "CUDA range-bin summary."},
    {"mutual_graph", cuda_forward_mutual_graph, METH_VARARGS, "CPU kernel (device port pending)."},
    {"voxel_counts", cuda_forward_voxel_counts, METH_VARARGS, "CPU kernel (device port pending)."},
    {"patch_candidates", cuda_forward_patch_candidates, METH_VARARGS, "CPU kernel (device port pending)."},
    {"strip_inside", cuda_forward_strip_inside, METH_VARARGS, "CPU kernel (device port pending)."},
    {"protrusion_ids", cuda_forward_protrusion_ids, METH_VARARGS, "CPU kernel (device port pending)."},
    {"within_radius", cuda_forward_within_radius, METH_VARARGS, "CPU kernel (device port pending)."},
    {"ground_values", cuda_forward_ground_values, METH_VARARGS, "CPU kernel (device port pending)."},
    {"mask_candidates", cuda_forward_mask_candidates, METH_VARARGS, "CPU kernel (device port pending)."},
    {"mask_apply", cuda_forward_mask_apply, METH_VARARGS, "CPU kernel (device port pending)."},
    {"cluster_components", cuda_forward_cluster_components, METH_VARARGS, "CPU kernel (device port pending)."},
    {"normal_covariances", cuda_forward_normal_covariances, METH_VARARGS, "CPU kernel (device port pending)."},
    {"component_labels", cuda_forward_component_labels, METH_VARARGS, "CPU kernel (device port pending)."},
    {"window_indices", cuda_forward_window_indices, METH_VARARGS, "CPU kernel (device port pending)."},
    {"remove_rows", cuda_forward_remove_rows, METH_VARARGS, "CPU kernel (device port pending)."},
    {"support_strips", cuda_forward_support_strips, METH_VARARGS, "CPU kernel (device port pending)."},
    {"ground_profile", cuda_forward_ground_profile, METH_VARARGS, "CPU kernel (device port pending)."},
    {"voxel_indices", cuda_forward_voxel_indices, METH_VARARGS, "CPU kernel (device port pending)."},
    {"voxel_count", cuda_forward_voxel_count, METH_VARARGS, "CPU kernel (device port pending)."},
    {"segment_plane", cuda_segment_plane, METH_VARARGS,
     "Exact port of Open3D 0.19 SegmentPlane (plane and inlier set bit-identical)."},
    {"segment_plane_seed", cuda_segment_plane_seed, METH_VARARGS,
     "Seed the global proposal stream, mirroring o3d.utility.random.seed."},
    {"cuda_entry_points", cuda_entry_points, METH_NOARGS, "Entry points that run on the device."},
    {"cuda_device_info", cuda_device_info, METH_NOARGS, "Device report."},
    {nullptr, nullptr, 0, nullptr},
};

}  // namespace

PyMODINIT_FUNC PyInit__native_cuda() {
    static struct PyModuleDef definition = {
        PyModuleDef_HEAD_INIT, "tunnel_guard._native_cuda",
        "CUDA backend for the detector's native kernels; unported entry points forward to _native.",
        -1, methods, nullptr, nullptr, nullptr, nullptr};
    PyObject* module = PyModule_Create(&definition);
    if (module == nullptr) return nullptr;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        Py_DECREF(module);
        PyErr_SetString(PyExc_ImportError, "no CUDA device available");
        return nullptr;
    }
    return module;
}
