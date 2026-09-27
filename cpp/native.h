// Shared declarations for the single `tunnel_guard._native` extension module.
#pragma once

#include <Python.h>

#include <array>
#include <algorithm>
#include <cstdint>
#include <thread>
#include <vector>

// Voxel key: floor(point / size) per axis, the same derivation for every kernel
// that has to agree with numpy.unique(..., axis=0).
using Key = std::array<int64_t, 3>;

struct KeyHash {
    size_t operator()(const Key& k) const noexcept {
        size_t h = 0;
        for (auto v : k) h ^= std::hash<int64_t>{}(v) + 0x9e3779b9U + (h << 6) + (h >> 2);
        return h;
    }
};

bool voxel_key_of(const double* point, double size, Key& key);

// Open-addressed key table. Iteration order is the caller's (insert order is
// preserved outside the table), and there is no per-key allocation: the arrays
// are reused across frames by the arena below.
struct KeyTable {
    std::vector<Key> keys;
    std::vector<int64_t> values;
    std::vector<uint8_t> used;
    size_t mask = 0;
    size_t count = 0;

    // Sized between 1x and 4x of what the next call needs: a modestly smaller
    // request keeps the current arrays (per-stack counting reuses them instead of
    // reallocating), while a much larger frame shrinks them back so probing stays
    // cache resident.
    void reset(size_t expected) {
        size_t capacity = 16;
        while (capacity < expected * 2 + 1) capacity <<= 1;
        if (keys.size() < capacity || keys.size() > capacity * 4) {
            keys.resize(capacity);
            values.resize(capacity);
            used.assign(capacity, 0);
        } else {
            std::fill(used.begin(), used.end(), 0);
        }
        mask = keys.size() - 1;
        count = 0;
    }

    // Index of the key's slot, inserting an empty slot when absent. The slot is
    // stable until the next reset(), so callers may cache it.
    size_t slot_of(const Key& key, bool& inserted) {
        size_t index = KeyHash{}(key) & mask;
        for (;;) {
            if (!used[index]) {
                used[index] = 1;
                keys[index] = key;
                values[index] = 0;
                ++count;
                inserted = true;
                return index;
            }
            if (keys[index] == key) {
                inserted = false;
                return index;
            }
            index = (index + 1) & mask;
        }
    }

    size_t find(const Key& key) const {
        size_t index = KeyHash{}(key) & mask;
        for (;;) {
            if (!used[index]) return static_cast<size_t>(-1);
            if (keys[index] == key) return index;
            index = (index + 1) & mask;
        }
    }
};

// Per-thread scratch that grows to the largest frame seen and is then reused, so
// steady-state frames allocate nothing beyond the copies the caller receives.
struct Arena {
    std::vector<double> d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10;
    std::vector<int64_t> i0, i1, i2, i3, i4, i5, i6;
    std::vector<uint8_t> b0, b1, b2, b3, b4;
    std::vector<std::pair<Key, int64_t>> ordered;
    std::vector<std::pair<uint64_t, int64_t>> packed;
    std::vector<std::vector<int64_t>> sinks;
    KeyTable table;
    std::vector<int64_t> cell_start;   // prefix sums over table slots + 1
    std::vector<int64_t> cell_items;   // flattened memberships, insert order
    std::vector<int64_t> cell_cursor;
    std::vector<Key> keys;             // per-point voxel keys, computed in parallel
    std::vector<uint8_t> flags;        // per-point crop/predicate results
    std::vector<std::pair<int64_t, int64_t>> spans;  // non-empty column extents
};

Arena& arena();

// Worker count for a pass over `rows` values: one below the threshold where
// thread setup would cost more than the work, otherwise the cores the process
// can actually use. `parallel_chunks` and its callers must agree on this value,
// so it is asked once and passed in.
unsigned worker_capacity(Py_ssize_t rows);

// Runs [0, rows) as contiguous chunks, one thread per worker, calling
// body(start, stop, worker). Element-wise kernels use it to get the cores the
// frame is otherwise leaving idle; because a chunk only ever writes the output
// elements it owns and the chunks are contiguous, every output element is
// computed by exactly the same arithmetic, in the same order, as the serial
// loop it replaces. Kernels that filter keep one sink per worker and
// concatenate them in worker order, which is the serial order again.
template <typename Body>
void parallel_chunks(Py_ssize_t rows, unsigned workers, Body body) {
    if (rows <= 0) return;
    if (workers <= 1) {
        body(static_cast<Py_ssize_t>(0), rows, 0u);
        return;
    }
    std::vector<std::thread> pool;
    const Py_ssize_t chunk = (rows + static_cast<Py_ssize_t>(workers) - 1) / static_cast<Py_ssize_t>(workers);
    for (unsigned worker = 0; worker < workers; ++worker) {
        const Py_ssize_t start = static_cast<Py_ssize_t>(worker) * chunk;
        const Py_ssize_t stop = std::min(rows, start + chunk);
        if (start >= stop) break;
        pool.emplace_back([&body, start, stop, worker] { body(start, stop, worker); });
    }
    for (auto& thread : pool) thread.join();
}

// Group point indices by voxel cell of the given size into the arena's table and
// flat membership arrays. Any point within `size` of another shares a cell, so
// the grouping is exact for a radius up to `size`.
void build_cells(const double* points, Py_ssize_t n, double size);

// cpp/voxel.cpp
PyObject* voxel_indices(PyObject* self, PyObject* args);
PyObject* voxel_count(PyObject* self, PyObject* args);
PyObject* select_crop_voxels(PyObject* self, PyObject* args);

// cpp/kernels.cpp -- registered in kernels.cpp
PyObject* range_indices(PyObject* self, PyObject* args);
PyObject* range_indices_open(PyObject* self, PyObject* args);
PyObject* mutual_graph(PyObject* self, PyObject* args);
PyObject* voxel_counts(PyObject* self, PyObject* args);
PyObject* patch_candidates(PyObject* self, PyObject* args);
PyObject* strip_inside(PyObject* self, PyObject* args);
PyObject* protrusion_ids(PyObject* self, PyObject* args);
PyObject* within_radius(PyObject* self, PyObject* args);
PyObject* ground_values(PyObject* self, PyObject* args);
PyObject* classify_geometry(PyObject* self, PyObject* args);
PyObject* mask_candidates(PyObject* self, PyObject* args);
PyObject* mask_apply(PyObject* self, PyObject* args);
PyObject* cluster_components(PyObject* self, PyObject* args);
PyObject* normal_covariances(PyObject* self, PyObject* args);
PyObject* component_labels(PyObject* self, PyObject* args);
PyObject* window_indices(PyObject* self, PyObject* args);
PyObject* remove_rows(PyObject* self, PyObject* args);
PyObject* support_strips(PyObject* self, PyObject* args);
PyObject* ground_profile(PyObject* self, PyObject* args);
PyObject* range_summary(PyObject* self, PyObject* args);
