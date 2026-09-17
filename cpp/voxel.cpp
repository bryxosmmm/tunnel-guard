#define PY_SSIZE_T_CLEAN
#include "native.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

bool voxel_key_of(const double* point, double size, Key& key) {
    for (int axis = 0; axis < 3; ++axis) {
        const double value = std::floor(point[axis] / size);
        if (!std::isfinite(value) || value < -0x1p63 || value >= 0x1p63) return false;
        key[axis] = static_cast<int64_t>(value);
    }
    return true;
}

struct ReleaseGIL {
    PyThreadState* state = PyEval_SaveThread();
    ~ReleaseGIL() { PyEval_RestoreThread(state); }
};

// Match numpy.unique(floor(points / size), axis=0, return_index=True):
// lexicographic voxel order, first original measurement in each voxel.
PyObject* voxel_indices(PyObject*, PyObject* args) {
    PyObject* object;
    double size;
    if (!PyArg_ParseTuple(args, "Od", &object, &size)) return nullptr;
    if (!(std::isfinite(size) && size > 0)) {
        PyErr_SetString(PyExc_ValueError, "voxel size must be finite and positive");
        return nullptr;
    }
    Py_buffer buffer{};
    if (PyObject_GetBuffer(object, &buffer, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) < 0) return nullptr;
    if (buffer.ndim != 2 || buffer.shape[1] != 3 || buffer.itemsize != sizeof(double)
        || !buffer.format || std::strcmp(buffer.format, "d") != 0) {
        PyBuffer_Release(&buffer);
        PyErr_SetString(PyExc_ValueError, "expected contiguous native float64 (N,3)");
        return nullptr;
    }
    std::vector<int64_t>* filled = nullptr;
    try {
        // The arena is only touched with the GIL released; the return value is
        // copied afterwards, when the guard has been destroyed.
        {
            ReleaseGIL released;
            const auto* data = static_cast<const double*>(buffer.buf);
            const auto n = buffer.shape[0];
            // Dedupe first: real clouds collapse to far fewer voxels than returns,
            // so probing N keys costs less than sorting all N. The arena table
            // grows to the largest frame seen and is then reused, so no node is
            // allocated per measurement.
            Arena& scratch = arena();
            KeyTable& table = scratch.table;
            table.reset(static_cast<size_t>(n));
            for (Py_ssize_t i = 0; i < n; ++i) {
                Key key{};
                if (!voxel_key_of(data + 3 * i, size, key))
                    throw std::invalid_argument("nonfinite or out-of-range voxel coordinate");
                bool inserted = false;
                const size_t slot = table.slot_of(key, inserted);
                if (inserted) table.values[slot] = static_cast<int64_t>(i);
            }
            auto& ordered = scratch.ordered;
            ordered.clear();
            ordered.reserve(table.count);
            for (size_t slot = 0; slot < table.keys.size(); ++slot)
                if (table.used[slot]) ordered.emplace_back(table.keys[slot], table.values[slot]);
            std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            auto& indices = scratch.i0;
            indices.clear();
            indices.reserve(ordered.size());
            for (const auto& entry : ordered) indices.push_back(entry.second);
            filled = &indices;
        }
        PyBuffer_Release(&buffer);
        return PyBytes_FromStringAndSize(reinterpret_cast<const char*>(filled->data()),
                                         static_cast<Py_ssize_t>(filled->size() * sizeof(int64_t)));
    } catch (const std::bad_alloc&) {
        PyBuffer_Release(&buffer);
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        PyBuffer_Release(&buffer);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
}
// Count distinct voxels; identical key derivation to voxel_indices, without
// materialising or sorting the keys when only the count is consumed.
PyObject* voxel_count(PyObject*, PyObject* args) {
    PyObject* object;
    double size;
    if (!PyArg_ParseTuple(args, "Od", &object, &size)) return nullptr;
    if (!(std::isfinite(size) && size > 0)) {
        PyErr_SetString(PyExc_ValueError, "voxel size must be finite and positive");
        return nullptr;
    }
    Py_buffer buffer{};
    if (PyObject_GetBuffer(object, &buffer, PyBUF_FORMAT | PyBUF_C_CONTIGUOUS) < 0) return nullptr;
    if (buffer.ndim != 2 || buffer.shape[1] != 3 || buffer.itemsize != sizeof(double)
        || !buffer.format || std::strcmp(buffer.format, "d") != 0) {
        PyBuffer_Release(&buffer);
        PyErr_SetString(PyExc_ValueError, "expected contiguous native float64 (N,3)");
        return nullptr;
    }
    Py_ssize_t distinct = 0;
    try {
        ReleaseGIL released;
        const auto* data = static_cast<const double*>(buffer.buf);
        const auto n = buffer.shape[0];
        KeyTable& table = arena().table;
        table.reset(static_cast<size_t>(n));
        for (Py_ssize_t i = 0; i < n; ++i) {
            Key key{};
            if (!voxel_key_of(data + 3 * i, size, key))
                throw std::invalid_argument("nonfinite or out-of-range voxel coordinate");
            bool inserted = false;
            table.slot_of(key, inserted);
        }
        distinct = static_cast<Py_ssize_t>(table.count);
    } catch (const std::bad_alloc&) {
        PyBuffer_Release(&buffer);
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        PyBuffer_Release(&buffer);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
    PyBuffer_Release(&buffer);
    return PyLong_FromSsize_t(distinct);
}
