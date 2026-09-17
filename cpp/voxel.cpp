#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using Key = std::array<int64_t, 3>;
struct Hash {
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

// Match numpy.unique(floor(points / size), axis=0, return_index=True):
// lexicographic voxel order, first original measurement in each voxel.
static PyObject* voxel_indices(PyObject*, PyObject* args) {
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
    std::vector<int64_t> indices;
    try {
        ReleaseGIL released;
        const auto* data = static_cast<const double*>(buffer.buf);
        const auto n = buffer.shape[0];
        // Dedupe first: real clouds collapse to far fewer voxels than returns,
        // so hashing N measurements costs less than sorting all N.
        std::unordered_map<Key, int64_t, Hash> first;
        first.reserve(static_cast<size_t>(n));
        for (Py_ssize_t i = 0; i < n; ++i) {
            Key key{};
            for (int axis = 0; axis < 3; ++axis) {
                const double v = std::floor(data[3*i+axis] / size);
                if (!std::isfinite(v) || v < -0x1p63 || v >= 0x1p63)
                    throw std::invalid_argument("nonfinite or out-of-range voxel coordinate");
                key[axis] = static_cast<int64_t>(v);
            }
            first.emplace(key, static_cast<int64_t>(i));
        }
        std::vector<std::pair<Key, int64_t>> ordered(first.begin(), first.end());
        std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        indices.reserve(ordered.size());
        for (const auto& entry : ordered) indices.push_back(entry.second);
    } catch (const std::bad_alloc&) {
        PyBuffer_Release(&buffer);
        return PyErr_NoMemory();
    } catch (const std::exception& error) {
        PyBuffer_Release(&buffer);
        PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
    PyBuffer_Release(&buffer);
    return PyBytes_FromStringAndSize(reinterpret_cast<const char*>(indices.data()),
                                    static_cast<Py_ssize_t>(indices.size() * sizeof(int64_t)));
}
// Count distinct voxels; identical key derivation to voxel_indices, without
// materialising or sorting the keys when only the count is consumed.
static PyObject* voxel_count(PyObject*, PyObject* args) {
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
        std::unordered_map<Key, int64_t, Hash> first;
        first.reserve(static_cast<size_t>(n));
        for (Py_ssize_t i = 0; i < n; ++i) {
            Key key{};
            for (int axis = 0; axis < 3; ++axis) {
                const double v = std::floor(data[3*i+axis] / size);
                if (!std::isfinite(v) || v < -0x1p63 || v >= 0x1p63)
                    throw std::invalid_argument("nonfinite or out-of-range voxel coordinate");
                key[axis] = static_cast<int64_t>(v);
            }
            first.emplace(key, static_cast<int64_t>(i));
        }
        distinct = static_cast<Py_ssize_t>(first.size());
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

static PyMethodDef methods[] = {
    {"voxel_indices", voxel_indices, METH_VARARGS, "First measurement indices, lexicographic voxel order."},
    {"voxel_count", voxel_count, METH_VARARGS, "Number of distinct voxels for the same keys."},
    {nullptr, nullptr, 0, nullptr}
};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_native", nullptr, -1, methods};
PyMODINIT_FUNC PyInit__native() { return PyModule_Create(&module); }
