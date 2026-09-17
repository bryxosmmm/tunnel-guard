// Shared declarations for the single `tunnel_guard._native` extension module.
#pragma once

#include <Python.h>

// cpp/voxel.cpp
PyObject* voxel_indices(PyObject* self, PyObject* args);
PyObject* voxel_count(PyObject* self, PyObject* args);

// cpp/kernels.cpp -- registered in kernels.cpp
PyObject* range_indices(PyObject* self, PyObject* args);
PyObject* mutual_graph(PyObject* self, PyObject* args);
PyObject* patch_candidates(PyObject* self, PyObject* args);
PyObject* strip_inside(PyObject* self, PyObject* args);
PyObject* protrusion_ids(PyObject* self, PyObject* args);
PyObject* within_radius(PyObject* self, PyObject* args);
PyObject* classify_geometry(PyObject* self, PyObject* args);
PyObject* ground_values(PyObject* self, PyObject* args);
