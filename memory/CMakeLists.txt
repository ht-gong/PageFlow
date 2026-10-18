# ---------- 1) Plain shared library for the custom allocator ----------
add_library(custom_allocator SHARED
    cuda_memory_allocator.cc
)

# Link against Torch. Prefer imported targets if available.
if(TARGET Torch::Torch)
  target_link_libraries(custom_allocator PRIVATE Torch::Torch)
elseif(TARGET torch)
  target_link_libraries(custom_allocator PRIVATE torch)
elseif(DEFINED TORCH_LIBRARIES)
  target_link_libraries(custom_allocator PRIVATE ${TORCH_LIBRARIES})
else()
  message(FATAL_ERROR "Torch not available while building custom_allocator")
endif()

set_target_properties(custom_allocator PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)

# ---------- 2) Python extension module (no pybind11_add_module) ----------
# Requires these to be available already:
#   - find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
#   - pybind11 targets (via CPM/find_package): pybind11::module, pybind11::headers

add_library(tqpmemory MODULE bindings.cpp)

# Link exactly what you linked before, plus pybind/Python module targets.
target_link_libraries(tqpmemory
  PRIVATE
    custom_allocator
    pybind11::module
    pybind11::headers
    Python3::Module
)

# Link against Torch (core + torch_python for pybind11 at::Tensor type caster)
if(TARGET Torch::Torch)
  target_link_libraries(tqpmemory PRIVATE Torch::Torch)
elseif(TARGET torch)
  target_link_libraries(tqpmemory PRIVATE torch)
elseif(DEFINED TORCH_LIBRARIES)
  target_link_libraries(tqpmemory PRIVATE ${TORCH_LIBRARIES})
endif()
target_link_libraries(tqpmemory PRIVATE ${TORCH_PYTHON_LIBRARY})

# Match Python extension naming (no 'lib' prefix; correct ABI suffix)
if(DEFINED Python3_EXTENSION_MODULE_SUFFIX)
  set(_pyext_suffix "${Python3_EXTENSION_MODULE_SUFFIX}")
else()
  set(_pyext_suffix ".so")  # fallback
endif()

set_target_properties(tqpmemory PROPERTIES
  PREFIX ""                                 # don't prepend 'lib'
  SUFFIX "${_pyext_suffix}"                 # e.g., .cpython-311-x86_64-linux-gnu.so
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)

# Keep build hygiene consistent with pybind11 defaults
set_property(TARGET tqpmemory custom_allocator PROPERTY CXX_STANDARD 17)
set_property(TARGET tqpmemory custom_allocator PROPERTY POSITION_INDEPENDENT_CODE ON)

# Optional: nicer errors from pybind11 (same macro pybind11 tends to set in dev)
target_compile_definitions(tqpmemory PRIVATE PYBIND11_DETAILED_ERROR_MESSAGES=1)
