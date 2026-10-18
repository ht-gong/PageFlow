# tqpcache: the granule cache's core (granule_core.cc), host only: no CUDA,
# no torch.  Staged into TQPlib/tqpcache/ like the other modules.
Python3_add_library(tqpcache MODULE
  granule_core.cc
  WITH_SOABI)

target_link_libraries(tqpcache PRIVATE Python3::Module pybind11::module pybind11::headers)
target_compile_options(tqpcache PRIVATE -O2)

set_target_properties(tqpcache PROPERTIES
  CXX_STANDARD 17
  POSITION_INDEPENDENT_CODE ON
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)
