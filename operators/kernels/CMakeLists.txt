# --- The operators' CUDA kernels: one Python module, TQPlib/tqpkernels ---
Python3_add_library(tqpkernels MODULE
  tqpkernels.cu
  partition_sort.cu
  join_core.cu
  WITH_SOABI)

target_link_libraries(tqpkernels PRIVATE Python3::Module pybind11::module)
# Link against Torch. Prefer imported targets if available.
if(TARGET Torch::Torch)
  target_link_libraries(tqpkernels PRIVATE Torch::Torch)
elseif(TARGET torch)
  target_link_libraries(tqpkernels PRIVATE torch)
elseif(DEFINED TORCH_LIBRARIES)
  target_link_libraries(tqpkernels PRIVATE ${TORCH_LIBRARIES})
else()
  message(FATAL_ERROR "Torch not available while building tqpkernels")
endif()

if (TARGET Torch::Python)
  target_link_libraries(tqpkernels PRIVATE Torch::Python)
elseif(DEFINED TORCH_PYTHON_LIBRARY)
  target_link_libraries(tqpkernels PRIVATE ${TORCH_PYTHON_LIBRARY})
else()
  message(FATAL_ERROR "Torch Python lib not available while building tqpkernels")
endif()

# Source lines for Nsight Compute and compute-sanitizer reports; the generated
# code is the same.
target_compile_options(tqpkernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

set_target_properties(tqpkernels PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)
