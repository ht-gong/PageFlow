# --- Define the Python Module ---
Python3_add_library(zero_copy_reader MODULE
  zero_copy_reader.cu
  WITH_SOABI)

target_link_libraries(zero_copy_reader PRIVATE Python3::Module pybind11::module)
# Link against Torch. Prefer imported targets if available.
if(TARGET Torch::Torch)
  target_link_libraries(zero_copy_reader PRIVATE Torch::Torch)
elseif(TARGET torch)
  target_link_libraries(zero_copy_reader PRIVATE torch)
elseif(DEFINED TORCH_LIBRARIES)
  target_link_libraries(zero_copy_reader PRIVATE ${TORCH_LIBRARIES})
else()
  message(FATAL_ERROR "Torch not available while building zero_copy_reader")
endif()

if (TARGET Torch::Python)
  target_link_libraries(zero_copy_reader PRIVATE Torch::Python)
elseif(DEFINED TORCH_PYTHON_LIBRARY)
  target_link_libraries(zero_copy_reader PRIVATE ${TORCH_PYTHON_LIBRARY})
else()
  message(FATAL_ERROR "Torch Python lib not available while building zero_copy_reader")
endif()


set_target_properties(zero_copy_reader PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)


add_subdirectory(cache)
