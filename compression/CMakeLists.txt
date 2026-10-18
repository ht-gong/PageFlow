find_package(CUDAToolkit COMPONENTS cudart REQUIRED)
find_package(nvcomp REQUIRED)

# The nvCOMP library tqpcompress links: the nvidia-libnvcomp-cu12 wheel's
# lib64/libnvcomp.so.5, the file libcudf loads by path.  An rpath to its
# directory makes the dynamic loader map that same file (or reuse it when
# cuDF loaded it first), so a process holds one nvCOMP.
get_target_property(_nvcomp_lib nvcomp::nvcomp IMPORTED_LOCATION_RELEASE)
get_filename_component(_nvcomp_lib_dir "${_nvcomp_lib}" DIRECTORY)

add_library(tqpcompress MODULE compression.cu nvcomp_manager.cu)

target_link_libraries(tqpcompress
  PRIVATE

    nvcomp::nvcomp

    CUDA::cudart

    Python3::Module
)

# If your bindings include torch headers (e.g., torch/extension.h), keep Torch here too
if(TARGET Torch::Torch)
  target_link_libraries(tqpcompress PRIVATE Torch::Torch)
elseif(TARGET torch)
  target_link_libraries(tqpcompress PRIVATE torch)
elseif(DEFINED TORCH_LIBRARIES)
  target_link_libraries(tqpcompress PRIVATE ${TORCH_LIBRARIES})
else()
  message(FATAL_ERROR "Torch not available while building tqpcompress")
endif()

if (TARGET Torch::Python)
  target_link_libraries(tqpcompress PRIVATE Torch::Python)
elseif(DEFINED TORCH_PYTHON_LIBRARY)
  target_link_libraries(tqpcompress PRIVATE ${TORCH_PYTHON_LIBRARY})
else()
  message(FATAL_ERROR "Torch Python lib not available while building tqpcompress")
endif()

# --- 4. Set Target Properties (from reference) ---


# Set properties for a Python module (no 'lib' prefix, correct '.so' suffix)
if(DEFINED Python3_EXTENSION_MODULE_SUFFIX)
  set(_pyext_suffix "${Python3_EXTENSION_MODULE_SUFFIX}")
else()
  set(_pyext_suffix ".so") # Fallback
endif()

# In your CMakeLists.txt:

# --- Corrected Block for Debug Flags ---
if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    
    # 1. Flags for NVCC (CUDA-specific compiler)
    target_compile_options(tqpcompress PRIVATE
        # Generator expression: Only apply these flags if the compiler is NVCC (CUDA)
        $<$<COMPILE_LANGUAGE:CUDA>:
            -G           # Enables device-side debugging
        >
    )

    # 2. Flags for Host C++ Compiler (GCC/Clang)
    # This is typically handled automatically by CMAKE_BUILD_TYPE,
    # but you can add specific GCC flags here if needed.
    # For example, turning on maximum warnings:
    target_compile_options(tqpcompress PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:
            -Wall -Wextra
        >
    )
    
    message(STATUS "Debugging flags (-G, --lineinfo) enabled only for CUDA source files.")

else()
    # Add aggressive optimizations for Release builds
    target_compile_options(tqpcompress PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -O3
            --use_fast_math
        >
    )
endif()

set_target_properties(tqpcompress PROPERTIES
  PREFIX ""                     # Don't prepend 'lib'
  SUFFIX "${_pyext_suffix}"     # e.g., .cpython-311-x86_64-linux-gnu.so
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  BUILD_RPATH "$ORIGIN;${_nvcomp_lib_dir}"
  INSTALL_RPATH "$ORIGIN;${_nvcomp_lib_dir}"
  POSITION_INDEPENDENT_CODE ON
)


# Print handy messages
message(STATUS "Found Torch: ${torch_VERSION} at ${TORCH_CMAKE_DIR}")
message(STATUS "Found nvcomp ${nvcomp_VERSION}: at ${nvcomp_DIR}, linking ${_nvcomp_lib}")
message(STATUS "Building Python module: tqpcompress${_pyext_suffix}")

