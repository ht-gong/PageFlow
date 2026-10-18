# ------------------------------------------------------------
# CrossTorchGPU.cmake
# - Finds Torch from either libtorch or a Python wheel
# - Requires the CUDA toolkit and enables the CUDA language
#
# Cache knobs (optional):
#   TORCH_HINT         : Path to torch root (libtorch or site-packages/torch)
#
# Exposed variables:
#   TORCH_FOUND, TORCH_ROOT, Torch_VERSION, TORCH_PYTHON_LIBRARY
#   CUDAToolkit_VERSION
# ------------------------------------------------------------
cmake_minimum_required(VERSION 3.20)
include_guard(GLOBAL)

# --- 1. Find PyTorch ---
# We prioritize TORCH_HINT, then try to find Torch via its Python package,
# and finally let CMake search default paths.
if(NOT Torch_FOUND)
  set(_torch_search_paths)
  if(DEFINED TORCH_HINT)
    list(APPEND _torch_search_paths "${TORCH_HINT}")
  endif()

  # A robust way to find torch from a Python install is to ask torch itself.
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
      OUTPUT_VARIABLE _torch_cmake_prefix
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_QUIET
    )
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c "import pathlib, torch, sys; lib=(pathlib.Path(torch.__file__).resolve().parent/'lib'); print(lib)"
      OUTPUT_VARIABLE TORCH_PY_LIB_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    find_library(TORCH_PYTHON_LIBRARY
      NAMES torch_python
      PATHS "${TORCH_PY_LIB_DIR}" "${TORCH_INSTALL_PREFIX}/lib"
    )
    if(NOT TORCH_PYTHON_LIBRARY)
      message(FATAL_ERROR "Set -DTORCH_PYTHON_LIBRARY=/full/path/to/libtorch_python.so")
    endif()
    if(IS_DIRECTORY "${_torch_cmake_prefix}")
      list(APPEND _torch_search_paths "${_torch_cmake_prefix}")
    endif()
  endif()

  find_package(Torch HINTS ${_torch_search_paths})
endif()

if(NOT Torch_FOUND)
  message(FATAL_ERROR "Torch not found. Set -DTORCH_HINT to your libtorch root or Python's site-packages/torch directory.")
endif()

# Expose TORCH_ROOT derived from the found Torch_DIR
get_filename_component(TORCH_ROOT "${Torch_DIR}/../../.." ABSOLUTE)
message(STATUS "CrossTorchGPU: Torch found: ${Torch_FOUND} | Version: ${Torch_VERSION} | Root: ${TORCH_ROOT}")

# --- 2. CUDA ---
# compression/ hard-requires nvCOMP, so CUDA is the only supported backend.
find_package(CUDAToolkit REQUIRED)
enable_language(CUDA)
message(STATUS "CrossTorchGPU: CUDA Toolkit ${CUDAToolkit_VERSION}")
