include(FetchContent)
message(STATUS "get cccl ...")

set(cccl_DOWNLOAD_URL
    "https://github.com/NVIDIA/cccl/archive/refs/tags/v2.6.1.tar.gz"
    CACHE STRING "")

if(cccl_LOCAL_SOURCE)
  FetchContent_Declare(
    cccl
    SOURCE_DIR ${cccl_LOCAL_SOURCE}
    OVERRIDE_FIND_PACKAGE)
else()
  FetchContent_Declare(
    cccl
    URL ${cccl_DOWNLOAD_URL}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    OVERRIDE_FIND_PACKAGE)
endif()

# Wrap it in a function to restrict the scope of the variables
function(get_cccl)
  FetchContent_GetProperties(cccl)
  if(NOT cccl_POPULATED)
    FetchContent_MakeAvailable(cccl)
  endif()
endfunction()
get_cccl()
