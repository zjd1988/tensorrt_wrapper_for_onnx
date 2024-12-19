project(tensorrt_wrapper)
cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()
message("build for ${CMAKE_BUILD_TYPE}")

# prepare dependency libs
# jsoncpp
include(getjsoncpp)
set(JSONCPP_INCLUDE_DIRS ${jsoncpp_SOURCE_DIR}/include)
set(JSONCPP_LIBRARY_DIR ${jsoncpp_SOURCE_DIR}/build)
set(JSONCPP_LIBS jsoncpp::jsoncpp)

# cccl
include(getcccl)
set(CCCL_INCLUDE_DIRS ${cccl_SOURCE_DIR}/cub)

# opencv
find_package(OpenCV REQUIRED)

# cuda
find_package(CUDAToolkit REQUIRED)
set(CUDA_VERSION ${CUDAToolkit_VERSION} CACHE STRING "")
list(APPEND CUDAToolkit_LIBS CUDA::cublas)
list(APPEND CUDAToolkit_LIBS CUDA::curand)
list(APPEND CUDAToolkit_LIBS CUDA::cusolver)
list(APPEND CUDAToolkit_LIBS CUDA::cufft)
if(CUDA_VERSION VERSION_GREATER_EQUAL "10.1")
    list(APPEND CUDAToolkit_LIBS CUDA::cublasLt)
endif()
if(CUDA_VERSION VERSION_GREATER_EQUAL "10.2")
    list(APPEND CUDAToolkit_LIBS CUDA::nvjpeg)
    list(APPEND CUDAToolkit_LIBS CUDA::nppc)
    list(APPEND CUDAToolkit_LIBS CUDA::nppig)
endif()

# tensorrt
find_package(TENSORRT REQUIRED)

message(STATUS "------------------------------------------------------")
message(STATUS "jsoncpp include path           : ${JSONCPP_INCLUDE_DIRS}")
message(STATUS "jsoncpp library path           : ${JSONCPP_LIBRARY_DIR}")
message(STATUS "jsoncpp libs                   : ${JSONCPP_LIBS}")
message(STATUS "------------------------------------------------------")
message(STATUS "cccl include path              : ${CCCL_INCLUDE_DIRS}")
message(STATUS "------------------------------------------------------")
message(STATUS "opencv include path            : ${OpenCV_INCLUDE_DIRS}")
message(STATUS "opencv library path            : ${OpenCV_LIBRARY_DIR}")
message(STATUS "opencv libs                    : ${OpenCV_LIBS}")
message(STATUS "------------------------------------------------------")
message(STATUS "cuda toolkit version           : ${CUDAToolkit_VERSION}")
message(STATUS "cuda toolkit include path      : ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "cuda toolkit library path      : ${CUDAToolkit_LIBRARY_DIR}")
message(STATUS "cuda toolkit libs              : ${CUDAToolkit_LIBS}")
message(STATUS "------------------------------------------------------")
message(STATUS "TensorRT Version               : ${TENSORRT_VERSION}")
message(STATUS "TensorRT Root(optional)        : ${TENSORRT_ROOT_DIR}")
message(STATUS "TensorRT include path          : ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT Libs                  : ${TENSORRT_LIBS}")

set(PROJECT_INCLUDE_DIRS ${JSONCPP_INCLUDE_DIRS}
                         ${CCCL_INCLUDE_DIRS}
                         ${OpenCV_INCLUDE_DIRS}
                         ${CUDAToolkit_INCLUDE_DIRS}
                         ${TENSORRT_INCLUDE_DIR}
                         ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(PROJECT_LINK_DIRS ${JSONCPP_LIBRARY_DIR}
                      ${OpenCV_LIBRARY_DIR}
                      ${CUDAToolkit_LIBRARY_DIR}
                      ${TENSORRT_LIBRARY_DIR})

file(GLOB COMMON_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp)
file(GLOB NODE_CREATE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/node_create/*.cpp)
file(GLOB NODE_INFO_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/node_info/*.cpp)
file(GLOB EXECUTION_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/execution_info/*.c*)
file(GLOB ENGINE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/infer_engine/*.cpp)
file(GLOB WARPPER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

cuda_add_library(tensorrtWrapper SHARED ${COMMON_FILES} 
                                        ${NODE_CREATE_FILES} 
                                        ${NODE_INFO_FILES} 
                                        ${EXECUTION_FILES} 
                                        ${ENGINE_FILES}
                                        ${WARPPER_FILES})
target_include_directories(tensorrtWrapper PRIVATE ${PROJECT_INCLUDE_DIRS})
target_link_directories(tensorrtWrapper PRIVATE ${PROJECT_LINK_DIRS})
target_link_libraries(tensorrtWrapper PRIVATE ${JSONCPP_LIBS}
                                              ${OpenCV_LIBS}
                                              ${CUDAToolkit_LIBS}
                                              ${TENSORRT_LIBS})

# file(GLOB EXAMPLE_FILES ${PROJECT_BINARY_DIR}/../example/*.cpp)
# foreach(example_file IN LISTS EXAMPLE_FILES)
#     get_filename_component(target_file ${example_file} NAME_WE)
#     cuda_add_executable(${target_file} ${example_file})
#     target_link_libraries(${target_file} tensorrtWrapper)
#     target_link_libraries(${target_file} jsoncpp::jsoncpp)
#     target_link_libraries(${target_file} nvinfer)
#     target_link_libraries(${target_file} cudart)
#     target_link_libraries(${target_file} cusolver)
#     target_link_libraries(${target_file} ${OpenCV_LIBS})
# endforeach(example_file)

