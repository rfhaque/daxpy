cmake_minimum_required(VERSION 3.10)

project(daxpy)

set(CMAKE_CXX_STANDARD 14)

add_executable(test test.cpp)

find_package(MPI REQUIRED)
target_link_libraries(test PUBLIC MPI::MPI_CXX)

if (ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_link_libraries(test PUBLIC OpenMP::OpenMP_CXX)
  add_definitions(-DUSE_OPENMP)
elseif (ENABLE_HIP)
  set(HCC_AMDGPU_TARGET gfx90a)
  set(ROCM_PATH "$ENV{ROCM_PATH}")
  SET(GCC_COVERAGE_COMPILE_FLAGS "-O3 -x hip")
  add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})
  add_definitions(-DUSE_HIP)
  include_directories(${ROCM_PATH}/include)
  include_directories(${ROCM_PATH}/llvm/lib/clang/14.0.3/include)
  include_directories(${ROCM_PATH}/hsa/include)
  include_directories(${ROCM_PATH}/hip/include)
  target_link_libraries(test PUBLIC amdhip64 -L${ROCM_PATH}/lib)
endif()

install(TARGETS test DESTINATION bin)
