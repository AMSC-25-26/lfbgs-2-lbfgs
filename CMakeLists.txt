cmake_minimum_required(VERSION 3.12.0)

set(CMAKE_C_COMPILER_WORKS 1)
set(CMAKE_CXX_COMPILER_WORKS 1)
project(LBFGS CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(src)
include_directories(src/utils)
include_directories(/usr/include/eigen3)


add_library(lbfgs_lib STATIC
    src/BFGS.cpp
    src/utils/LineSearch.cpp
)

add_executable(test_bfgs test/bfgs.cpp)
target_link_libraries(test_bfgs lbfgs_lib)

add_executable(test_lbfgs test/lbfgs.cpp)
target_link_libraries(test_lbfgs lbfgs_lib)

add_executable(test_gradient test/gradient.cpp)
target_link_libraries(test_gradient lbfgs_lib)

add_executable(test_linesearch test/linesearch.cpp)
target_link_libraries(test_linesearch lbfgs_lib)
