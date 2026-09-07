# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.18)
project(fastpls_consumer LANGUAGES CXX)
find_package(fastpls_native CONFIG REQUIRED)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE fastpls::native)
enable_testing()
add_test(NAME installed_consumer COMMAND consumer)
