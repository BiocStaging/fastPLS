# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.18)
project(fastpls_core_consumer LANGUAGES CXX)
find_package(fastpls_core CONFIG REQUIRED)
add_executable(core_consumer main.cpp)
target_link_libraries(core_consumer PRIVATE fastpls::core)
enable_testing()
add_test(NAME installed_core_consumer COMMAND core_consumer)
