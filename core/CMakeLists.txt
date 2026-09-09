# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.18)
project(fastpls_native VERSION 0.0.1 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

option(FASTPLS_BUILD_TRANSITIONAL_NATIVE
  "Build the transitional Armadillo numerical layer" ON)

add_library(fastpls_core INTERFACE)
add_library(fastpls::core ALIAS fastpls_core)
set_target_properties(fastpls_core PROPERTIES EXPORT_NAME core)
target_include_directories(fastpls_core INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../inst/include>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_compile_features(fastpls_core INTERFACE cxx_std_17)

if(FASTPLS_BUILD_TRANSITIONAL_NATIVE)
  find_path(ARMADILLO_INCLUDE_DIR armadillo REQUIRED)
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
  add_library(fastpls_native INTERFACE)
  add_library(fastpls::native ALIAS fastpls_native)
  set_target_properties(fastpls_native PROPERTIES EXPORT_NAME native)
  target_include_directories(fastpls_native INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../inst/include>"
    "$<BUILD_INTERFACE:${ARMADILLO_INCLUDE_DIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
  target_compile_features(fastpls_native INTERFACE cxx_std_17)
  target_compile_definitions(fastpls_native INTERFACE ARMA_DONT_USE_WRAPPER)
  target_link_libraries(fastpls_native INTERFACE
    fastpls_core LAPACK::LAPACK BLAS::BLAS)
endif()

install(TARGETS fastpls_core EXPORT fastplsNativeTargets)
if(FASTPLS_BUILD_TRANSITIONAL_NATIVE)
  install(TARGETS fastpls_native EXPORT fastplsNativeTargets)
endif()
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../inst/include/fastpls/core"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fastpls"
  FILES_MATCHING PATTERN "*.hpp")
install(FILES "${CMAKE_CURRENT_LIST_DIR}/../inst/include/fastpls/core.hpp"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fastpls")
if(FASTPLS_BUILD_TRANSITIONAL_NATIVE)
  install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../inst/include/fastpls/native"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fastpls"
    FILES_MATCHING PATTERN "*.hpp")
  install(FILES "${CMAKE_CURRENT_LIST_DIR}/../inst/include/fastpls/native/LICENSE"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/fastpls_native")
endif()
set(FASTPLS_CONFIG_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/fastpls_native")
configure_package_config_file(cmake/fastpls_nativeConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/fastpls_nativeConfig.cmake"
  INSTALL_DESTINATION "${FASTPLS_CONFIG_DIR}")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/fastpls_nativeConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}" COMPATIBILITY SameMajorVersion)
install(EXPORT fastplsNativeTargets NAMESPACE fastpls::
  DESTINATION "${FASTPLS_CONFIG_DIR}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/fastpls_nativeConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/fastpls_nativeConfigVersion.cmake"
  DESTINATION "${FASTPLS_CONFIG_DIR}")

if(BUILD_TESTING)
  add_executable(test_core_classification tests/test_core_classification.cpp)
  target_link_libraries(test_core_classification PRIVATE fastpls_core)
  add_test(NAME core_classification COMMAND test_core_classification)

  add_executable(test_core_linalg tests/test_core_linalg.cpp)
  target_link_libraries(test_core_linalg PRIVATE fastpls_core)
  add_test(NAME core_linalg COMMAND test_core_linalg)

  add_executable(test_core_operators tests/test_core_operators.cpp)
  target_link_libraries(test_core_operators PRIVATE fastpls_core)
  add_test(NAME core_operators COMMAND test_core_operators)

  add_executable(test_core_operator_rsvd tests/test_core_operator_rsvd.cpp)
  target_link_libraries(test_core_operator_rsvd PRIVATE fastpls_core)
  add_test(NAME core_operator_rsvd COMMAND test_core_operator_rsvd)

  add_executable(test_core_simpls tests/test_core_simpls.cpp)
  target_link_libraries(test_core_simpls PRIVATE fastpls_core)
  add_test(NAME core_simpls COMMAND test_core_simpls)

  add_executable(test_core_plssvd tests/test_core_plssvd.cpp)
  target_link_libraries(test_core_plssvd PRIVATE fastpls_core)
  add_test(NAME core_plssvd COMMAND test_core_plssvd)

  add_executable(test_core_kernels tests/test_core_kernels.cpp)
  target_link_libraries(test_core_kernels PRIVATE fastpls_core)
  add_test(NAME core_kernels COMMAND test_core_kernels)

  add_executable(test_core_kernelpls tests/test_core_kernelpls.cpp)
  target_link_libraries(test_core_kernelpls PRIVATE fastpls_core)
  add_test(NAME core_kernelpls COMMAND test_core_kernelpls)

  add_executable(test_core_lda tests/test_core_lda.cpp)
  target_link_libraries(test_core_lda PRIVATE fastpls_core)
  add_test(NAME core_lda COMMAND test_core_lda)

  add_executable(test_core_statistics tests/test_core_statistics.cpp)
  target_link_libraries(test_core_statistics PRIVATE fastpls_core)
  add_test(NAME core_statistics COMMAND test_core_statistics)

  add_executable(test_core_supervised tests/test_core_supervised.cpp)
  target_link_libraries(test_core_supervised PRIVATE fastpls_core)
  add_test(NAME core_supervised COMMAND test_core_supervised)

  add_executable(test_core_opls tests/test_core_opls.cpp)
  target_link_libraries(test_core_opls PRIVATE fastpls_core)
  add_test(NAME core_opls COMMAND test_core_opls)

  add_executable(test_core_oplspls tests/test_core_oplspls.cpp)
  target_link_libraries(test_core_oplspls PRIVATE fastpls_core)
  add_test(NAME core_oplspls COMMAND test_core_oplspls)

  if(FASTPLS_BUILD_TRANSITIONAL_NATIVE)
    add_executable(test_native_simpls tests/test_simpls.cpp)
    target_link_libraries(test_native_simpls PRIVATE fastpls_native)
    add_test(NAME native_simpls COMMAND test_native_simpls)

    add_executable(test_native_plssvd tests/test_plssvd.cpp)
    target_link_libraries(test_native_plssvd PRIVATE fastpls_native)
    add_test(NAME native_plssvd COMMAND test_native_plssvd)

    add_executable(test_native_operators tests/test_operators.cpp)
    target_link_libraries(test_native_operators PRIVATE fastpls_native)
    add_test(NAME native_operators COMMAND test_native_operators)

    add_executable(test_native_recovery tests/test_recovery.cpp)
    target_link_libraries(test_native_recovery PRIVATE fastpls_native)
    add_test(NAME native_recovery COMMAND test_native_recovery)
    add_executable(test_native_input_contract tests/test_input_contract.cpp)
    target_link_libraries(test_native_input_contract PRIVATE fastpls_native)
    add_test(NAME native_input_contract COMMAND test_native_input_contract)
    add_executable(test_native_opls tests/test_opls.cpp)
    target_link_libraries(test_native_opls PRIVATE fastpls_native)
    add_test(NAME native_opls COMMAND test_native_opls)
    add_executable(test_native_kernels tests/test_kernels.cpp)
    target_link_libraries(test_native_kernels PRIVATE fastpls_native)
    add_test(NAME native_kernels COMMAND test_native_kernels)
  endif()
endif()
