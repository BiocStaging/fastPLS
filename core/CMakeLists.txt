# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.18)
project(fastpls_native VERSION 0.0.1 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

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
target_link_libraries(fastpls_native INTERFACE LAPACK::LAPACK BLAS::BLAS)

install(TARGETS fastpls_native EXPORT fastplsNativeTargets)
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../inst/include/fastpls/native"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fastpls"
  FILES_MATCHING PATTERN "*.hpp")
install(FILES "${CMAKE_CURRENT_LIST_DIR}/../inst/include/fastpls/native/LICENSE"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/licenses/fastpls_native")
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
add_executable(test_native_rsvd tests/test_rsvd.cpp)
target_link_libraries(test_native_rsvd PRIVATE fastpls_native)
add_test(NAME native_rsvd COMMAND test_native_rsvd)

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
add_executable(test_native_models tests/test_models.cpp)
target_link_libraries(test_native_models PRIVATE fastpls_native)
add_test(NAME native_models COMMAND test_native_models)
add_executable(test_native_lda tests/test_lda.cpp)
target_link_libraries(test_native_lda PRIVATE fastpls_native)
add_test(NAME native_lda COMMAND test_native_lda)
endif()
