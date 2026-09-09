# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.18)
project(fastpls_core VERSION 0.0.1 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)

add_library(fastpls_core INTERFACE)
add_library(fastpls::core ALIAS fastpls_core)
set_target_properties(fastpls_core PROPERTIES EXPORT_NAME core)
target_include_directories(fastpls_core INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../inst/include>"
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>")
target_compile_features(fastpls_core INTERFACE cxx_std_17)

install(TARGETS fastpls_core EXPORT fastplsCoreTargets)
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../inst/include/fastpls/core"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fastpls"
  FILES_MATCHING PATTERN "*.hpp")
install(FILES "${CMAKE_CURRENT_LIST_DIR}/../inst/include/fastpls/core.hpp"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/fastpls")
set(FASTPLS_CONFIG_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/fastpls_core")
configure_package_config_file(cmake/fastpls_coreConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/fastpls_coreConfig.cmake"
  INSTALL_DESTINATION "${FASTPLS_CONFIG_DIR}")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/fastpls_coreConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}" COMPATIBILITY SameMajorVersion)
install(EXPORT fastplsCoreTargets NAMESPACE fastpls::
  DESTINATION "${FASTPLS_CONFIG_DIR}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/fastpls_coreConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/fastpls_coreConfigVersion.cmake"
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

endif()
