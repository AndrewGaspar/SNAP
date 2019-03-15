cmake_minimum_required(VERSION 3.10)
project(SNAP LANGUAGES CXX Fortran)

file(GLOB SOURCES src/*.f90 src/*.F90)
file(GLOB CXX_SOURCES src/*.cpp)

find_package(MPI 3.0 REQUIRED)
find_package(OpenMP REQUIRED)

add_subdirectory(deps/fargo)

add_custom_command(
    OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/tdl/cpp/include/snap-defs.hpp
        ${CMAKE_CURRENT_BINARY_DIR}/tdl/cpp/src/snap-binds.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/tdl/fortran/snap-tasks.f90
    COMMAND 
        $<TARGET_FILE:tdl> ${CMAKE_CURRENT_SOURCE_DIR}/src/snap.yml
        --output-cpp-defs
            ${CMAKE_CURRENT_BINARY_DIR}/tdl/cpp/include/snap-defs.hpp
        --output-cpp-binds
            ${CMAKE_CURRENT_BINARY_DIR}/tdl/cpp/src/snap-binds.cpp
        --output-fortran-tasks
            ${CMAKE_CURRENT_BINARY_DIR}/tdl/fortran/snap-tasks.f90
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/snap.yml
)

add_custom_target(
    snap-generate-defs
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/tdl/cpp/include/snap-defs.hpp)

add_library(
    snap-tasks
    ${CXX_SOURCES}
    ${CMAKE_CURRENT_BINARY_DIR}/tdl/cpp/src/snap-binds.cpp)
add_dependencies(snap-tasks snap-generate-defs)
target_link_libraries(
    snap-tasks
    PRIVATE OpenMP::OpenMP_CXX kokkos fargo-cpp)
target_include_directories(
    snap-tasks
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/tdl/cpp/include)

add_executable(
    snap ${SOURCES}
    ${CMAKE_CURRENT_BINARY_DIR}/tdl/fortran/snap-tasks.f90)
target_compile_definitions(snap PRIVATE MPI)
target_link_libraries(
    snap
    snap-tasks
    MPI::MPI_Fortran ffargo)