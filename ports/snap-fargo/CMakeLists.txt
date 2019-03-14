cmake_minimum_required(VERSION 3.10)
project(SNAP LANGUAGES CXX Fortran)

file(GLOB SOURCES src/*.f90 src/*.F90)

find_package(MPI 3.0 REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(snap ${SOURCES})
target_link_libraries(snap MPI::MPI_Fortran OpenMP::OpenMP_CXX kokkos)
target_compile_definitions(snap PRIVATE MPI)