cmake_minimum_required(VERSION 3.10)
project(SNAP LANGUAGES Fortran)

file(GLOB SOURCES src/*.f90 src/*.F90)

find_package(MPI 3.0 REQUIRED)

add_executable(snap ${SOURCES})
target_link_libraries(snap MPI::MPI_Fortran)
target_compile_definitions(snap PRIVATE MPI)