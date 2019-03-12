cmake_minimum_required(VERSION 3.10)
project(SNAP LANGUAGES Fortran)

file(GLOB SOURCES src/*.f90 src/*.F90)

add_executable(snap ${SOURCES})