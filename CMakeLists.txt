cmake_minimum_required(VERSION 3.16)
project(cs348k-final-project)

# Set up language features.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(Halide REQUIRED)

# Generator.
add_executable(portrait.generator src/portrait_generator.cpp src/benchmark.h)
target_link_libraries(portrait.generator PRIVATE Halide::Generator)

# Options are Halide::{Mullapudi2016,Li2018,Adams2019}.
if ("${PORTRAIT_SCHEDULER}" EQUAL "")
    set(PORTRAIT_SCHEDULER "Halide::Mullapudi2016")#  CACHE STRING "Scheduler to use")
endif()

message(STATUS "Using AutoScheduler: ${PORTRAIT_SCHEDULER}")

add_halide_library(portrait_gen FROM portrait.generator)
add_halide_library(portrait_gen_auto FROM portrait.generator
                   GENERATOR portrait_gen
                   AUTOSCHEDULER "${PORTRAIT_SCHEDULER}")

# Main executable.
add_executable(portrait src/portrait_main.cpp src/args.hxx src/benchmark.h)

if(NOT ("${PORTRAIT_SCHEDULER}" STREQUAL "Halide::Adams2019"))
    target_compile_definitions(portrait PUBLIC -DOUTPUT_DEPTH_MAP)
    target_compile_definitions(portrait.generator PUBLIC -DOUTPUT_DEPTH_MAP)
endif()

# TODO (rohany): Can potentially add compilation options here.
target_link_libraries(portrait
                      PRIVATE
                      Halide::Tools
                      Halide::Runtime
                      Halide::ImageIO
                      portrait_gen
                      portrait_gen_auto
                     )

