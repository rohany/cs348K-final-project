cmake_minimum_required(VERSION 3.16)
project(cs348k-final-project)

# Set up language features.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(Halide REQUIRED)

# Generator.
add_executable(portrait.generator src/portrait_generator.cpp)
target_link_libraries(portrait.generator PRIVATE Halide::Generator)

add_halide_library(portrait_gen FROM portrait.generator)

# Main executable.
add_executable(portrait src/portrait_main.cpp src/args.hxx)
# TODO (rohany): Can potentially add compilation options here.
# target_compile_options(portrait)
target_link_libraries(portrait
                      PRIVATE
                      Halide::Tools
                      Halide::Runtime
                      Halide::ImageIO
                      portrait_gen
                     )
