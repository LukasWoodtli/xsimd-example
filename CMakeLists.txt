cmake_minimum_required(VERSION 3.18)

project(xsimd-example)

include(FetchContent)

FetchContent_Declare(xsimd
    GIT_REPOSITORY git@github.com:xtensor-stack/xsimd.git
    GIT_TAG 12.1.1
)

FetchContent_Declare(Catch2
    GIT_REPOSITORY git@github.com:catchorg/Catch2.git
    GIT_TAG v3.5.3
)

FetchContent_MakeAvailable(xsimd Catch2)

enable_testing()

add_subdirectory(test)
