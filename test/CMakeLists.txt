
add_executable(test_xsimd test_xsimd.cpp)
target_link_libraries(test_xsimd PRIVATE Catch2::Catch2WithMain xsimd)

add_test(test_xsimd test_xsimd)
