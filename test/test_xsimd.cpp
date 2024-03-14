#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>

#include "xsimd/xsimd.hpp"

#include <array>

#include <iostream>


void mean(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& res) {
  std::size_t size = res.size();
  for (std::size_t i = 0; i < size; ++i) {
    res[i] = (a[i] + b[i]) / 2;
  }
}

TEST_CASE("no vectorization") {
  std::vector<float> a = { 1.5, 2.5, 3.5, 4.5, 1.5, 2.5, 3.5, 4.5, 1.5, 2.5, 3.5, 4.5 };
  std::vector<float> b = { 2.5, 3.5, 4.5, 5.5, 2.5, 3.5, 4.5, 5.5, 2.5, 3.5, 4.5, 5.5, };
  std::vector<float> res;
  res.resize(12);

  mean(a, b, res);

  REQUIRE(res.size() == 12);
  REQUIRE(res[0] == 2.0);
  REQUIRE(res[1] == 3.0);
  REQUIRE(res[2] == 4.0);
  REQUIRE(res[3] == 5.0);
  REQUIRE(res[4] == 2.0);
  REQUIRE(res[5] == 3.0);
  REQUIRE(res[6] == 4.0);
  REQUIRE(res[7] == 5.0);
  REQUIRE(res[8] == 2.0);
  REQUIRE(res[9] == 3.0);
  REQUIRE(res[10] == 4.0);
  REQUIRE(res[11] == 5.0);

  BENCHMARK("not vecorized") {
    mean(a, b, res);
  };
}

using vector_type = std::vector<float, xsimd::default_allocator<float>>;
void mean_aligned(const vector_type& a, const vector_type & b, vector_type& res) {
  using batch_type = xsimd::batch<double>;

  std::size_t inc = batch_type::size;
  std::size_t size = res.size();

  // vectorization
  const auto vec_size = size % inc;
  for (auto i = 0; i< vec_size; i += inc) {

    batch_type avec = batch_type::load_aligned(&a[i]);
    batch_type bvec = batch_type::load_aligned(&b[i]);

    batch_type rvec = (avec + bvec) / 2;

    rvec.store_aligned(&res[i]);
  }

  // remaining part (can't be vectorized)
  for (auto i = vec_size; i < size; ++i) {
    res[i] = (a[i] + b[i]) / 2;
  }
}

TEST_CASE("basic vectorization") {
  std::cout << "Best SIMD architecture: " <<xsimd::best_arch::name() << '\n';

  vector_type a = { 1.5, 2.5, 3.5, 4.5, 1.5, 2.5, 3.5, 4.5, 1.5, 2.5, 3.5, 4.5 };
  vector_type b = { 2.5, 3.5, 4.5, 5.5, 2.5, 3.5, 4.5, 5.5, 2.5, 3.5, 4.5, 5.5 };
  vector_type res;
  res.resize(12);

  mean_aligned(a, b, res);

  REQUIRE(res.size() == 12);
  REQUIRE(res[0] == 2.0);
  REQUIRE(res[1] == 3.0);
  REQUIRE(res[2] == 4.0);
  REQUIRE(res[3] == 5.0);
  REQUIRE(res[4] == 2.0);
  REQUIRE(res[5] == 3.0);
  REQUIRE(res[6] == 4.0);
  REQUIRE(res[7] == 5.0);
  REQUIRE(res[8] == 2.0);
  REQUIRE(res[9] == 3.0);
  REQUIRE(res[10] == 4.0);
  REQUIRE(res[11] == 5.0);

  BENCHMARK("vecorized") {
    mean_aligned(a, b, res);
  };
}
