// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Stefano Cacciatore
#ifndef FASTPLS_CORE_MATRIX_HPP
#define FASTPLS_CORE_MATRIX_HPP

#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace fastpls {
namespace core {

template<class T>
class BasicMatrixView {
 public:
  using value_type = typename std::remove_const<T>::type;
  using pointer = T*;
  using reference = T&;

  constexpr BasicMatrixView() noexcept = default;

  constexpr BasicMatrixView(pointer data, std::size_t rows,
                            std::size_t columns,
                            std::size_t leading_dimension) noexcept
      : data_(data), rows_(rows), columns_(columns),
        leading_dimension_(leading_dimension) {}

  template<class U,
           typename std::enable_if<
             std::is_const<T>::value &&
             std::is_same<U, value_type>::value,
             int
           >::type = 0>
  constexpr BasicMatrixView(const BasicMatrixView<U>& other) noexcept
      : data_(other.data()), rows_(other.rows()), columns_(other.columns()),
        leading_dimension_(other.leading_dimension()) {}

  constexpr pointer data() const noexcept { return data_; }
  constexpr std::size_t rows() const noexcept { return rows_; }
  constexpr std::size_t columns() const noexcept { return columns_; }
  constexpr std::size_t leading_dimension() const noexcept {
    return leading_dimension_;
  }
  constexpr bool empty() const noexcept {
    return rows_ == 0 || columns_ == 0;
  }
  constexpr bool contiguous() const noexcept {
    return leading_dimension_ == rows_;
  }

  reference operator()(std::size_t row, std::size_t column) const {
    if (row >= rows_ || column >= columns_) {
      throw std::out_of_range("fastPLS matrix-view index is out of range");
    }
    return data_[row + column * leading_dimension_];
  }

 private:
  pointer data_ = nullptr;
  std::size_t rows_ = 0;
  std::size_t columns_ = 0;
  std::size_t leading_dimension_ = 0;
};

template<class T>
using MatrixView = BasicMatrixView<T>;

template<class T>
using ConstMatrixView = BasicMatrixView<const T>;

template<class T>
class Matrix {
 public:
  Matrix() = default;

  Matrix(std::size_t rows, std::size_t columns)
      : values_(rows * columns), rows_(rows), columns_(columns) {}

  void resize(std::size_t rows, std::size_t columns) {
    values_.resize(rows * columns);
    rows_ = rows;
    columns_ = columns;
  }

  T* data() noexcept { return values_.data(); }
  const T* data() const noexcept { return values_.data(); }
  std::size_t rows() const noexcept { return rows_; }
  std::size_t columns() const noexcept { return columns_; }
  std::size_t size() const noexcept { return values_.size(); }

  MatrixView<T> view() noexcept {
    return MatrixView<T>(data(), rows_, columns_, rows_);
  }

  ConstMatrixView<T> view() const noexcept {
    return ConstMatrixView<T>(data(), rows_, columns_, rows_);
  }

  T& operator()(std::size_t row, std::size_t column) {
    return view()(row, column);
  }

  const T& operator()(std::size_t row, std::size_t column) const {
    return view()(row, column);
  }

 private:
  std::vector<T> values_;
  std::size_t rows_ = 0;
  std::size_t columns_ = 0;
};

template<class T>
ConstMatrixView<T> make_const_view(const T* data, std::size_t rows,
                                   std::size_t columns,
                                   std::size_t leading_dimension) noexcept {
  return ConstMatrixView<T>(data, rows, columns, leading_dimension);
}

template<class T>
MatrixView<T> make_view(T* data, std::size_t rows, std::size_t columns,
                        std::size_t leading_dimension) noexcept {
  return MatrixView<T>(data, rows, columns, leading_dimension);
}

}  // namespace core
}  // namespace fastpls

#endif
