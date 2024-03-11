#pragma once 

#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <tuple>
#include <functional>
#include <algorithm>
#include <utility>
#include <type_traits>

template<typename T>
using enable_if_arithmetic_t = typename std::enable_if<std::is_arithmetic<T>::value, T>::type;


template<typename T>
class Matrix {
private:
	size_t mRows, mCols;
	std::vector<T> mData;

	static constexpr T EPSILON = static_cast<T>(1e-9);

private:
	static auto isEffectivelyZero(T value) noexcept -> bool;
	static auto binomialCoefficient(std::size_t n, std::size_t k) noexcept -> T;
	static auto generateWalshMatrixRecursively(std::vector<T>& data, int order, int dim, int startRow, int startCol, T val) noexcept -> void;
	static auto checkDiagonal(const Matrix<T> mat, std::size_t i, std::size_t j) -> bool;
	static auto subtractProjection(std::vector<T>& u, const std::vector<T>& v) -> void;
	static auto normalizeVector(const std::vector<T>& v) -> std::vector<T>;

	template<typename U>
	static auto binomialFactorial(U n, U k) noexcept -> U {
		if (k > n) {
			return static_cast<U>(0);
		}
		if (k > n - k) {
			k = n - k;
		}

		U result = static_cast<U>(1);
		for (U i = 1; i <= k; ++i) {
			result = result * (n - k + i) / i;
		}
		return result;
	}

public:
	Matrix() : mRows(0), mCols(0) {}
	Matrix(size_t row, size_t col);
	Matrix(size_t row, size_t col, std::vector<T>& data);
	Matrix(size_t row, size_t col, std::initializer_list<T> data);
	Matrix(const Matrix& other) noexcept;
	Matrix(Matrix&& other) noexcept;
	
	[[nodiscard]] auto set(size_t row, size_t col, const T& value) -> bool;
	[[nodiscard]] auto get(size_t row, size_t col) const -> T;
	[[nodiscard]] auto isSquare() const noexcept -> bool;
	[[nodiscard]] auto isEqual(const Matrix& matrix) const noexcept -> bool;
	[[nodiscard]] auto isIdentity() const noexcept -> bool;
	[[nodiscard]] auto isIdempotent() const noexcept -> bool;
	[[nodiscard]] auto isRow() const noexcept -> bool;
	[[nodiscard]] auto isColumnar() const noexcept -> bool;
	[[nodiscard]] auto isSymmetric() const noexcept -> bool;
	[[nodiscard]] auto isUpperTriangular() const noexcept -> bool;
	[[nodiscard]] auto isLowerTriangular() const noexcept -> bool;
	[[nodiscard]] auto isSkewSymmetric() const noexcept -> bool;
	[[nodiscard]] auto isDiagonal() const noexcept -> bool;
	[[nodiscard]] auto isSparse() const -> bool;
	[[nodiscard]] auto isOrthogonal() const -> bool;
	[[nodiscard]] auto isHankle() const -> bool;
	[[nodiscard]] auto isToeplitz() const -> bool;
 
	auto fill(const T& value) -> void;
	auto swapRows(size_t row1, size_t row2) -> void;
	auto swapCols(size_t col1, size_t col2) -> void;
	
	auto l1Norm() const noexcept -> T;
	auto infinityNorm() const noexcept -> T;
	auto frobeniusNorm() const noexcept -> T;
	auto minElement() const -> T;
	auto maxElement() const -> T;

	auto size() const noexcept -> std::tuple<std::size_t, std::size_t>;
	auto sizeRow() const noexcept -> std::size_t;
	auto sizeCol() const noexcept -> std::size_t;
	auto trace() const-> T;
	auto determinant() const -> T;

	auto getRow(std::size_t row) const->Matrix<T>;
	auto getCol(std::size_t col) const->Matrix<T>;
	auto transpose() const->Matrix<T>;
	auto getMainDiagonalAsColumn() const->Matrix<T>;
	auto getMainDiagonalAsRow() const->Matrix<T>;
	auto getMinorDiagonalAsRow() const->Matrix<T>;
	auto getMinorDiagonalAsCol() const->Matrix<T>;
	auto adjugate() const->Matrix<T>;
	auto createSubmatrix(std::size_t excludeRow, std::size_t excludeCol) const->Matrix<T>;
	auto power(const int& power) const->Matrix<T>; 
	auto inverse() const->Matrix<T>;
	auto kroneckerProduct(const Matrix<T>& matrix) const->Matrix<T>;
	auto cofactor() const->Matrix<T>;
	auto choleskyDecomposition() const->Matrix<T>;
	auto inverseGaussJordan() const->Matrix<T>;
	auto projection() const->Matrix<T>;
	auto vandermonde(std::size_t n) const->Matrix<T>;
	auto companion(std::size_t degree) const->Matrix<T>;
	auto addRowToRow(std::size_t targetRow, std::size_t sourceRow, T scale = static_cast<T>(1)) -> bool;
	auto addColToCol(std::size_t targetCol, std::size_t sourceCol, T scale = static_cast<T>(1)) -> bool;
	auto applyToRow(std::size_t row, std::function<T(T)> func) -> bool;
	auto applyToCol(std::size_t col, std::function<T(T)> func) -> bool;
	auto luDecomposition() const->std::pair<Matrix<T>, Matrix<T>>;
	auto qrDecomposition() const->std::pair<Matrix<T>, Matrix<T>>;

	template <typename Func>
	auto map(Func func) const -> Matrix<T> {
		if (this->mData.empty()) {
			throw std::runtime_error("Matrix is Empty");
		}

		Matrix<T> result(this->mRows, this->mCols);
		std::transform(this->mData.begin(), this->mData.end(), result.mData.begin(), func);

		return result;
	}

public:
	static auto createIdentity(size_t n) noexcept ->Matrix<T>;
	static auto createHankle(const Matrix<T>& firstRow, const Matrix<T>& lastCol) ->Matrix<T>;
	static auto createToeplitz(const Matrix<T>& firstRow, const Matrix<T>& firstCol) -> Matrix<T>;
	static auto createFromArray(const std::initializer_list<T>& data, std::size_t rows, std::size_t cols) -> Matrix <T>;
	static auto createCirculant(const Matrix<T>& firstRow) -> Matrix<T>;
	static auto createHilbert(std::size_t n) -> Matrix<T>;
	static auto createHelmert(std::size_t n, bool full = true) -> Matrix<T>;
	static auto createPascal(std::size_t n) -> Matrix<T>;
	static auto createLeslie(const Matrix<T>& f, std::size_t fsize, const Matrix<T>& s, std::size_t ssize) -> Matrix<T>;
	static auto createFiedler(const Matrix<T>& matrix) -> Matrix<T>;
	static auto createInverseHilbert(std::size_t n) -> Matrix<T>;
	static auto createBlockDiagonal(const std::vector<Matrix<T>>& matrix) -> Matrix<T>;
	static auto createRandomMatrix(std::size_t rows, std::size_t cols, T start, T end) -> Matrix<T>;
	static auto createWalsh(size_t n) -> Matrix<T>;

public:
	T& operator()(size_t row, size_t col);
	const T& operator()(size_t row, size_t col) const;

	Matrix<T> operator+(const Matrix& matrix) const;
	Matrix<T> operator-(const Matrix& matrix) const;
	Matrix<T> operator*(const Matrix& matrix) const;
	Matrix<T>& operator=(const Matrix& other) noexcept;
	Matrix& operator=(Matrix&& other) noexcept;

	bool operator==(const Matrix<T>& other) const {
		if (this->mRows != other.mRows || this->mCols != other.mCols) {
			return false;
		}
		return std::equal(this->mData.begin(), this->mData.end(), other.mData.begin());
	}

	bool operator!=(const Matrix<T>& other) const {
		return !(*this == other);
	}

	bool operator<(const Matrix<T>& other) const {
		return this->frobeniusNorm() < other.frobeniusNorm();
	}

	bool operator<=(const Matrix<T>& other) const {
		return this->frobeniusNorm() <= other.frobeniusNorm();
	}

	bool operator>(const Matrix<T>& other) const {
		return this->frobeniusNorm() > other.frobeniusNorm();
	}

	bool operator>=(const Matrix<T>& other) const {
		return this->frobeniusNorm() >= other.frobeniusNorm();
	}

	friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix) {
		int max_width = 1;
		for (const T& value : matrix.mData) {
			std::stringstream ss;
			ss << std::fixed << std::setprecision(5) << value;
			int width = ss.str().length();
			max_width = std::max(max_width, width);
		}

		for (size_t row = 0; row < matrix.mRows; ++row) {
			os << "|";
			for (size_t col = 0; col < matrix.mCols; ++col) {
				size_t index = row * matrix.mCols + col;
				if (matrix.mData[index] == T{}) { // Using T{} to represent "zero" for type T
					os << std::setw((col == 0) ? 2 : max_width) << "0";
				}
				else {
					os << std::setw((col == 0) ? 0 : max_width) << std::fixed << std::setprecision(5) << matrix.mData[index];
				}
				os << " ";
			}
			os << "|\n";
		}
		return os;
	};

};
