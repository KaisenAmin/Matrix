#include "matrix.hpp"

#include <iostream>
#include <iterator>
#include <iomanip>
#include <string>
#include <cmath>
#include <sstream>
#include <numeric>
#include <random>
#include <chrono>


template<typename T>
auto Matrix<T>::isEffectivelyZero(T value) noexcept -> bool {
	return std::fabs(value) < EPSILON;
}

template<typename T>
auto Matrix<T>::binomialCoefficient(std::size_t n, std::size_t k) noexcept -> T{
	std::vector<T> C(k + 1, 0);
	C[0] = 1; 

	for (std::size_t i = 1; i <= n; ++i) {
		for (int j = std::min(i, k); j > 0; --j) { // Cast to int to prevent underflow.
			C[j] = C[j] + C[j - 1];
		}
	}
	return C[k];
}

template<typename T>
auto Matrix<T>::generateWalshMatrixRecursively(std::vector<T>& data, int order, int dim, int startRow, int startCol, T val) noexcept -> void {
	if (order == 1) {
		data[startRow * dim + startCol] = val;
		return;
	}

	int halfOrder = order / 2;
	// Top-left quadrant
	generateWalshMatrixRecursively(data, halfOrder, dim, startRow, startCol, val);
	// Top-right quadrant
	generateWalshMatrixRecursively(data, halfOrder, dim, startRow, startCol + halfOrder, val);
	// Bottom-left quadrant
	generateWalshMatrixRecursively(data, halfOrder, dim, startRow + halfOrder, startCol, val);
	// Bottom-right quadrant (invert the values)
	generateWalshMatrixRecursively(data, halfOrder, dim, startRow + halfOrder, startCol + halfOrder, -val);
}

template <typename T>
auto Matrix<T>::checkDiagonal(const Matrix<T> mat, std::size_t i, std::size_t j) -> bool{
	T res = mat.get(i, j);
	
	while (++i < mat.sizeRow() && ++j < mat.sizeCol()) {
		if (mat.get(i, j) != res) {
			return false;
		}
	}

	return true;
}

template<typename T>
auto Matrix<T>::subtractProjection(std::vector<T>& u, const std::vector<T>& v) -> void {
	T dot_uv = std::inner_product(u.begin(), u.end(), v.begin(), T{ 0 });
	T dot_vv = std::inner_product(v.begin(), v.end(), v.begin(), T{ 0 });
	T scale = dot_uv / dot_vv;

	for (size_t i = 0; i < u.size(); ++i) {
		u[i] -= scale * v[i];
	}
}

template<typename T>
auto Matrix<T>::normalizeVector(const std::vector<T>& v) -> std::vector<T> {
	T norm = std::sqrt(std::accumulate(v.begin(), v.end(), T(0), [](T a, T b) { return a + b * b; }));
	std::vector<T> normalized(v.size());
	std::transform(v.begin(), v.end(), normalized.begin(), [norm](T val) { return val / norm; });
	return normalized;
}

template<typename T>
Matrix<T>::Matrix(size_t row, size_t col) : mRows(row), mCols(col), mData(row* col) {}

template<typename T>
Matrix<T>::Matrix(size_t row, size_t col, std::vector<T>& data) : mRows(row), mCols(col), mData(data) {
	if (data.size() != row * col) {
		throw std::invalid_argument("Data does not match matrix dimension.");
	}
}

template<typename T>
Matrix<T>::Matrix(size_t row, size_t col, std::initializer_list<T> data) : mRows(row), mCols(col), mData(data) {
	if (data.size() != row * col) {
		throw std::invalid_argument("Data does not match matrix dimension.");
	}
}

template<typename T>
Matrix<T>::Matrix(const Matrix& other) noexcept : mRows(other.mRows), mCols(other.mCols), mData(other.mData) {}

template<typename T>
Matrix<T>::Matrix(Matrix&& other) noexcept : mRows(std::exchange(other.mRows, 0)), mCols(std::exchange(other.mCols, 0)), mData(std::move(other.mData)) {}

template<typename T>
[[nodiscard]] auto Matrix<T>::set(size_t row, size_t col, const T& value) -> bool {
	if (row >= this->mRows || col >= this->mCols) {
		throw std::invalid_argument("Error : rows or cols or both of them are bigger than matrix rows and cols");
	}

	size_t index = this->mCols * row;
	this->mData[index + col] = value;

	return true;
}

template<typename T>
[[nodiscard]] auto Matrix<T>::get(size_t row, size_t col) const -> T {
	if (row >= this->mRows || col >= this->mCols) {
		throw std::invalid_argument("Error : rows or cols or both of them are bigger than matrix rows and cols");
	}

	size_t index = row * this->mCols + col;
	return this->mData[index];
}

template<typename T>
[[nodiscard]] auto Matrix<T>::isSquare() const noexcept -> bool {
	if (this->mRows == this->mCols) {
		return true;
	}
	return false;
}

template<typename T>
[[nodiscard]] auto Matrix<T>::isEqual(const Matrix& matrix) const noexcept -> bool {
	if (this->mRows != matrix.mRows || this->mCols != matrix.mCols) {
		return false;
	}
	if (this->mData == matrix.mData) {
		return true;
	}

	return false;
}

template<typename T>
[[nodiscard]] auto Matrix<T>::isIdentity() const noexcept -> bool {
	if (!this->isSquare()) {
		return false;
	}
	for (size_t i = 0; i < this->mRows; ++i) {
		for (size_t j = 0; j < this->mCols; ++j) {
			if (i == j) {
				// Check diagonal elements for 1
				if (this->mData[i * this->mCols + j] != 1) {
					return false;
				}
			}
			else {
				// Check off-diagonal elements for 0
				if (this->mData[i * this->mCols + j] != 0) {
					return false;
				}
			}
		}
	}

	return true;
}

template<typename T>
[[nodiscard]] auto Matrix<T>::isIdempotent() const noexcept -> bool {
	if (!this->isSquare()) {
		return false;
	}
	Matrix<T> result = *this * *this;

	return this->isEqual(result);
}

template<typename T>
[[nodiscard]] inline auto Matrix<T>::isRow() const noexcept -> bool {
	return this->mRows == 1 ? true : false;
}

template<typename T>
[[nodiscard]] inline auto Matrix<T>::isColumnar() const noexcept -> bool {
	return this->mCols == 1? true : false;
}

template<typename T>
[[nodiscard]] auto Matrix<T>::isSymmetric() const noexcept -> bool {
	if (!this->isSquare()) {
		return false;
	}

	for (size_t i = 0; i < this->mRows; i++) {
		for (size_t j = i + 1; j < this->mCols; j++) { // Start from j = i + 1 to skip the diagonal
			if (this->mData[i * this->mCols + j] != this->mData[j * this->mCols + i]) {
				return false;
			}
		}
	}

	return true;
}

template<typename T>
[[nodiscard]] auto Matrix<T>::isUpperTriangular() const noexcept -> bool {
	if (!this->isSquare()) {
		return false;
	}

	for (size_t i = 0; i < this->mRows; i++) {
		for (size_t j = 0; j < i; j++) {
			if (this->mData[i * this->mCols + j] != 0) {
				return false;
			}
		}
	}

	return true;
}

template<typename T>
[[nodiscard]] auto Matrix<T>::isLowerTriangular() const noexcept -> bool {
	if (!this->isSquare()) {
		return false;
	}
	
	for (size_t i = 0; i < this->mRows; i++) {
		for (size_t j = i + 1; j < this->mCols; j++) {
			if (this->mData[i * this->mCols + j] != 0) {
				return false;
			}
		}
	}

	return true;
}

template<typename T>
[[nodiscard]] auto Matrix<T>::isSkewSymmetric() const noexcept-> bool {
	if (!this->isSquare()) {
		return false;
	}

	auto isDiagonalZero = [this](size_t i) {
		return isEffectivelyZero(this->mData[i * this->mCols + i]);
	};

	auto isOffDiagonalNegatives = [this](size_t i, size_t j) {
		return this->mData[i * this->mCols + j] == -this->mData[j * this->mCols + i];
	};

	for (size_t i = 0; i < this->mRows; i++) {
		if (!isDiagonalZero(i)) {
			return false;
		}

		for (size_t j = i + 1; j < this->mCols; j++) {
			if (!isOffDiagonalNegatives(i, j)) {
				return false;
			}
		}
	}
	return true;
}

template<typename T>
[[nodiscard]] auto Matrix<T>::isDiagonal() const noexcept -> bool {
	if (!this->isSquare()) {
		return false;
	}

	auto isOffDiagonalZero = [this](size_t i, size_t j) {
		return i != j ? isEffectivelyZero(this->mData[i * this->mCols + j]): true;
	};

	for (size_t i = 0; i < this->mRows; i++) {
		for (size_t j = 0; j < this->mCols; j++) {
			if (!isOffDiagonalZero(i, j)) {
				return false;
			}
		}
	}

	return true;
}

template <typename T>
auto Matrix<T>::isSparse() const -> bool {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is Empty");
	}

	auto nonZeroCount = std::count_if(this->mData.begin(), this->mData.end(), [](const T& element) {
		return element != 0;
	});

	std::size_t totalElements = this->mRows * this->mCols;
	double nonZeroPercentage = static_cast<double>(nonZeroCount) / static_cast<double>(totalElements);

	return nonZeroPercentage < 0.3;
}

template<typename T>
auto Matrix<T>::isOrthogonal() const -> bool {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is Empty");
	}
	else if (!this->isSquare()) {
		throw std::runtime_error("Matrix is not Square");
	}

	Matrix<T> transposeMatrix = this->transpose();
	Matrix<T> productMatrix = *this * transposeMatrix;
	bool isOrthogonal = productMatrix.isIdentity();

	return isOrthogonal;
}

template<typename T>
auto Matrix<T>::isHankle() const -> bool {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is Empty");
	}

	for (std::size_t i = 0; i < this->mRows; i++) {
		for (std::size_t j = 0; j < this->mCols; j++) {
			if (i + j > this->mRows - 1) {
				continue;
			}
			T value = this->get(i, j);
			
			if (i + 1 < this->mRows && j > 0) {
				T next = this->get(i + 1, j - 1);
				if (!isEffectivelyZero(value - next)) {
					return false;
				}
			}
		}
	}

	return true;
}

template<typename T>
auto Matrix<T>::isToeplitz() const -> bool {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is Empty");
	}

	for (std::size_t i = 0; i < this->mCols; i++) {
		if (checkDiagonal(*this, 0, i)) {
			return false;
		}
	}

	for (std::size_t i = 1; i < this->mRows; i++) {
		if (checkDiagonal(*this, i, 0)) {
			return false;
		}
	}

	return true;
}

template<typename T>
auto Matrix<T>::fill(const T& value) -> void {
	std::fill(this->mData.begin(), this->mData.end(), value);
}

template<typename T>
auto Matrix<T>::swapRows(size_t row1, size_t row2) -> void {
	if (row1 >= this->mRows || row2 >= this->mRows) {
		throw std::invalid_argument("Invalid row indices");
	}

	for (size_t i = 0; i < mCols; ++i) {
		std::swap(mData[row1 * mCols + i], mData[row2 * mCols + i]);
	}
}

template<typename T>
void Matrix<T>::swapCols(size_t col1, size_t col2) {
	if (col1 >= mCols || col2 >= mCols) {
		throw std::invalid_argument("Invalid column indices");
	}

	for (size_t i = 0; i < mRows; ++i) {
		std::swap(mData[i * mCols + col1], mData[i * mCols + col2]);
	}
}

template<typename T>
auto Matrix<T>::l1Norm() const noexcept -> T {
	T maxSum = 0;

	for (size_t j = 0; j < this->mCols; j++) {
		T columnSum = 0;

		for (size_t i = 0; i < this->mRows; i++) {
			columnSum += std::fabs(this->get(i, j));
		}

		maxSum = std::max(maxSum, columnSum);
	}

	return maxSum;
}

template <typename T>
auto Matrix<T>::infinityNorm() const noexcept -> T {
	T maxSum = 0;

	for (size_t i = 0; i < this->mRows; ++i) {
		T rowSum = std::accumulate(this->mData.begin() + i * this->mCols, this->mData.begin() + (i + 1) * this->mCols, T(0),
			[](const T& total, const T& value) { return total + std::fabs(value); });

		maxSum = std::max(maxSum, rowSum);
	}

	return maxSum;
}

template<typename T>
auto Matrix<T>::frobeniusNorm() const noexcept -> T {
	return std::sqrt(std::accumulate(this->mData.begin(), this->mData.end(), 0.0, [](T sum, const T& value) {
		return sum + value * value;
	}));
}

template<typename T>
auto Matrix<T>::minElement() const -> T {
	if (!this->mData.empty()) {
		return *std::min_element(this->mData.begin(), this->mData.end());
	}
	else {
		throw std::runtime_error("Matrix is empty");
	}
}

template<typename T>
auto Matrix<T>::maxElement() const -> T {
	if (!this->mData.empty()) {
		return *std::max_element(this->mData.begin(), this->mData.end());
	}
	else {
		throw std::runtime_error("Matrix is empty");
	}
}

template<typename T>
auto Matrix<T>::size() const noexcept -> std::tuple<std::size_t, std::size_t> {
	return std::tuple<std::size_t, std::size_t> {this->mRows, this->mCols};
}

template<typename T>
auto Matrix<T>::trace() const -> T {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is empty");
	}
	if (!this->isSquare()) {
		throw std::runtime_error("Matrix is not Square");
	}

	T trace = T{}; 
	for (size_t i = 0; i < this->mRows; ++i) {
		trace += this->mData[i * this->mCols + i];
	}

	return trace;
}

template<typename T>
auto Matrix<T>::determinant() const -> T {
	if (!this->isSquare()) {
		throw std::runtime_error("Derminant can only be calculated for Square matrix.");
	}
	else if (this->mRows == 1) {
		return this->mData[0];
	}
	else if (this->mRows == 2) {
		return this->mData[0] * this->mData[3] - this->mData[1] * this->mData[2];
	} 
	else {
		T det{};

		for (std::size_t j1 = 0; j1 < this->mCols; j1++) {
			Matrix<T> submatrix(this->mRows - 1, this->mCols - 1);

			for (std::size_t i = 1; i < this->mRows; i++) {
				std::size_t j2 = 0;
				for (std::size_t j = 0; j < this->mCols; j++) {
					if (j == j1) {
						continue;
					}
					submatrix(i - 1, j2++) = this->get(i, j);
				}
			}
			det += static_cast<T>((j1 % 2 == 0 ? 1 : -1) * this->get(0, j1) * submatrix.determinant());
		}

		return det;
	}
}

template<typename T>
auto Matrix<T>::getRow(std::size_t row) const -> Matrix<T> {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is empty");
	}
	else if (row >= this->mRows) {
		throw std::runtime_error("Chosen row is bigger than Matrix Row");
	}

	Matrix<T> r(1, this->mCols);

	for (std::size_t j = 0; j < this->mCols; j++) {
		if (r.set(0, j, static_cast<T>(this->get(row, j)))) {
			continue;
		}
		else {
			throw std::runtime_error("Can not set value in Matrix Row");
		}
	}

	return r;
}

template<typename T>
auto Matrix<T>::getCol(std::size_t col) const -> Matrix<T> {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is empty");
	}
	else if (col >= this->mCols) {
		throw std::runtime_error("Chosen col is bigger than Matrix Col");
	}

	Matrix<T> c(this->mRows, 1);

	for (std::size_t i = 0; i < this->mRows; i++) {
		if (c.set(i, 0, static_cast<T>(this->get(i, col)))) {
			continue;
		}
		else {
			throw std::runtime_error("Can not set value in Matrix Col");
		}
	}

	return c;
}

template<typename T>
auto Matrix<T>::transpose() const -> Matrix<T> {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is Empty");
	}
	
	Matrix<T> transpose(this->mCols, this->mRows);

	for (std::size_t i = 0; i < this->mRows; i++) {
		for (std::size_t j = 0; j < this->mCols; j++) {
			transpose(j, i) = this->get(i, j); 
		}
	}

	return transpose;
}

template<typename T>
auto Matrix<T>::getMainDiagonalAsColumn() const -> Matrix<T> {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is Empty");
	}
	else if (!this->isSquare()) {
		throw std::runtime_error("Mattrix is not square");
	}
	Matrix<T> diagonalMatrix(this->mRows, 1);

	for (std::size_t i = 0; i < this->mRows; i++) {
		if (diagonalMatrix.set(i, 0, static_cast<T>(this->mData[i * this->mCols + i]))) {
			continue;
		}
		else {
			throw std::runtime_error("Can not set value in Matrix diagonal");
		}
	}

	return diagonalMatrix;
}

template<typename T>
auto Matrix<T>::getMainDiagonalAsRow() const -> Matrix<T> {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is Empty");
	}
	else if (!this->isSquare()) {
		throw std::runtime_error("Mattrix is not square");
	}
	Matrix<T> diagoanlMatrix(1, this->mCols);

	for (std::size_t i = 0; i < this->mCols; i++) {
		if (diagoanlMatrix.set(0, i, static_cast<T>(this->mData[i * this->mCols + i]))) {
			continue;
		}
		else {
			throw std::runtime_error("Can not set value in Matrix Diagonal");
		}
	}
}

template<typename T>
auto Matrix<T>::getMinorDiagonalAsRow() const -> Matrix<T> {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is Empty");
	}
	else if (!this->isSquare()) {
		throw std::runtime_error("Matrix is not Square");
	}

	Matrix diagonalMatrix(1, this->mCols);

	for (std::size_t i = 0; i < this->mCols; i++) {
		if (diagonalMatrix.set(0, i, static_cast<T>(this->mData[i * this->mCols + (this->mCols - 1 - i)]))) {
			continue;
		}
		else {
			throw std::runtime_error("Can not set value in Matrix Diagonal");
		}
	}

	return diagonalMatrix;
}

template<typename T>
auto Matrix<T>::getMinorDiagonalAsCol() const -> Matrix<T> {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is Empty");
	}
	else if (!this->isSquare()) {
		throw std::runtime_error("Matrix is not Square");
	}

	Matrix diagonalMatrix(this->mRows, 1);

	for (std::size_t i = 0; i < this->mRows; i++) {
		if (diagonalMatrix.set(i, 0, static_cast<T>(this->mData[i * this->mCols + (this->mCols - 1 - i)]))) {
			continue;
		}
		else {
			throw std::runtime_error("Can not set value in Matrix Diagonal");
		}
	}
}

template<typename T>
auto Matrix<T>::adjugate() const -> Matrix<T> {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is Empty");
	}
	else if (!this->isSquare()) {
		throw std::runtime_error("Matrix is not Square");
	}
	
	Matrix<T> cofactorMatrix(this->mRows, this->mCols);
	
	for (std::size_t i = 0; i < this->mRows; i++) {
		for (std::size_t j = 0; j < this->mCols; j++) {
			Matrix<T> submatrix = this->createSubmatrix(i, j);
			T minor = submatrix.determinant();
			std::cout << minor << std::endl;
			T cofactor = static_cast<T>(std::pow(-1, i + j)) * minor;
			
			cofactorMatrix(i, j) = cofactor;
		}
	}
	Matrix<T> adjugate = cofactorMatrix.transpose();

	return adjugate;
}

template<typename T>
auto Matrix<T>::createSubmatrix(std::size_t excludeRow, std::size_t excludeCol) const -> Matrix<T> {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is Empty in createSubmatrix");
	}
	else if (excludeRow >= this->mRows || excludeCol >= this->mCols) {
		throw std::runtime_error("excludeRow or excludeCol out of the bounds in createSubmatrix");
	}

	Matrix<T> submatrix(this->mRows - 1, this->mCols - 1);

	for (std::size_t i = 0, sub_i = 0; i < this->mRows; i++) {
		if (i == excludeRow) {
			continue;
		}
		for (std::size_t j = 0, sub_j = 0; j < this->mCols; j++) {
			if (j == excludeCol) {
				continue;
			}

			T value = static_cast<T>(this->get(i, j));
			if (submatrix.set(sub_i, sub_j, value)) {
				continue;
			}
			sub_j++;
		}
		sub_i++;
	}
	return submatrix;
}

template<typename T>
auto Matrix<T>::power(const int& power) const -> Matrix<T> {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is Empty in power");
	}
	else if (!this->isSquare()) {
		throw std::runtime_error("Matrix is not Square in power");
	}
	else if (power < 0) {
		throw std::runtime_error("Negative Power is not supported in Matrix power");
	}

	if (power == 0) {
		return Matrix<T>::createIdentity(this->mRows);
	}
	if (power == 1) {
		return *this;
	}

	Matrix<T> result = this->power(power / 2);
	result = result * result;

	if (power % 2 == 0) {
		result = result * *this;
	}

	return result;
}

template<typename T>
auto Matrix<T>::inverse() const -> Matrix<T> {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is Empty in inverse");
	}
	else if (!this->isSquare()) {
		throw std::runtime_error("Matrix is not Square in inverse");
	}

	T det{ this->determinant() };
	if (det == 0) {
		throw std::runtime_error("Matrix is singular and can not bet inverted Matrix");
	}

	Matrix<T> adj = this->adjugate();
	std::cout << adj << std::endl;
	T detInverse = static_cast<T>(1) / static_cast<T>(det); 

	std::cout << detInverse << std::endl;
	for (std::size_t i = 0; i < this->mRows; ++i) {
		for (std::size_t j = 0; j < this->mCols; ++j) {
			adj(i, j) *= detInverse;
		}
	}

	return adj;
}

template<typename T>
auto Matrix<T>::kroneckerProduct(const Matrix<T>& matrix) const -> Matrix<T> {
	std::size_t row, col;
	std::tie(row, col) = matrix.size();

	if (this->mData.empty() or (row == 0 or col == 0)) {
		throw std::runtime_error("Matrix is empty");
	}

	std::size_t m = this->sizeRow(), n = this->sizeCol(), p = matrix.sizeRow(), q = matrix.sizeCol();
	Matrix<T> product(m * p, n * q);

	for (std::size_t i = 0; i < m; ++i) {
		for (std::size_t j = 0; j < n; ++j) {
			for (std::size_t k = 0; k < p; ++k) {
				for (std::size_t l = 0; l < q; ++l) {
					T a = this->get(i, j);
					T b = matrix.get(k, l);

					product(i * p + k, j * q + l) = static_cast<T>(a * b);
				}
			}
		}
	}

	return product;
}

template <typename T>
auto Matrix<T>::sizeRow() const noexcept -> std::size_t {
	return this->mRows;
}

template<typename T>
auto Matrix<T>::sizeCol() const noexcept -> std::size_t {
	return this->mCols;
}

template<typename T>
auto Matrix<T>::createIdentity(size_t n) noexcept -> Matrix<T> {
	Matrix<T> matrix(n, n);

	for (size_t index = 0; index < n; index++) {
		matrix(index, index) = static_cast<T>(1);
	}

	return matrix;
}

template<typename T>
auto Matrix<T>::createHankle(const Matrix<T>& firstRow, const Matrix<T>& lastCol) -> Matrix<T> {
	if (firstRow.sizeRow() != 1 || lastCol.sizeCol() != 1) {
		throw std::runtime_error("Input Matrices are invalid in Matrix Hankle");
	}

	std::size_t n = firstRow.sizeRow();
	if (lastCol.sizeRow() != n) {
		throw std::runtime_error("Incompatible dimensions between first row and last col");
	}

	Matrix<T> hankle(n, n);

	for (std::size_t i = 0; i < n; i++) {
		for (std::size_t j = 0; j < n; j++) {
			T value{};

			if (i + j < n) {
				value = firstRow.get(0, i + j);
			}
			else {
				value = lastCol.get(i + j - n + 1, 0);
			}
			
			hankle(i, j) = value;
		}
	}

	return hankle;
}

template <typename T>
auto Matrix<T>::createToeplitz(const Matrix<T>& firstRow, const Matrix<T>& firstCol) -> Matrix<T> {
	if (firstRow.sizeRow() != 1) {
		throw std::runtime_error("FirstRow should be a row vector in Matrix toeplitz");
	}
	else if (firstCol.sizeCol() != 1) {
		throw std::runtime_error("FirstCol should be a col vector in Matrix toeplitz");
	}

	std::size_t rows = firstCol.sizeRow();
	std::size_t cols = firstRow.sizeCol();
	Matrix<T> toeplitzMatrix(rows, cols);

	for (std::size_t i = 0; i < rows; i++) {
		for (std::size_t j = 0; j < cols; j++) {
			if (j >= i) {
				T value{ firstRow.get(0, j - i) };
				toeplitzMatrix(i, j) = value;
			}
			else {
				T value{ firstCol.get(i - j, 0) };
				toeplitzMatrix(i, j) = value;
			}
		}
	}

	return toeplitzMatrix;
}

template <typename T>
auto Matrix<T>::createFromArray(const std::initializer_list<T>& data, std::size_t rows, std::size_t cols) -> Matrix<T> {
	if (data.size() != rows * cols) {
		throw std::runtime_error("Mismatch between data size and matrix dimensions in Matrix");
	}
	else if (data.size() == 0) {
		throw std::runtime_error("Input data is null in Matrix create From Array");
	}
	else if (rows == 0 || cols == 0) {
		throw std::runtime_error("Rows or cols can not be zero in create From Array");
	}

	Matrix<T> matrix(rows, cols);
	auto it = data.begin();

	for (std::size_t i = 0; i < rows; i++) {
		for (std::size_t j = 0; j < cols; j++) {
			matrix(i, j) = *it++;
		}
	}

	return matrix;
}

template <typename T>
auto Matrix<T>::createCirculant(const Matrix<T>& firstRow) -> Matrix<T> {
	if (firstRow.sizeRow() != 1) {
		throw std::runtime_error("Input Must be a single row Matrix");
	}

	std::size_t n = firstRow.sizeCol();
	Matrix<T> circulantMatrix(n, n);

	for (std::size_t row = 0; row < n; row++) {
		for (std::size_t col = 0; col < n; col++) {
			std::size_t index = (col + row) % n;
			T value{ firstRow.get(0, index) };

			circulantMatrix(row, col) = value;
		}
	}

	return circulantMatrix;
}

template <typename T>
auto Matrix<T>::createHilbert(std::size_t n) -> Matrix<T> {
	if (n == 0) {
		throw std::runtime_error("Size is zero and its invalid for creating hilbert Matrix");
	}
	
	Matrix<T> hilbertMatrix(n, n);

	for (std::size_t i = 0; i < n; i++) {
		for (std::size_t j = 0; j < n; j++) {
			T value = static_cast<T>(1) / ((i + 1) + (j + 1) - 1.0);

			hilbertMatrix(i, j) = value;
		}
	}

	return hilbertMatrix;
}

template <typename T>
auto Matrix<T>::createHelmert(std::size_t n, bool full) -> Matrix<T> {
	if (n == 0) {
		throw std::runtime_error("Size is zero, which is invalid for creating a Helmert matrix.");
	}

	const std::size_t cols = full ? n : n - 1;
	Matrix<T> helmertMatrix(n, cols);

	for (std::size_t i = 0; i < n; i++) {
		for (std::size_t j = 0; j < (full ? n : i + 1); j++) {
			if (i == 0) {
				helmertMatrix(i, j) = static_cast<T>(1) / std::sqrt(static_cast<T>(n));
			}
			else if (j < i) {
				helmertMatrix(i, j) = static_cast<T>(1) / std::sqrt(static_cast<T>(i) * (i + 1));
			}
			else if (j == i) {
				helmertMatrix(i, j) = static_cast<T>(-std::sqrt(i / (i + 1.0)));
			}
			else { // fill remaining entries with 0
				if (full) {
					helmertMatrix(i, j) = static_cast<T>(0);
				}
			}
		}
	}

	return helmertMatrix;
}

template<typename T>
auto Matrix<T>::createPascal(std::size_t n) -> Matrix<T> {
	Matrix<T> pascalMatrix(n, n);

	for (std::size_t i = 0; i < n; i++) {
		for (std::size_t j = 0; j <= i; j++) {
			T value = binomialCoefficient(i + j, i);

			pascalMatrix(i, j) = value;
			pascalMatrix(j, i) = value;
		}
	}

	return pascalMatrix;
}

template<typename T>
auto Matrix<T>::createLeslie(const Matrix<T>& f, std::size_t fsize, const Matrix<T>& s, std::size_t ssize) -> Matrix<T> {
	if (fsize != ssize + 1) {
		throw std::runtime_error("The length of s must be less than the length of of the f");
	}

	Matrix<T> leslie(fsize, ssize);

	for (std::size_t i = 0; i < fsize; i++) {
		leslie(0, i) = f(0, i);
	}

	for (std::size_t i = 1; i < fsize; i++) {
		leslie(i, i - 1) = s(0, i - 1);
	}

	for (std::size_t i = 1; i < fsize; ++i) {
		for (std::size_t j = 0; j < fsize; ++j) {
			if (j != i - 1) { // Skip the sub-diagonal already set
				leslie(i, j) = static_cast<T>(0);
			}
		}
	}

	return leslie;
}

template <typename T>
auto Matrix<T>::createFiedler(const Matrix<T>& matrix) -> Matrix<T> {
	std::size_t a, b;

	if (std::tie(a, b) = matrix.size(); not(a and b)) {
		throw std::runtime_error("size of matrix is not true");
	}

	std::size_t n = matrix.sizeCol() >= matrix.sizeRow() ? matrix.sizeCol() : matrix.sizeRow();
	Matrix<T> fiedler(n, n);

	for (std::size_t i = 0; i < n; i++) {
		for (std::size_t j = 0; j < n; j++) {
			T value = static_cast<T>(std::fabs(matrix(0, i) - matrix(0, j)));
			fiedler(i, j) = value;
		}
	}
	return fiedler;
}

template<typename T>
auto Matrix<T>::createInverseHilbert(std::size_t n) -> Matrix<T> {
	Matrix<T> invH(n, n); 

	for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t j = 0; j < n; ++j) {
			int s = static_cast<int>(i + j);
			T sign = static_cast<T>((s % 2 == 0) ? 1 : -1);
			T numerator = sign * static_cast<T>(i + j + 1)
				* binomialFactorial(static_cast<T>(n + i), static_cast<T>(n - j - 1))
				* binomialFactorial(static_cast<T>(n + j), static_cast<T>(n - i - 1))
				* binomialFactorial(static_cast<T>(s), static_cast<T>(i))
				* binomialFactorial(static_cast<T>(s), static_cast<T>(j));
			T denominator = static_cast<T>(1); // The denominator for Hilbert matrix inverse entries when n <= 14 are effectively 1
			T value = numerator / denominator;

			invH(i, j) = value;
		}
	}

	return invH;
}

template<typename T>
auto Matrix<T>::createBlockDiagonal(const std::vector<Matrix<T>>& matrix) -> Matrix<T> {
	std::size_t totalRows = 0, totalCols = 0;

	for (const auto& mat : matrix) {
		totalRows += mat.mRows;
		totalCols += mat.mCols; 
	}

	Matrix<T> result(totalRows, totalCols);
	result.fill(static_cast<T>(0));

	std::size_t currentRow = 0, currentCol = 0;

	for (const auto& mat : matrix) {
		for (std::size_t i = 0; i < mat.mRows; ++i) {
			for (std::size_t j = 0; j < mat.mCols; ++j) {
				result(currentRow + i, currentCol + j) = mat(i, j);
			}
		}
		currentRow += mat.mRows;
		currentCol += mat.mCols; // Adjust for next matrix block position
	}

	return result;
}

template<typename T>
auto Matrix<T>::createRandomMatrix(std::size_t rows, std::size_t cols, T start, T end) -> Matrix<T> {
	Matrix<T> matrix(rows, cols);
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::uniform_int_distribution<int> distribution(start, end);

	auto randValue = [&]() -> T {
		if constexpr (std::is_integral<T>::value) {
			return distribution(generator);
		}
		else {
			std::uniform_real_distribution<float> floatDistribution(start, end);
			return floatDistribution(generator);
		}
	};

	for (std::size_t i = 0; i < rows; i++) {
		for (std::size_t j = 0; j < cols; j++) {
			matrix(i, j) = randValue();
		}
	}

	return matrix;
}

template<typename T>
auto Matrix<T>::createWalsh(size_t n) -> Matrix<T> {
	if (n & (n - 1)) {
		throw std::invalid_argument("Error: 'n' is not a power of 2.");
	}

	Matrix<T> walshMatrix(n, n);
	generateWalshMatrixRecursively(walshMatrix.mData, n, n, 0, 0, static_cast<T>(1));

	return walshMatrix;
}

template <typename T>
auto Matrix<T>::cofactor() const -> Matrix<T> {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is empty");
	}
	else if (!this->isSquare()) {
		throw std::runtime_error("Matrix object should be Square");
	}

	std::size_t n = this->mRows;
	Matrix<T> cofactorMatrix(n, n);

	for (std::size_t i = 0; i < n; i++) {
		for (std::size_t j = 0; j < n; j++) {
			Matrix<T> submatrix = Matrix<T>::createSubmatrix(i, j);
			T det{ submatrix.determinant() };
			T cofactor =  (i + j) % 2 == 0 ? 1 : -1 * det;

			cofactorMatrix(i, j) = cofactor;
		}
	}

	return cofactorMatrix;
}

template<typename T>
auto Matrix<T>::choleskyDecomposition() const -> Matrix<T> {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is empty");
	}
	if (!this->isSquare()) {
		throw std::runtime_error("Matrix must be square for Cholesky decomposition");
	}

	std::size_t n = this->mRows;
	Matrix<T> chol(n, n); 

	for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t j = i; j < n; ++j) {
			T sum = this->get(i, j);

			for (std::size_t k = 0; k < i; ++k) {
				sum -= chol.get(k, i) * chol.get(k, j);
			}

			if (i == j) {
				if (sum <= 0) {
					throw std::runtime_error("Matrix is not positive definite");
				}
				chol(i, i) = std::sqrt(sum);
			}
			else {
				chol(i, j) = sum / chol.get(i, i);
				chol(j, i) = 0;
			}
		}
	}

	// Optionally clear the lower triangular part, depending on how the constructor initializes Matrix
	for (std::size_t i = 1; i < n; ++i) {
		for (std::size_t j = 0; j < i; ++j) {
			chol(i, j) = static_cast<T>(0);
		}
	}

	return chol;
}

template<typename T>
auto Matrix<T>::inverseGaussJordan() const->Matrix<T> {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is Empty");
	}
	else if (!this->isSquare()) {
		throw std::runtime_error("Matris should be Square in Matrix inverse Gauss Jordan");
	}

	std::size_t n = this->mRows;
	Matrix<T> augmented(n, 2 * n);

	for (std::size_t i = 0; i < n; i++) {
		for (std::size_t j = 0; j < n; j++) {
			augmented(i, j) = this->operator()(i, j);
			augmented(i, j + n) = (i == j) ? static_cast<T>(1) : static_cast<T>(0);
		}
	}

	for (std::size_t col = 0; col < n; ++col) {
		if (augmented(col, col) == static_cast<T>(0)) {
			std::size_t swapRow = col + 1;

			while (swapRow < n && augmented(swapRow, col) == static_cast<T>(0)) {
				++swapRow;
			}
			if (swapRow == n) {
				throw std::runtime_error("Matrix is singular and cannot be inverted.");
			}
			augmented.swapRows(col, swapRow);
		}

		T pivotValue = augmented(col, col);
		for (std::size_t j = 0; j < 2 * n; ++j) {
			augmented(col, j) /= pivotValue;
		}

		for (std::size_t row = 0; row < n; ++row) {
			if (row != col) {
				T factor = augmented(row, col);
				for (std::size_t j = 0; j < 2 * n; ++j) {
					augmented(row, j) -= factor * augmented(col, j);
				}
			}
		}
	}

	Matrix<T> inverse(n, n);
	for (std::size_t i = 0; i < n; ++i) {
		for (std::size_t j = 0; j < n; ++j) {
			inverse(i, j) = augmented(i, j + n); 
		}
	}

	return inverse;
}

template <typename T>
auto Matrix<T>::projection() const -> Matrix<T> {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is Empty");
	}
	
	Matrix<T> matrixTranspose = this->transpose();
	Matrix<T> mta = matrixTranspose * *this;
	Matrix<T> mtaInv = mta.inverse();
	Matrix<T> mMtaInv = *this * mtaInv;
	Matrix<T> projection = mMtaInv * matrixTranspose;

	return projection;
}

template <typename T>
auto Matrix<T>::vandermonde(std::size_t n) const -> Matrix<T> {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is Empty for vandermonde");
	}

	Matrix<T> vandermondeMatrix(n, n);

	for (std::size_t i = 0; i < n; i++) {
		T value = static_cast<T>(i);
		for (std::size_t j = 0; j < n; ++j) {
			vandermondeMatrix(i, j) = std::pow(value, static_cast<T>(j));
		}
	}

	return vandermondeMatrix;
}

template<typename T>
auto Matrix<T>::companion(std::size_t degree) const -> Matrix<T> {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is Empty");
	}
	else if (degree < 1) {
		throw std::runtime_error("Degree must be at least 1.");
	}

	std::size_t n = degree - 1;
	Matrix<T> companionMatrix(n, n);

	for (std::size_t i = 0; i < n; i++) {
		for (std::size_t j = 0; j < n; j++) {
			if (j == n - 1) {
				companionMatrix(i, j) = -this->mData[n - 1 - i] / this->mData[degree - 1];
			}
			else if (i == j + 1) {
				companionMatrix(i, j) = 1;
			}
			else {
				companionMatrix(i, j) = 0;
			}
		}
	}

	return companionMatrix;
}

template<typename T>
auto Matrix<T>::addRowToRow(std::size_t targetRow, std::size_t sourceRow, T scale) -> bool {
	if (targetRow >= this->mRows || sourceRow >= this->mRows) {
		throw std::runtime_error("targetRow or sourceRow are bigger than matrix row");
	}
	else if (this->mData.empty()) {
		throw std::runtime_error("Matrix is Empty");
	}

	for (size_t j = 0; j < mCols; ++j) {
		this->mData[targetRow * this->mCols + j] += scale * this->mData[sourceRow * this->mCols + j];
	}

	return true;
}

template<typename T>
auto Matrix<T>::addColToCol(std::size_t targetCol, std::size_t sourceCol, T scale) -> bool {
	if (targetCol >= this->mCols || sourceCol >= this->mCols) {
		throw std::runtime_error("targetCol or sourceCol are bigger than matrix col");
	}
	else if (this->mData.empty()) {
		throw std::runtime_error("Matrix is Empty");
	}
	
	for (size_t i = 0; i < this->mRows; ++i) {
		this->mData[i * this->mCols + targetCol] += scale * this->mData[i * this->mCols + sourceCol];
	}

	return true;
}

template<typename T>
auto Matrix<T>::applyToRow(std::size_t row, std::function<T(T)> func) -> bool {
	if (row >= this->mRows) {
		throw std::runtime_error("row is bigger than matrix row");
	}
	else if (this->mData.empty()) {
		throw std::runtime_error("Matrix is Empty");
	}

	for (size_t j = 0; j < this->mCols; ++j) {
		this->mData[row * this->mCols + j] = func(this->mData[row * this->mCols + j]);
	}
	return true;
}

template<typename T>
auto Matrix<T>::applyToCol(std::size_t col, std::function<T(T)> func) -> bool {
	if (col >= this->mCols) {
		throw std::runtime_error("row is bigger than matrix row");
	}
	else if (this->mData.empty()) {
		throw std::runtime_error("Matrix is Empty");
	}

	for (size_t i = 0; i < mRows; ++i) {
		this->mData[i * this->mCols + col] = func(this->mData[i * this->mCols + col]);
	}
	return true;
}

template<typename T>
auto Matrix<T>::luDecomposition() const -> std::pair<Matrix<T>, Matrix<T>> {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is Empty for Lu Decomposition");
	}
	else if (!isSquare()) {
		throw std::runtime_error("Matrix must be square for LU decomposition.");
	}
	
	size_t n = mRows;
	Matrix<T> L(n, n), U(n, n);

	for (size_t i = 0; i < n; i++) {
		// Upper Triangular
		for (size_t k = i; k < n; k++) {
			T sum = T{};
			for (size_t j = 0; j < i; j++) {
				sum += L(i, j) * U(j, k);
			}
			U(i, k) = (*this)(i, k) - sum;
		}

		// Lower Triangular
		for (size_t k = i; k < n; k++) {
			if (i == k) {
				L(i, i) = T{ 1 }; // Diagonals of L are set to 1
			}
			else {
				T sum = T{};
				for (size_t j = 0; j < i; j++) {
					sum += L(k, j) * U(j, i);
				}
				L(k, i) = ((*this)(k, i) - sum) / U(i, i);
			}
		}
	}

	return std::make_pair(std::move(L), std::move(U));
}

template<typename T>
auto Matrix<T>::qrDecomposition() const -> std::pair<Matrix<T>, Matrix<T>> {
	if (this->mData.empty()) {
		throw std::runtime_error("Matrix is empty in qr Decomposion");
	}
	else if (this->mRows < this->mCols) {
		throw std::invalid_argument("Matrix must have more rows than columns for QR decomposition.");
	}

	size_t m = mRows;
	size_t n = mCols;
	Matrix<T> Q(m, n);
	Matrix<T> R(n, n);

	std::vector<T> a_col(m);
	std::vector<T> q_col(m);

	for (size_t i = 0; i < n; ++i) {
		for (size_t j = 0; j < m; ++j) {
			a_col[j] = this->get(j, i);
		}

		for (size_t k = 0; k < i; ++k) {
			for (size_t j = 0; j < m; ++j) {
				q_col[j] = Q.get(j, k);
			}
			subtractProjection(a_col, q_col); 
		}

		a_col = Matrix<T>::normalizeVector(a_col);

		for (size_t j = 0; j < m; ++j) {
			if (Q.set(j, i, a_col[j])) {
				continue;
			}
		}
	}

	for (size_t j = 0; j < n; ++j) {
		for (size_t i = 0; i <= j; ++i) {
			T r_ij = 0.0;
			for (size_t k = 0; k < m; ++k) {
				r_ij += Q.get(k, i) * this->get(k, j);
			}
			if (R.set(i, j, r_ij)) {
				continue;
			}
			
		}
	}

	return { std::move(Q), std::move(R) };
}

template<typename T>
inline T& Matrix<T>::operator()(size_t row, size_t col) {
	size_t index = row * mCols + col; // Ensure this calculation is correct
	if (index >= mData.size()) {
		throw std::out_of_range("Index out of range");
	}
	return mData[index];
}

template<typename T>
const T& Matrix<T>::operator()(size_t row, size_t col) const {
	size_t index = row * mCols + col; // Ensure this calculation is correct
	if (index >= mData.size()) {
		throw std::out_of_range("Index out of range");
	}

	return mData[index];
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix& matrix) const {
	if (this->mRows != matrix.mRows || this->mCols != matrix.mCols) {
		throw std::invalid_argument("Error: matrix indicess are not in same dimension.");
	}
	Matrix<T> result(this->mRows, this->mCols);

	std::transform(this->mData.begin(), this->mData.end(), matrix.mData.begin(), result.mData.begin(), std::plus<T>());

	return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix& matrix) const {
	if (this->mRows != matrix.mRows || this->mCols != matrix.mCols) {
		throw std::invalid_argument("Error: matrix indicess are not in same dimension.");
	}
	Matrix<T> result(this->mRows, this->mCols);

	std::transform(this->mData.begin(), this->mData.end(), matrix.mData.begin(), result.mData.begin(), std::minus<T>());

	return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix& matrix) const {
	if (this->mCols != matrix.mRows) {
		throw std::invalid_argument("Error: matrix indicess are not in same dimension.");
	}

	Matrix<T> result(this->mRows, this->mCols);

	for (size_t i = 0; i < this->mRows; i++) {
		for (size_t j = 0; j < matrix.mCols; j++) {
			T sum = T{};

			for (size_t k = 0; k < this->mCols; k++) {
				sum += this->mData[i * this->mCols + k] * matrix.mData[k * matrix.mCols + j];
			}
			result(i, j) = sum;
		}
	}

	return result;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix& other) noexcept {
	if (this != &other) {
		mRows = other.mRows;
		mCols = other.mCols;
		mData = other.mData;
	}
	return *this;
}

template<typename T>
Matrix<T>& Matrix<T>::operator=(Matrix&& other) noexcept {
	mRows = std::exchange(other.mRows, 0);
	mCols = std::exchange(other.mCols, 0);
	mData = std::move(other.mData);
	return *this;
}



template class Matrix<double>;
template class Matrix<float>;
template class Matrix<long double>;
template class Matrix<int>;
template class Matrix<size_t>;
template class Matrix<int32_t>;
template class Matrix<int64_t>;
template class Matrix<int16_t>;
template class Matrix<int8_t>;
template class Matrix<uint64_t>;
template class Matrix<uint32_t>;
template class Matrix<uint16_t>;
template class Matrix<uint8_t>;
