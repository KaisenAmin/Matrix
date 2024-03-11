# Matrix Library

Author: Amin Tahmasebi

This project is a comprehensive matrix manipulation library implemented in C++23, designed to perform a wide range of matrix operations efficiently and intuitively. The project is built using CMake, ensuring easy build and integration into other projects.

## Features

- **Arithmetic Operations:** Perform basic arithmetic operations such as addition, subtraction, and multiplication on matrices.
- **Decompositions:** Includes methods for LU and QR decompositions.
- **Determinant and Inverse:** Compute the determinant and inverse of square matrices.
- **Norms and Other Properties:** Calculate various matrix norms and check matrix properties like symmetry, orthogonality, etc.
- **Transformation and Decomposition:** Transform matrices and decompose them into simpler forms.
- **Utility Functions:** Various utility functions like transpose, trace, and submatrix creation are included for convenience.


## Usage

To use the Matrix library in your project, include the `Matrix.hpp` header file and link against the compiled library. Example usage:

### 1. Creating a Matrix and Printing Its Content

```cpp
Matrix<int> mat1(2, 3, {1, 2, 3, 4, 5, 6});
std::cout << "Matrix 1:\n" << mat1 << std::endl;
```

### 2. Matrix Addition

```cpp
Matrix<int> mat2(2, 3, {6, 5, 4, 3, 2, 1});
auto result = mat1 + mat2;
std::cout << "Matrix Addition Result:\n" << result << std::endl;
```

### 3. Matrix Transpose

```cpp
auto transposed = mat1.transpose();
std::cout << "Transposed Matrix:\n" << transposed << std::endl;
```

### 4. Computing Determinant of a Square Matrix

```cpp
Matrix<double> squareMat(3, 3, {1, 2, 3, 0, 4, 5, 1, 0, 6});
double det = squareMat.determinant();
std::cout << "Determinant: " << det << std::endl;
```

### 5. Inverse of a Matrix

```cpp
auto inverseMat = squareMat.inverse();
std::cout << "Inverse Matrix:\n" << inverseMat << std::endl;
```

### 6. Checking if a Matrix is Symmetric

```cpp
bool isSymmetric = squareMat.isSymmetric();
std::cout << "Is Symmetric? " << std::boolalpha << isSymmetric << std::endl;
```

### 7. LU Decomposition

```cpp
auto [L, U] = squareMat.luDecomposition();
std::cout << "L Matrix:\n" << L << "\nU Matrix:\n" << U << std::endl;
```

### 8. QR Decomposition

```cpp
Matrix<double> matQR(3, 2, {12, -51, 4, 6, 167, -68});
auto [Q, R] = matQR.qrDecomposition();
std::cout << "Q Matrix:\n" << Q << "\nR Matrix:\n" << R << std::endl;
```

### 9. Creating a Random Matrix

```cpp
auto randomMat = Matrix<int>::createRandomMatrix(3, 3, 1, 10);
std::cout << "Random Matrix:\n" << randomMat << std::endl;
```

### 10. Kronecker Product

```cpp
Matrix<int> matA(2, 2, {1, 2, 3, 4});
Matrix<int> matB(2, 2, {0, 5, 6, 7});
auto kronecker = matA.kroneckerProduct(matB);
std::cout << "Kronecker Product:\n" << kronecker << std::endl;
```

## Documentation

Each method in the `Matrix` class is briefly explained below:

### Constructor and Destructor

- **Matrix()**: Default constructor, creates an empty matrix.
- **Matrix(size_t row, size_t col)**: Constructs a matrix of given dimensions, uninitialized data.
- **Matrix(size_t row, size_t col, std::vector<T>& data)**: Constructs a matrix with data.
- **Matrix(size_t row, size_t col, std::initializer_list<T> data)**: Constructs a matrix from an initializer list.
- **Matrix(const Matrix& other)**: Copy constructor.
- **Matrix(Matrix&& other)**: Move constructor.

### Assignment Operators

- **operator=()**: Copy and move assignment operators.

### Access and Modification

- **set()**: Sets the value at a specific row and column.
- **get()**: Retrieves the value at a specific row and column.
- **operator()**: Access or modify elements via (row, col) syntax.

### Properties and Checks

- **isSquare()**: Checks if the matrix is square.
- **isSymmetric()**: Checks if the matrix is symmetric.
- ... (Include a brief description for each method similarly.)

### Matrix Operations

- **transpose()**: Returns the transpose of the matrix.
- **determinant()**: Computes the determinant of the matrix.
- **inverse()**: Calculates the inverse of the matrix.
- **luDecomposition()**: Performs LU decomposition.
- **qrDecomposition()**: Performs QR decomposition.

### Utility Functions

- **sizeRow() / sizeCol()**: Returns the number of rows/columns.
- **fill()**: Fills the matrix with a specific value.
- **swapRows() / swapCols()**: Swaps two rows/columns.

### Advanced Operations

- **kroneckerProduct()**: Computes the Kronecker product with another matrix.
- **cofactor()**: Calculates the cofactor matrix.

For detailed examples and more complex operations, refer to the included tests or example programs.

## Contributing

Contributions are welcome. Please open an issue or pull request to suggest improvements or add new features.

## License

This project is open-source and available under the MIT License.
