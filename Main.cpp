#include <iostream>
#include <format>
#include "matrix/matrix.hpp"


auto main() -> int {
	Matrix<int> matA(2, 2, { 1, 2, 3, 4 });
	Matrix<int> matB(2, 2, { 0, 5, 6, 7 });
	auto kronecker = matA.kroneckerProduct(matB);
	std::cout << "Kronecker Product:\n" << kronecker << std::endl;


	std::cin.get();
	return EXIT_SUCCESS;
}
