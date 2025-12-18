#include <iostream>
#include <cmath>
#include "LBFGS.hpp"


void test_with_lambda_function() {
    auto myFunc = [](const Vector &x) -> double {
        return std::pow(x[0] - 2.0, 2) + std::pow(x[1] + 3.0, 2);
    };

    Vector x0(2);
    x0 << 0.0, 0.0;
    
    double tol = 1e-6;

    LBFGS<30> optimizer(x0, myFunc, tol, lfbgs::LineSearchType::StrongWolfe);

    optimizer.run();

    std::cout << "Minimum in: " << optimizer.getCurrentX().transpose() << std::endl;
    std::cout << "Function value: " << myFunc(optimizer.getCurrentX()) << std::endl;

}



int main() {
  test_with_lambda_function();

  return 0;
}
