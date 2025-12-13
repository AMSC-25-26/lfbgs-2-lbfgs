#include <iostream>
#include "BFGS.hpp"
#include "LBFGS.hpp"

#include "LineSearch.hpp"

#include <Eigen/Dense>
#include <cmath>

#include "Gradient.hpp"

using PointType = Eigen::VectorXd;

void test_with_lambda_function() {
    auto myFunc = [](const Eigen::VectorXd &x) -> double {
        return std::pow(x[0] - 2.0, 2) + std::pow(x[1] + 3.0, 2);
    };

    Eigen::VectorXd x0(2);
    x0 << 0.0, 0.0;
    
    double tol = 1e-6;

    LBFGS<30> optimizer(x0, myFunc, tol, lfbgs::LineSearchType::StrongWolfe);

    optimizer.run();

    std::cout << "Minimo trovato in: " << optimizer.getCurrentX().transpose() << std::endl;
    std::cout << "Valore funzione: " << myFunc(optimizer.getCurrentX()) << std::endl;

}



int main() {
  test_with_lambda_function();

  return 0;
}
