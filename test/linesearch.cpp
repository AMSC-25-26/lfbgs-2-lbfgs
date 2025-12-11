#include <algorithm>
#include <cmath>
#include <iostream>

#include "LineSearch.hpp"
#include "MathTools.hpp"

using lfbgs::LineSearchType;


bool approx_equal(double a, double b, double tol = 1e-6) {
  return std::abs(a - b) <= tol * std::max({1.0, std::abs(a), std::abs(b)});
}

/**
 * @brief Quadratic test function f(x) = 0.5 * ||x||^2
 * @param x Input vector
 * @return Function value
 */
double quadratic(const Eigen::VectorXd& x) {
  return 0.5 * x.squaredNorm();
}

/**
 * @brief Gradient of quadratic function
 * @param x Input vector
 * @return Gradient vector (equal to x for this function)
 */
Eigen::VectorXd quadratic_grad(const Eigen::VectorXd& x) {
  return x;
}

/**
 * @brief Test BacktrackingArmijo line search on quadratic function
 * @return 0 on success, 1 on failure
 */
int test_backtracking_armijo() {
  auto line_search = lfbgs::make_line_search(LineSearchType::BacktrackingArmijo);
  
  Eigen::VectorXd x(2);
  x << 1.0, 1.0;
  Eigen::VectorXd g = quadratic_grad(x);
  Eigen::VectorXd p = -g;
  
  double alpha = line_search->compute(quadratic, x, g, p);
  
  if (!approx_equal(alpha, 1.0, 1e-2)) {
    std::cerr << "BacktrackingArmijo expected alpha ~ 1, got " << alpha << "\n";
    return 1;
  }
  return 0;
}

/**
 * @brief Test StrongWolfe line search on quadratic function
 * @return 0 on success, 1 on failure
 */
int test_strong_wolfe() {
  auto line_search = lfbgs::make_line_search(LineSearchType::StrongWolfe);
  
  Eigen::VectorXd x(2);
  x << 1.0, 1.0;
  Eigen::VectorXd g = quadratic_grad(x);
  Eigen::VectorXd p = -g;
  
  double alpha = line_search->compute(quadratic, x, g, p);
  
  if (!approx_equal(alpha, 1.0, 1e-2)) {
    std::cerr << "StrongWolfe expected alpha ~ 1, got " << alpha << "\n";
    return 1;
  }
  return 0;
}

int main() {
  int failures = 0;
  failures += test_backtracking_armijo();
  failures += test_strong_wolfe();
  
  if (failures == 0) {
    std::cout << "All line_search tests passed" << std::endl;
  }
  return failures;
}
