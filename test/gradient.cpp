#include <iostream>

#include <Eigen/Dense>
#include <cmath>

#include "MathTools.hpp"

using PointType = Eigen::VectorXd;

void test_with_lambda_function() {
  auto f = [](const PointType & x) {
    // f(x,y) = 4x^2 * y + sqrt(y) * y^2
    return 4*x(0)*x(0) * x(1) + std::sqrt(x(1))*x(1)*x(1);
  };

  PointType point(2);
  point << 1, 1;
  // evaluate the gradient of the given f function in a point
  PointType grad = MathTools::gradient(f, point);
  std::cout << "[ " << grad(0) << ", " << grad(1) << " ]" << std::endl;

  std::cout << "expected: 8, 6.5" << std::endl;
}

void test_with_function_object() {
  struct Functor {
    double operator()(const PointType & x) const {
      // f(x,y) = 4x^2 * y + sqrt(y) * y^2
      // f(x,y) = exp(x) * sin(y) + x^3 * y^2 + cos(x*y)
      return std::exp(x(0)) * std::sin(x(1)) + x(0)*x(0)*x(0) * x(1)*x(1) + std::cos(x(0)*x(1));
    }
  };

  Functor f;

  PointType point(2);
  point << 0, 1;
  // evaluate the gradient of the given f function in a point
  PointType grad = MathTools::gradient(f, point);
  std::cout << "[ " << grad(0) << ", " << grad(1) << " ]" << std::endl;

  std::cout << "expected: 0.84147, 0.54030" << std::endl;
}

int main() {
  test_with_lambda_function();
  test_with_function_object();

  return 0;
}
