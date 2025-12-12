#ifndef MATHTOOLS_HPP
#define MATHTOOLS_HPP

#include <functional>
#include <ranges>
#include <concepts>

/**
 * @brief Concept to check whether a type represents a point/vector in R^n.
 *
 * The type must:
 *  - contain elements of type double,
 *  - support .size() returning size_t,
 *  - support element assignment via operator[].
 */
template<typename T>
concept Point =
  std::is_same_v<std::ranges::range_value_t<T>, double> &&
  requires(T a, size_t i, double d) {
      { a.size() } -> std::convertible_to<size_t>;
      { a[i] = d };
  };

/**
 * @brief Concept to check whether a type represents a function from P to double.
 *
 * The function must be invocable with const P& and return a type convertible to double.
 */
template<typename F, typename P>
concept Function =
  std::invocable<F, const P &> &&
  std::convertible_to<std::invoke_result_t<F, const P &>, double>;

/**
 * @class MathTools
 * @brief Utility class for mathematical operations.
 *
 * Currently provides:
 *  - Numerical gradient computation.
 */
class MathTools {
private:
    // Private members (if any) ...

public:

    /**
     * @brief Compute the numerical gradient of a given function at a given point.
     *
     * Uses central finite differences with step size h = 1e-8.
     *
     * @tparam P Type of the point/vector.
     * @tparam F Type of the function.
     * @param f Function for which the gradient is computed.
     * @param point Point at which to compute the gradient.
     * @return Gradient vector evaluated at the point.
     */
    template<
      Point P,
      Function<P> F
    >
    static P gradient(
      const F & f,
      const P & point
    ) {
      P grad = point;

      double h = 1e-8; // finite difference step
      for (size_t i = 0; i < point.size(); ++i) {
        P x_plus_h = point;
        P x_minus_h = point;
        x_plus_h[i] += h;
        x_minus_h[i] -= h;

        grad[i] = (f(x_plus_h) - f(x_minus_h)) / (2.0 * h);
      }
      return grad;
    }

};

#endif // MATHTOOLS_HPP
