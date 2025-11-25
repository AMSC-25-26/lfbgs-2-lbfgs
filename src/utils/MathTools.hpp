#ifndef MATHTOOLS_HPP
#define MATHTOOLS_HPP

#include <functional>
#include <ranges>

#include <concepts>


template<typename T>
concept Point =
  // component type must be exactly double.
  std::is_same_v<std::ranges::range_value_t<T>, double> &&

  // support size check and writing to elements.
  requires(T a, size_t i, double d) {
      { a.size() } -> std::convertible_to<size_t>;
      { a[i] = d };
  };

template<typename F, typename P>
concept Function =
  std::invocable<F, const P &> &&
  std::convertible_to<std::invoke_result_t<F, const P &>, double>;

/*!
  \brief A collection of mathematical tools.
*/
class MathTools {
  private:

    // . . .

  public:

    /*!
      \brief Compute the numerical gradient of a given function in a given point.

      \param f Function for which the gradient has to be computed.
      \param point Point in which the gradient has to be computed.
      \return The gradient of f in the given point.
    */
    template<
      Point P,
      Function<P> F   // function taking a const P & and returning double
    >
    static P gradient(
      const F & f,
      const P & point
    ) {
      P grad = point;

      double h = 1e-8; // half eps
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

#endif
