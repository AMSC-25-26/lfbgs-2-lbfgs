#include <vector>
#include <utility>
#include <Eigen/Dense>
#include "BFGS.hpp"
#include "utils/MathTools.hpp"

using namespace Eigen;

/**
 * @brief Limited-memory BFGS optimizer (L-BFGS).
 *
 * This class implements the limited-memory variant of the BFGS algorithm.
 * Instead of storing the full Hessian approximation, it keeps only the last
 * @p m correction pairs (s_k, y_k) and uses the classical two-loop recursion
 * to compute the search direction.
 *
 * @tparam m Maximum number of (s, y) pairs to store.
 */
template <unsigned int m>
class LBFGS : public BFGS {
  
  public:
    /**
     * @brief Constructs an L-BFGS optimizer.
     *
     * @param x0 Initial point.
     * @param fun Objective function f(x) to minimize.
     * @param tol Gradient norm tolerance for stopping.
     * @param type Line search strategy.
     */
    LBFGS(const VectorXd &x0,
          const std::function<double(const VectorXd &)> &fun,
          double tol,
          lfbgs::LineSearchType type)
      : BFGS(x0, fun, tol, type)
    {}

    /**
     * @brief Executes the L-BFGS optimization algorithm.
     *
     * Overrides the virtual method defined in BFGS.
     */
    void run() override;

  private:

    /**
     * @brief Limited memory of correction pairs (s_k, y_k).
     *
     * Each element contains:
     * - s_k = x_{k+1} − x_k
     * - y_k = ∇f(x_{k+1}) − ∇f(x_k)
     */
    std::vector<std::pair<VectorXd, VectorXd>> history;

    /**
     * @brief Alpha coefficients used in the two-loop recursion.
     */
    std::vector<double> alpha;

    /**
     * @brief First loop of the two-loop recursion.
     *
     * Computes the alpha coefficients and updates vector q.
     *
     * @param q Gradient-like vector to be modified in place.
     */
    void backward_pass(VectorXd &q);

    /**
     * @brief Second loop of the two-loop recursion.
     *
     * Updates the search direction approximation.
     *
     * @param r Vector updated to the final approximate direction.
     */
    void forward_pass(VectorXd &r);
};


template <unsigned int m>
void LBFGS<m>::backward_pass(VectorXd &q) {
  for (int i = history.size() - 1; i >= 0; --i) {
    double rho = 1.0 / (history[i].second.dot(history[i].first)); // 1 / (y_i^T s_i)
    alpha[i] = rho * (history[i].first.dot(q));
    q -= alpha[i] * history[i].second;
  }
}

template <unsigned int m>
void LBFGS<m>::forward_pass(VectorXd &r) {
  for (size_t i = 0; i < history.size(); ++i) {
    double rho = 1.0 / (history[i].second.dot(history[i].first));
    double beta = rho * (history[i].second.dot(r));
    r += (alpha[i] - beta) * history[i].first;
  }
}

template <unsigned int m>
void LBFGS<m>::run() {
  history.reserve(m);
  alpha.resize(m);

  int iter = 0;
  VectorXd x = x0_;
  VectorXd grad = MathTools::gradient(fun_, x);
  VectorXd d, s, y;
  
  while (grad.norm() > tol_) {

    // Use full BFGS direction until enough correction pairs are accumulated
    if (iter < m) {
      d = computeDirectionP(grad);
    } else {
      // Two-loop recursion for L-BFGS direction
      VectorXd q = grad;
      backward_pass(q);
      VectorXd r = B * q;
      forward_pass(r);
      d = -r;
    }

    // Update x, gradient, and compute correction pairs
    updateSolution(x, grad, d, s, y, type_);

    // Update limited-memory history
    if (history.size() == m) {
      history.erase(history.begin());
    }
    history.push_back({s, y});

    ++iter;
  }
  
  solution_ = x;
}
