#include "BFGS.hpp"
#include "Gradient.hpp"
#include "LineSearch.hpp"

/**
 * @brief Computes the search direction using the current Hessian approximation B.
 *
 * Solves the linear system B * p = -grad using Conjugate Gradient.
 * Assumes B is SPD (symmetric positive definite).
 *
 * @param grad Current gradient vector.
 * @return Search direction vector p.
 * @throws std::runtime_error if B is not SPD and CG fails.
 */
Vector BFGS::computeDirectionP(const Vector& grad)
{
  solver.compute(B);

  if (solver.info() != Eigen::Success)
      throw std::runtime_error("CG failed: B is not SPD.");

  Vector p = solver.solve(-grad);
  return p;
}

/**
 * @brief Updates the Hessian approximation B using the BFGS formula.
 *
 * @param gamma Difference of gradients: gamma = grad_new - grad_old
 * @param delta Step taken: delta = x_new - x_old
 */
void BFGS::updateB(const Vector& gamma, const Vector& delta)
{
  double yBy = gamma.transpose().dot(B*gamma);
  double sy = delta.transpose().dot(gamma);

  Eigen::MatrixXd term1 = ((sy + yBy) * (delta*delta.transpose())) / (sy*sy);
  Eigen::MatrixXd term2 = ((B * gamma * delta.transpose()) + (delta * gamma.transpose() * B)) / sy;

  B += term1 - term2;
}

/**
 * @brief Updates the current solution x and gradient, computing delta and gamma.
 *
 * Uses a line search algorithm to determine step size alpha.
 *
 * @param x_old Current solution (will be updated to x_new).
 * @param grad_old Current gradient (will be updated to grad_new).
 * @param d Search direction vector.
 * @param delta Step vector (delta = alpha * d).
 * @param gamma Gradient difference (gamma = grad_new - grad_old).
 * @param type Type of line search to use.
 */
void BFGS::updateSolution(Vector& x_old, Vector& grad_old,
                          const Vector& d, Vector& delta, Vector& gamma,
                          lfbgs::LineSearchType type)
{
  auto line_search = lfbgs::make_line_search(type);
  double alpha = line_search->compute(fun_, x_old, grad_old, d);

  delta = alpha * d;
  Vector x_new = x_old + delta;

  Vector grad_new = gradTool.compute(fun_, x_new);
  gamma = grad_new - grad_old;

  x_old = x_new;
  grad_old = grad_new;
}

/**
 * @brief Executes the full BFGS optimization algorithm.
 *
 * Iteratively updates x using computed search directions until
 * the gradient norm is below the specified tolerance.
 */
void BFGS::run()
{
  unsigned int iter = 0;

  Vector x = x0_;
  Vector grad = gradTool.compute(fun_, x);
  Vector d, delta, gamma;

  while(grad.norm() > tol_) {
    d = BFGS::computeDirectionP(grad);
    updateSolution(x, grad, d, delta, gamma, type_);
    updateB(gamma, delta);

    ++iter;
  }

  solution_ = x;
}

/**
 * @brief Returns the current solution vector.
 * 
 * @return Current solution x.
 */
Eigen::VectorXd BFGS::getCurrentX() const
{
  return solution_;
}
