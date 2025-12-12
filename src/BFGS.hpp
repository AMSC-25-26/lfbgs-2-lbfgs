/**
 * @file BFGS.hpp
 * @brief Declaration of the BFGS optimization algorithm class.
 *
 * This file defines the BFGS class, which implements the classical
 * quasi-Newton BFGS method for unconstrained optimization.  
 * The class provides tools to approximate the Hessian, compute the 
 * search direction, update the iterate, and run the optimization.
 */

#ifndef BFGS_HPP
#define BFGS_HPP

#include <Eigen/Dense>
#include <functional>
#include "LineSearch.hpp"

using Vector = Eigen::VectorXd;

/**
 * @class BFGS
 * @brief Classical BFGS quasi-Newton optimizer.
 *
 * This class implements the standard BFGS algorithm:
 * - maintains a full Hessian approximation \( B_k \)
 * - computes descent directions by solving \( B_k p_k = -g_k \)
 * - updates the Hessian using the BFGS rank-two formula
 * - updates the iterate using a line search (Armijo or Wolfe)
 *
 * The run() method executes the optimization loop until convergence.
 */
class BFGS {
public:

    /**
     * @brief Construct a BFGS solver.
     *
     * @param x0  Initial point of the optimization.
     * @param fun Objective function to minimize.
     * @param tol Stopping tolerance on the gradient norm.
     * @param type Type of line search (Armijo or Strong Wolfe).
     */
    BFGS(
        const Vector & x0,
        const std::function<double(Vector const&)>& fun,
        const double & tol,
        lfbgs::LineSearchType type
    )
    :
        x0_(x0),
        B(Eigen::MatrixXd::Identity(x0.rows(), x0.rows())),
        fun_(fun),
        tol_(tol),
        type_(type)
    {}

    /**
     * @brief Run the BFGS optimization loop.
     *
     * Executes:
     * 1. Gradient computation  
     * 2. Direction computation  
     * 3. Line search  
     * 4. Hessian approximation update  
     *
     * Stops when ||grad|| < tol.
     */
    virtual void run();

    /**
     * @brief Get the last computed solution.
     * @return The current estimate of the minimizer.
     */
    Vector getCurrentX() const;

protected:

    /** @brief Objective function. */
    std::function<double(Vector const&)> fun_;

    /** @brief Initial point. */
    Vector x0_;

    /** @brief Last computed solution. */
    Vector solution_ = x0_;

    /** @brief Hessian approximation matrix \(B_k\). */
    Eigen::MatrixXd B;

    /** @brief Cholesky solver for systems with matrix B . */
    Eigen::LLT<Eigen::MatrixXd> solver;

    /** @brief Stopping tolerance. */
    double tol_;

    /** @brief Type of line search algorithm. */
    lfbgs::LineSearchType type_;

    /**
     * @brief Compute the search direction.
     *
     * Solves the linear system:
     * \f[
     *     B_k p_k = - \nabla f(x_k)
     * \f]
     * using Eigen’s conjugate gradient solver.
     *
     * @param grad Current gradient.
     * @return The search direction \( p_k \).
     */
    Vector computeDirectionP(const Vector& grad);

    /**
     * @brief Update iterate, gradient and step vectors.
     *
     * Computes:
     * - \( x_{k+1} = x_k + \alpha p_k \)
     * - \( \delta_k = x_{k+1} - x_k \)
     * - \( \gamma_k = g_{k+1} - g_k \)
     *
     * @param x_old  Previous iterate (updated in-place).
     * @param g_old  Previous gradient (updated in-place).
     * @param d      Search direction.
     * @param delta  Step vector \( \delta_k = x_{k+1}-x_k \).
     * @param gamma  Gradient difference \( \gamma_k = g_{k+1}-g_k \).
     * @param type   Line search type.
     */
    void updateSolution(
        Vector& x_old,
        Vector& g_old,
        const Vector& d,
        Vector& delta,
        Vector& gamma,
        lfbgs::LineSearchType type
    );

private:

    /**
     * @brief Update the Hessian approximation using BFGS formula.
     *
     * Computes:
     * \f[
     *      B_{k+1}
     *      = B_k 
     *      + \frac{ \gamma\gamma^T }{ \gamma^T \delta }
     *      - \frac{ B_k \delta \delta^T B_k }{ \delta^T B_k \delta }
     * \f]
     *
     * @param gamma Gradient difference.
     * @param delta Step vector.
     */
    void updateB(
        const Vector& gamma,
        const Vector& delta
    );
};

#endif // BFGS_HPP
