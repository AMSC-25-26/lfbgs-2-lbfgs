
#pragma once

#include <memory>
#include <cmath>
#include <functional>
#include <Eigen/Dense>
#include "MathTools.hpp"

namespace lfbgs {

/**
 * @brief Enumeration of available line search strategies.
 *
 * This enum specifies which line search algorithm should be used
 * to determine the step size during an optimization iteration.
 */
enum class LineSearchType {
    BacktrackingArmijo,  ///< Classic Armijo backtracking line search.
    StrongWolfe          ///< Line search enforcing the Strong Wolfe conditions.
};


/**
 * @brief Abstract base class for line search algorithms.
 *
 * A line search algorithm computes a suitable step size @f$ \alpha @f$
 * that satisfies some descent condition given:
 *  - a point @f$ x @f$
 *  - its gradient @f$ g @f$
 *  - a search direction @f$ p @f$
 *
 * Concrete implementations (e.g., Armijo, Strong Wolfe)
 * must inherit from this class and implement the compute() method.
 */
class LineSearch {
public:
    /// Virtual destructor (required for polymorphic deletion).
    virtual ~LineSearch() = default;

    /**
     * @brief Computes the step size for the given search direction.
     *
     * @param fun Objective function to minimize.
     * @param x Current point.
     * @param g Gradient at the current point.
     * @param p Search direction.
     * @return A positive scalar step size @f$ \alpha @f$.
     */
    virtual double compute(
        const std::function<double(const Eigen::VectorXd&)>& fun,
        const Eigen::VectorXd& x,
        const Eigen::VectorXd& g,
        const Eigen::VectorXd& p
    ) = 0;
};


/**
 * @brief Factory function for constructing a line search object.
 *
 * Given a LineSearchType, this function creates and returns the
 * corresponding line search strategy.
 *
 * @param type Chosen line search algorithm (Armijo, Strong Wolfe).
 * @return A unique_ptr to the corresponding concrete LineSearch implementation.
 */
std::unique_ptr<LineSearch> make_line_search(LineSearchType type);

} // namespace lfbgs
