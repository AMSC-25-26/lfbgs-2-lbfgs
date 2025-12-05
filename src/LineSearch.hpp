
#pragma once
#include <memory>
#include <cmath>
#include <functional>
#include <Eigen/Dense>
#include "MathTools.hpp"

namespace lfbgs {
/**
 * Enum class for line search types
 */
enum class LineSearchType {
    BacktrackingArmijo,
    StrongWolfe
};

/**
 * Abstract base class for line search algorithms
 *
 */
class LineSearch {
public:
    virtual ~LineSearch() = default;
    /**
     * Compute the step size for the line search
     * 
     * @param fun function to evaluate
     * @param x current point
     * @param g gradient at current point
     * @param p search direction
     * @return step size
     */
    virtual double compute(const std::function<double(const Eigen::VectorXd&)>& fun, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&) = 0;
};

/**
 * Factory function to create a line search object
 * 
 * @param type type of the line search algorithm (BacktrackingArmijo or StrongWolfe)
 * @return unique pointer to the line search object of the specified type
 */
std::unique_ptr<LineSearch> make_line_search(LineSearchType type);

}
