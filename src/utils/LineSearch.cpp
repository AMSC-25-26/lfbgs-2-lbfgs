#include "LineSearch.hpp"
#include "utils/MathTools.hpp"

#define MAX_ITER 100

namespace lfbgs {

/**
 * @class BacktrackingArmijo
 * @brief Implements the classic backtracking Armijo line search.
 *
 * Iteratively reduces the step size alpha until the Armijo condition is satisfied:
 * f(x + alpha * p) <= f(x) + c1 * alpha * g.dot(p)
 */
class BacktrackingArmijo : public LineSearch {
public:
    /**
     * @brief Compute step size using backtracking Armijo rule.
     * @param fun Objective function.
     * @param x Current point.
     * @param g Gradient at current point.
     * @param p Search direction.
     * @return Step size alpha satisfying Armijo condition.
     */
    double compute(const std::function<double(const Eigen::VectorXd&)>& fun,
                   const Eigen::VectorXd& x,
                   const Eigen::VectorXd& g,
                   const Eigen::VectorXd& p) override {
        double alpha = 1.0;
        const double c1 = 1e-4;
        const double rho = 0.5;

        double fx = fun(x);
        double gtp = g.dot(p);

        for (int i = 0; i < MAX_ITER; ++i) {
            double f_new = fun(x + alpha * p);
            if (f_new <= fx + c1 * alpha * gtp) {
                return alpha;
            }
            alpha *= rho;
        }
        return alpha;
    }
};

/**
 * @class StrongWolfe
 * @brief Implements a line search that enforces the Strong Wolfe conditions.
 *
 * The algorithm iteratively searches for alpha such that:
 *  - Armijo condition (sufficient decrease) is satisfied
 *  - Curvature condition |grad(x+alpha*p).dot(p)| <= -c2 * grad(x).dot(p) is satisfied
 */
class StrongWolfe : public LineSearch {
public:
    /**
     * @brief Compute step size using Strong Wolfe conditions.
     * @param fun Objective function.
     * @param x Current point.
     * @param g Gradient at current point.
     * @param p Search direction.
     * @return Step size alpha satisfying Strong Wolfe conditions.
     */
    double compute(const std::function<double(const Eigen::VectorXd&)>& fun,
                   const Eigen::VectorXd& x,
                   const Eigen::VectorXd& g,
                   const Eigen::VectorXd& p) override {
        double alpha = 1.0;
        double alpha_prev = 0.0;
        const double c1 = 1e-4;
        const double c2 = 0.9;
        const double max_step_norm = 100.0;
        const double alpha_max = max_step_norm / p.norm();

        double phi0 = fun(x);
        double dphi0 = g.dot(p);
        double phi_prev = phi0;

        for (int i = 1; i <= MAX_ITER; ++i) {
            double phi = fun(x + alpha * p);

            // Check Armijo and monotonicity
            if (phi > phi0 + c1 * alpha * dphi0 || (i > 1 && phi >= phi_prev)) {
                return zoom(fun, x, p, alpha_prev, alpha, phi0, dphi0, c1, c2);
            }

            Eigen::VectorXd x_new = x + alpha * p;
            Eigen::VectorXd g_new = MathTools::gradient(fun, x_new);
            double dphi = g_new.dot(p);

            // Curvature condition
            if (std::abs(dphi) <= -c2 * dphi0) {
                return alpha;
            }

            if (dphi >= 0) {
                return zoom(fun, x, p, alpha, alpha_prev, phi0, dphi0, c1, c2);
            }

            alpha_prev = alpha;
            phi_prev = phi;
            alpha = std::min(alpha * 2.0, alpha_max);
        }
        return alpha;
    }

private:
    /**
     * @brief Zoom function used in Strong Wolfe line search.
     *
     * Performs bisection search between alpha_lo and alpha_hi to satisfy
     * both Armijo and curvature conditions.
     */
    double zoom(const std::function<double(const Eigen::VectorXd&)>& fun,
                const Eigen::VectorXd& x,
                const Eigen::VectorXd& p,
                double alpha_lo,
                double alpha_hi,
                double phi0,
                double dphi0,
                double c1,
                double c2) {

        for (int i = 0; i < MAX_ITER; ++i) {
            double alpha_j = 0.5 * (alpha_lo + alpha_hi);
            double phi_j = fun(x + alpha_j * p);
            double phi_lo = fun(x + alpha_lo * p);

            if (phi_j > phi0 + c1 * alpha_j * dphi0 || phi_j >= phi_lo) {
                alpha_hi = alpha_j;
            } else {
                Eigen::VectorXd x_new = x + alpha_j * p;
                double dphi_j = MathTools::gradient(fun, x_new).dot(p);

                if (std::abs(dphi_j) <= -c2 * dphi0) {
                    return alpha_j;
                }

                if (dphi_j * (alpha_hi - alpha_lo) >= 0) {
                    alpha_hi = alpha_lo;
                }
                alpha_lo = alpha_j;
            }
        }
        return alpha_lo;
    }
};

/**
 * @brief Factory function for creating a line search object.
 * @param type The type of line search to create.
 * @return Unique pointer to a LineSearch object of the requested type.
 */
std::unique_ptr<LineSearch> make_line_search(LineSearchType type) {
    switch (type) {
        case LineSearchType::BacktrackingArmijo:
            return std::make_unique<BacktrackingArmijo>();
        default:
            return std::make_unique<StrongWolfe>();
    }
}

} // namespace lfbgs
