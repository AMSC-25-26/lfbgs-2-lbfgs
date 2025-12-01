#include "LineSearch.hpp"

#define MAX_ITER 100

namespace lfbgs {

class BacktrackingArmijo : public LineSearch {
public:
    double compute(const std::function<double(const Eigen::VectorXd&)>& fun, const Eigen::VectorXd& x, const Eigen::VectorXd& g, const Eigen::VectorXd& p) override {
        double alpha = 1.0;
        const double c1 = 1e-4;
        const double rho = 0.5;
        
        double fx = fun(x);
        double gtp = g.dot(p);
        

        int max_iter = MAX_ITER;
        for (int i = 0; i < max_iter; ++i) {
            double f_new = fun(x + alpha * p);
            if (f_new <= fx + c1 * alpha * gtp) {
                return alpha;
            }
            alpha *= rho;
        }
        return alpha;
    }
};

class StrongWolfe : public LineSearch {
public:
    double compute(const std::function<double(const Eigen::VectorXd&)>& fun, const Eigen::VectorXd& x, const Eigen::VectorXd& g, const Eigen::VectorXd& p) override {
        double alpha = 1.0;
        double alpha_prev = 0.0;
        const double c1 = 1e-4;
        const double c2 = 0.9;
        const double alpha_max = 10.0;
        
        double phi0 = fun(x);
        double dphi0 = g.dot(p);
        double phi_prev = phi0;

        for (int i = 1; i <= MAX_ITER; ++i) {
            double phi = fun(x + alpha * p);
            
            //armijo e monotonia
            if (phi > phi0 + c1 * alpha * dphi0 || (i > 1 && phi >= phi_prev)) {
                return zoom(fun, x, p, alpha_prev, alpha, phi0, dphi0, c1, c2);
            }
            
            Eigen::VectorXd g_new = gradient(fun, x + alpha * p);
            double dphi = g_new.dot(p);
            
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
    double zoom(const std::function<double(const Eigen::VectorXd&)>& fun, const Eigen::VectorXd& x, const Eigen::VectorXd& p, 
                double alpha_lo, double alpha_hi, 
                double phi0, double dphi0, double c1, double c2) {
        
        for (int i = 0; i < MAX_ITER; ++i) {
           
           double alpha_j = 0.5 * (alpha_lo + alpha_hi);
           
           double phi_j = fun(x + alpha_j * p);
           double phi_lo = fun(x + alpha_lo * p);
           
           //armijo e monotonia
           if (phi_j > phi0 + c1 * alpha_j * dphi0 || phi_j >= phi_lo) {
               alpha_hi = alpha_j;
           } else {
               double dphi_j = gradient(fun, x + alpha_j * p).dot(p);
               
               //curvatura
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

std::unique_ptr<LineSearch> make_line_search(const std::string& s) {
    if (s == "armijo") return std::make_unique<BacktrackingArmijo>();
    else return std::make_unique<StrongWolfe>(); //default strong wolfe
}
} 
