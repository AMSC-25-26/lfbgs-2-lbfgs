
#include "LineSearch.hpp"
#include <Eigen/Dense>
#include <memory>
#include <stdexcept>
#include <string>

#include "Functor.hpp"
#include <cmath>

namespace lfbgs {

class BacktrackingArmijo : public LineSearch {
public:
    double compute(const Functor& obj, const Eigen::VectorXd& x, const Eigen::VectorXd& g, const Eigen::VectorXd& p) override {
        double alpha = 1.0;
        const double c1 = 1e-4;
        const double rho = 0.5;
        
        double fx = obj(x);
        double gtp = g.dot(p);
        

        int max_iter = 100;
        for (int i = 0; i < max_iter; ++i) {
            double f_new = obj(x + alpha * p);
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
    double compute(const Functor& obj, const Eigen::VectorXd& x, const Eigen::VectorXd& g, const Eigen::VectorXd& p) override {
        double alpha = 1.0;
        double alpha_prev = 0.0;
        const double c1 = 1e-4;
        const double c2 = 0.9;
        const double alpha_max = 10.0;
        
        double phi0 = obj(x);
        double dphi0 = g.dot(p);
        double phi_prev = phi0;

        int max_iter = 20;
        for (int i = 1; i <= max_iter; ++i) {
            double phi = obj(x + alpha * p);
            
            if (phi > phi0 + c1 * alpha * dphi0 || (i > 1 && phi >= phi_prev)) {
                return zoom(obj, x, p, alpha_prev, alpha, phi0, dphi0, c1, c2);
            }
            
            Eigen::VectorXd g_new = obj.gradient(x + alpha * p);
            double dphi = g_new.dot(p);
            
            if (std::abs(dphi) <= -c2 * dphi0) {
                return alpha;
            }
            
            if (dphi >= 0) {
                return zoom(obj, x, p, alpha, alpha_prev, phi0, dphi0, c1, c2);
            }
            
            alpha_prev = alpha;
            phi_prev = phi;
            alpha = std::min(alpha * 2.0, alpha_max); 
        }
        return alpha;
    }

private:
    double zoom(const Functor& obj, const Eigen::VectorXd& x, const Eigen::VectorXd& p, 
                double alpha_lo, double alpha_hi, 
                double phi0, double dphi0, double c1, double c2) {
        
        for (int i = 0; i < 20; ++i) {
           
           double alpha_j = 0.5 * (alpha_lo + alpha_hi);
           
           double phi_j = obj(x + alpha_j * p);
           double phi_lo = obj(x + alpha_lo * p);
           
           if (phi_j > phi0 + c1 * alpha_j * dphi0 || phi_j >= phi_lo) {
               alpha_hi = alpha_j;
           } else {
               Eigen::VectorXd g_j = obj.gradient(x + alpha_j * p);
               double dphi_j = g_j.dot(p);
               
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
    if (s == "wolfe")  return std::make_unique<StrongWolfe>();
    throw std::runtime_error("unknown LS");
}
} 
