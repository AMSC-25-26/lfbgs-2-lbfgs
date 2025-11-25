#include <BFGS.hpp>
#include <MathTools.hpp>
#include <iostream>

using namespace Eigen;

BFGS::BFGS(VectorXd x0_, int n)
{
    //Initialize the initial condition
    x0 = x0_;
    // initialize with identity
    B = MatrixXd::Identity(n, n);
}

VectorXd BFGS::computeDirectionP( const MatrixXd& B,
                                  const VectorXd& grad)
{
    Eigen::ConjugateGradient<Eigen::MatrixXd, Eigen::Lower|Eigen::Upper> cg;

    cg.compute(B);

    // If B is not SPD, CG will not converge
    //This condition should NEVER be true (from theory)
    if (cg.info() != Eigen::Success) {
        throw std::runtime_error("CG failed: B is not SPD.");
    }

    // Solve B * p = -grad
    Eigen::VectorXd p = cg.solve(-grad);

    if (cg.info() != Eigen::Success) {
        throw std::runtime_error("CG failed to solve system.");
    }

    return p;

}

MatrixXd BFGS::updateB( const MatrixXd& B_old,
                        const VectorXd& x_old, 
                        const VectorXd& g_old,
                        const VectorXd& p,
                        const std::function<double(VectorXd const&)>& fun,
                        double alpha)
{
    VectorXd s = alpha * p;
    VectorXd x_new = x_old + s;
    const VectorXd g_new = gradient(fun, x_new);      
    VectorXd y = g_new - g_old;       

    double yBy = y.transpose().dot(B_old*y);
    double sy = s.transpose().dot(y);

    MatrixXd term1 = (( sy + yBy )*(s*s.transpose())) / (sy*sy);
    MatrixXd term2 = ((B_old * y * s.transpose())+(s*y.transpose()* B_old)) / sy;

    return B_old + term1 - term2;
}

