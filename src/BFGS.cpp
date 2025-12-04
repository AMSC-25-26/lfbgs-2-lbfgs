#include "BFGS.hpp"
#include <MathTools.hpp>
#include <iostream>
#include <Eigen/IterativeLinearSolvers>

using namespace Eigen;

BFGS::BFGS(const VectorXd & x0, const std::function<double(VectorXd const&)>& fun, const double & tol)
{
    //Initialize the initial condition
    x0_ = x0;
    int n=x0_.rows();
    // initialize with identity
    B = MatrixXd::Identity(n, n);
    fun_=fun;
    tol_=tol;
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
                        const VectorXd& y,
                        const VectorXd& s)
{   

    double yBy = y.transpose().dot(B_old*y);
    double sy = s.transpose().dot(y);

    MatrixXd term1 = (( sy + yBy )*(s*s.transpose())) / (sy*sy);
    MatrixXd term2 = ((B_old * y * s.transpose())+(s*y.transpose()* B_old)) / sy;

    return B_old + term1 - term2;
}

void BFGS::updateSolution(VectorXd& x_old, VectorXd& grad_old, const VectorXd& d, VectorXd& s, VectorXd& y) {
        double alpha = 1.0; // line search
        s = alpha * d;
        VectorXd x_new = x_old + s;
        
        VectorXd grad_new = MathTools::gradient(fun_, x_new);
        y = grad_new - grad_old;

        x_old = x_new;
        grad_old = grad_new; 
    }

void BFGS::run(){
    VectorXd x = x0_;
    VectorXd grad = MathTools::gradient(fun_, x);
    VectorXd d, s, y;
    while(grad.norm() > tol_) {
        d = BFGS::computeDirectionP(B, grad);
        updateSolution(x, grad, d, s, y);
        B = updateB(B, y, s);
    }    
}

