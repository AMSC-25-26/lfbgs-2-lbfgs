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

VectorXd BFGS::computeDirectionP(const VectorXd& grad)
{
  Eigen::ConjugateGradient<MatrixXd, Eigen::Lower|Eigen::Upper> cg;
  cg.compute(B);

  // If B is not SPD, CG will not converge
  // This condition should NEVER be true (from theory)
  if (cg.info() != Eigen::Success)
    throw std::runtime_error("CG failed: B is not SPD.");

  // Solve B * p = -grad
  VectorXd p = cg.solve(-grad);
  
  return p;
}


void BFGS::updateB(
                        const VectorXd& gamma,
                        const VectorXd& delta)
{   
  double yBy = gamma.transpose().dot(B*gamma);
  double sy = delta.transpose().dot(gamma);

  MatrixXd term1 = (( sy + yBy )*(delta*delta.transpose())) / (sy*sy);
  MatrixXd term2 = ((B * gamma * delta.transpose())+(delta*gamma.transpose()* B)) / sy;

  B += term1 - term2;
}


void BFGS::updateSolution(VectorXd& x_old, VectorXd& grad_old, const VectorXd& d, VectorXd& delta, VectorXd& gamma) {
  double alpha = 1.0; // line search

  delta = alpha * d;
  VectorXd x_new = x_old + delta;
  
  VectorXd grad_new = MathTools::gradient(fun_, x_new);
  gamma = grad_new - grad_old;

  x_old = x_new;
  grad_old = grad_new; 
}


void BFGS::run() {
  unsigned int iter = 0;

  VectorXd x = x0_;
  VectorXd grad = MathTools::gradient(fun_, x);
  VectorXd d, delta, gamma;

  while(grad.norm() > tol_) {
    d = BFGS::computeDirectionP(grad);
    updateSolution(x, grad, d, delta, gamma);
    updateB(gamma, delta);

    ++iter;
  }
}