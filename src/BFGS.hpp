#ifndef BFGS_HPP
#define BFGS_HPP

#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <functional>
#include "LineSearch.hpp"

using namespace Eigen;

class BFGS {
  public:

      BFGS(const VectorXd & x0, const std::function<double(VectorXd const&)>& fun, const double & tol, lfbgs::LineSearchType type) :
        x0_(x0),
        B(MatrixXd::Identity(x0.rows(), x0.rows())), 
        fun_(fun), 
        tol_(tol),
        type_(type) 
      {};
      
      void updateB(const VectorXd&, const VectorXd&);

      VectorXd computeDirectionP(const VectorXd&);
      
      void updateSolution(VectorXd&, VectorXd&, const VectorXd&, VectorXd&, VectorXd&, lfbgs::LineSearchType type);
      
      virtual void run();

      Eigen::VectorXd getCurrentX() const;


  protected:
      std::function<double(VectorXd const&)> fun_; //Function
      VectorXd x0_;   //Initial condition
      VectorXd solution_ = x0_;
      MatrixXd B; //Hessian Aproximation

      double tol_;
      lfbgs::LineSearchType type_;
};

#endif