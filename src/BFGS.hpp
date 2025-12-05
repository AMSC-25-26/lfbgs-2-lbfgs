#ifndef BFGS_HPP
#define BFGS_HPP

#include <Eigen/Dense>
#include <functional>

class BFGS
{
public:
    // Constructor
    BFGS(const Eigen::VectorXd& x0, const std::function<double(VectorXd const&)>& fun, const double & tol);

      // Update Hessian approximation:
      // Inputs:
      //   gamma : difference between grad(x_k+1) and grad(x_k)
      //   delta: difference between x_k+1 and x_k
      //Output:
      //   B_new : current aproximation
      void updateB(const VectorXd&, const VectorXd&);

      //Update the direction p_k:
      //Inputs:
      //   grad  : gradient at x_k
      //Output:
      //   p_new : current direction p_k
      VectorXd computeDirectionP(const VectorXd&);
      
      void updateSolution(VectorXd&, VectorXd&, const VectorXd&, VectorXd&, VectorXd&);
      virtual void run();


  private:
      std::function<double(VectorXd const&)> fun_; //Function
      VectorXd x0_;   //Initial condition
      MatrixXd B; //Hessian Aproximation

      double tol_;
};

#endif