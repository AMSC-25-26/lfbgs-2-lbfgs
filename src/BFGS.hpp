#ifndef BFGS_HPP
#define BFGS_HPP

#include <Eigen/Dense>
#include <functional>

class BFGS
{
public:
    // Constructor
    BFGS(const Eigen::VectorXd& x0, int n);

    // Update Hessian approximation:
    // Inputs:
    //   B_old : previous aproximation
    //   x_old : previous solution  
    //   g_old : gradient at x_k
    //   p     : search direction
    //   fun   : function to analyze
    //   alpha : step size
    //Output:
    //   B_new : current aproximation
    Eigen::MatrixXd updateB( const Eigen::MatrixXd& B_old,
                            const Eigen::VectorXd& x_old,
                            const Eigen::VectorXd& g_old,
                            const Eigen::VectorXd& p,
                            const std::function<double(VectorXd const&)>& fun,
                            double alpha);

    //Update the direction p_k:
    //Inputs:
    //   B_old : previous aproximation 
    //   grad  : gradient at x_k
    //Output:
    //   p_new : current direction p_k
    Eigen::VectorXd computeDirectionP( const MatrixXd& B_old,
                                       const VectorXd& grad);


private:
    Eigen::MatrixXd B;   // Hessian approximation
    Eigen::VectorXd x0;   //Initial condition

};

#endif