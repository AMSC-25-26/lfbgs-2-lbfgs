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
    //   B_old : previous aproximation
    //   y : difference between grad(x_k+1) and grad(x_k)
    //   s: difference between x_k+1 and x_k
    //Output:
    //   B_new : current aproximation
    Eigen::MatrixXd updateB( const Eigen::MatrixXd& B_old,
                            const Eigen::VectorXd& y,
                            const Eigen::VectorXd& s);

    //Update the direction p_k:
    //Inputs:
    //   B_old : previous aproximation 
    //   grad  : gradient at x_k
    //Output:
    //   p_new : current direction p_k
    Eigen::VectorXd computeDirectionP( const  Eigen::MatrixXd& B_old,
                                       const Eigen::VectorXd& grad);
    
    void updateSolution(VectorXd& x_old, VectorXd& grad_old, const VectorXd& d, VectorXd& s, VectorXd& y);
    virtual void run();


private:
    std::function<double(VectorXd const&)> fun_; //Function
    Eigen::VectorXd x0_;   //Initial condition
    Eigen::MatrixXd B; //Hessian Aproximation
    double tol_;
};

#endif