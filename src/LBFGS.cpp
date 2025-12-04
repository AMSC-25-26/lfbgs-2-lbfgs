#include "LBFGS.hpp"
#include "MathTools.hpp"

template <unsigned int m>
VectorXd LBFGS<m>::backward_pass(const VectorXd &q) {
  for (int i=history.size()-1; i>=0; --i) {
    double rho = 1./(history[i]..dot(history[i].first));
    alpha[i] = rho * (history[i].first.dot());
    q = q - alpha[i] * history[i].second;
  }
  return q
}

template <unsigned int m>
VectorXd LBFGS<m>::forward_pass(const VectorXd &r) {
  for (int i=0; i<=history.size()-1; ++i) {
    double rho = 1./(history[i].dot(history[i].first));
    double beta = rho * (history[i].second.dot(r));
    r = r + (alpha[i]-beta) * history[i].first;
  }
  return r;
}

template <unsigned int m>
void LBFGS<m>::run() {
    int iter = 0;
    VectorXd x = x0_;
    VectorXd grad = MathTools::gradient(fun_, x);
    VectorXd d, s, y, q;
    while(grad.norm() > tol_) {
      if (iter < m)
        d = BFGS::computeDirectionP(B, grad);
      else {
        q = grad;
        q = backward_pass(q);
        VectorXd r = B*q;
        d = -forward_pass(r);
      }
        updateSolution(x, grad, d, s, y);
        if(history.size()==m)
          history.emplace_back(history.begin());
        history.push_back(s, y);  
        ++iter;
    }     
      
}