#include <vector>
#include <utility>
#include <Eigen/Dense>
#include "BFGS.hpp"
#include "utils/MathTools.hpp"

using namespace Eigen;

template <unsigned int m>
class LBFGS : public BFGS {
  
  public:
    LBFGS(const VectorXd &x0, const std::function<double(const VectorXd &)> &fun, double tol,
          lfbgs::LineSearchType type) :
      BFGS(x0, fun, tol, type)
    {};
    
    void run() override;

  private:
    std::vector<std::pair<VectorXd, VectorXd>> history; 
    std::vector<double> alpha;

    void backward_pass(VectorXd &q);
    void forward_pass(VectorXd &r);
};


template <unsigned int m>
void LBFGS<m>::backward_pass(VectorXd &q) {
  for (int i = history.size()-1; i >= 0; --i) {
    double rho = 1.0 / (history[i].second.dot(history[i].first));
    alpha[i] = rho * (history[i].first.dot(q));
    q -= alpha[i] * history[i].second;
  }
}

template <unsigned int m>
void LBFGS<m>::forward_pass(VectorXd &r) {
  for (size_t i = 0; i < history.size(); ++i) {
    double rho = 1.0 / (history[i].second.dot(history[i].first));
    double beta = rho * (history[i].second.dot(r));
    r += (alpha[i] - beta) * history[i].first;
  }
}

template <unsigned int m>
void LBFGS<m>::run() override {
  history.reserve(m);
  alpha.resize(m);

  int iter = 0;
  VectorXd x = x0_;
  VectorXd grad = MathTools::gradient(fun_, x);
  VectorXd d, s, y;
  
  while(grad.norm() > tol_) {

    if (iter < m) {
      d = computeDirectionP(grad);
    } else {
      VectorXd q = grad;
      backward_pass(q);
      VectorXd r = B * q;
      forward_pass(r);
      d = -r;
    }

    updateSolution(x, grad, d, s, y, type_);

    if(history.size() == m) {
      history.erase(history.begin());
    }
    history.push_back({s, y});

    ++iter;
  }
  
  solution_ = x;
}