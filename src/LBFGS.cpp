#include "LBFGS.hpp"
#include "MathTools.hpp"

template <unsigned int m>
double LBFGS<m>::backward_pass(const VectorXd &q) {
  for (int i=history.size()-1; i>=0; --i) {
    double rho = 1./(history[i]..dot(history[i].first));
    alpha[i] = rho * (history[i].first.dot());
    q = q - alpha[i] * history[i].second;
  }
}