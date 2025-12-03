#include <vector>
#include <utility>
#include <Eigen/Dense>

using namespace Eigen;

template <unsigned int m>
class LBFGS : BFGS {
  
  public:
    LBFGS(const VectorXd &, const std::function<double(const VectorXd &)> &) :  {}
    void run();

  private:
    std::vector<std::pair<const VectorXd &, const VectorXd &>> history; // reserve(m)
    std::vector<double> alpha; // reserve(m)

    double backward_pass(const VectorXd &);
    void forward_pass(double);

};