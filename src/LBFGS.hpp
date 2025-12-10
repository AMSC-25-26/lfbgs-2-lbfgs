#include <vector>
#include <utility>
#include <Eigen/Dense>

using namespace Eigen;

template <unsigned int m>
class LBFGS : public BFGS {
  
  public:
    LBFGS(const VectorXd &x0, const std::function<double(const VectorXd &)> &fun) :
      x0_(x0),
      fun_(fun)
    {};
    void run() override;

  private:
    std::vector<std::pair<const VectorXd, const VectorXd>> history; // reserve(m)
    std::vector<double> alpha; // reserve(m)

    void backward_pass(const VectorXd &);
    void forward_pass(const VectorXd &);
};