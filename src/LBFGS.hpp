#include <vector>
#include <utility>
#include <Eigen/Dense>

using namespace Eigen;

template <unsigned int m>
class LBFGS : public BFGS {
  
  public:
    LBFGS(const VectorXd &, const std::function<double(const VectorXd &)> &, const double &) :  {}
    void run() override;

  private:
    std::vector<std::pair<const VectorXd, const VectorXd>> history; // reserve(m)
    std::vector<double> alpha; // reserve(m)

    void backward_pass(const VectorXd &);
    void forward_pass(const VectorXd &);
};