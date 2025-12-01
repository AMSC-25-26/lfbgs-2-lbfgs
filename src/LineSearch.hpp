
#pragma once
#include <Eigen/Dense>

#include <memory>
#include <string>

namespace lfbgs {

class Objective;

class LineSearch {
public:
    virtual ~LineSearch() = default;
    virtual double compute(const Objective&, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&) = 0;
};

std::unique_ptr<LineSearch> make_line_search(const std::string& s);

}
