
#pragma once
#include <memory>
#include <string>
#include <cmath>
#include <functional>
#include <Eigen/Dense>
#include "MathTools.hpp"

namespace lfbgs {


class LineSearch {
public:
    virtual ~LineSearch() = default;
    virtual double compute(const std::function<double(const Eigen::VectorXd&)>& fun, const Eigen::VectorXd&, const Eigen::VectorXd&, const Eigen::VectorXd&) = 0;
};

std::unique_ptr<LineSearch> make_line_search(const std::string& s);

}
