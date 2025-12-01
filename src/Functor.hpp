#pragma once

#include <Eigen/Dense>

namespace lfbgs {

class Functor {
public:
    virtual ~Functor() = default;

    /// Valore della funzione in x
    virtual double operator()(const Eigen::VectorXd& x) const = 0;

    /// Gradiente della funzione in x
    virtual Eigen::VectorXd gradient(const Eigen::VectorXd& x) const = 0;
};

} // namespace lfbgs
