#include "BFGS.hpp"
#include "LineSearch.hpp"
#include <iostream>
//#include "LBFGS.hpp"

// Main function.
int
main(int /*argc*/, char * /*argv*/[])
{
  // Definisci la funzione da minimizzare
    auto myFunc = [](const Eigen::VectorXd &x) -> double {
        return std::pow(x[0] - 2.0, 2) + std::pow(x[1] + 3.0, 2);
    };

    // Punto iniziale
    Eigen::VectorXd x0(2);
    x0 << 0.0, 0.0;
    
    double tol = 1e-6;

    // Crea l’oggetto BFGS
    BFGS optimizer(x0, myFunc, tol, lfbgs::LineSearchType::BacktrackingArmijo);

    // Esegui l’ottimizzazione
    optimizer.run();

    // Stampa il risultato finale
    std::cout << "Minimo trovato in: " << optimizer.getCurrentX().transpose() << std::endl;
    std::cout << "Valore funzione: " << myFunc(optimizer.getCurrentX()) << std::endl;

    return 0;

  return 0;
}