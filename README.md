# LFBGS

Implementation of BFGS and L-BFGS (Limited-memory BFGS) optimization algorithms in C++. This project explores these quasi-Newton methods for solving unconstrained non-linear optimization problems.

See the description in `papers/lfbgs.pdf` for theoretical background.

## Features

- **BFGS Algorithm**: Full Broyden–Fletcher–Goldfarb–Shanno algorithm implementation.
- **L-BFGS Algorithm**: Limited-memory version suitable for large-scale problems.
- **Line Search**: Custom line search implementation satisfying Wolfe conditions.
- **Eigen3 Integration**: Uses the Eigen3 linear algebra library for efficient matrix operations.

## Code Design

The project follows a modular and object-oriented design, leveraging C++20 features and the Eigen library.

### Class Hierarchy

*   **`BFGS` (Base Class)**:
    *   Implements the standard BFGS algorithm.
    *   Manages the optimization state: current iterate ($x_k$), gradient ($\nabla f(x_k)$), and the full Hessian approximation matrix ($B_k$).
    *   **Key Methods**:
        *   `run()`: The main optimization loop.
        *   `computeDirectionP()`: Solves the linear system $B_k p_k = -\nabla f(x_k)$ to find the search direction.
        *   `updateB()`: Updates the Hessian approximation using the rank-two BFGS formula.

*   **`LBFGS<m>` (Derived Class)**:
    *   Inherits from `BFGS`.
    *   Implements the Limited-memory BFGS algorithm.
    *   **Template Parameter `m`**: Specifies the memory size (number of past correction pairs to store) at compile time.
    *   **Storage**: Uses a `std::deque` to efficiently manage the history of correction pairs $(s_k, y_k)$.
    *   **Two-Loop Recursion**: Overrides the `run()` method to compute the search direction using the efficient two-loop recursion algorithm, avoiding the storage and inversion of the full Hessian matrix.

### Utilities

*   **`Gradient`**: A helper class/tool responsible for computing the gradient of the objective function (likely using finite differences if an analytical gradient is not provided).
*   **`LineSearch`**: Implements line search strategies (Armijo and Strong Wolfe conditions) to determine the step size $\alpha_k$ that ensures sufficient decrease in the objective function.

### Dependencies & Data Structures

*   **Eigen3**: Used extensively for `VectorXd` and `MatrixXd` types, ensuring high-performance linear algebra operations.
*   **`std::function`**: The objective function is passed as a `std::function<double(Vector const&)>`, allowing flexibility in defining the optimization problem (e.g., using lambdas or functors).


## Building the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/AMSC-25-26/lfbgs-2-lbfgs
   cd lfbgs-2-lbfgs
   ```

2. Create a build directory:
   ```bash
   mkdir build
   cd build
   ```

3. Configure the project with CMake:
   ```bash
   cmake ..
   ```

4. Build the executables:
   ```bash
   make
   ```

## Running Tests

The build process generates several test executables in the `build` directory:

- `./test_bfgs`: Tests the standard BFGS implementation.
- `./test_lbfgs`: Tests the L-BFGS implementation.
- `./test_gradient`: Verifies gradient calculations.
- `./test_linesearch`: Tests the line search logic.

## Project Structure

- `src/`: Source code for the library.
  - `BFGS.cpp` / `BFGS.hpp`: BFGS implementation.
  - `LBFGS.hpp`: L-BFGS implementation.
  - `utils/`: Utility functions, including `LineSearch`.
- `test/`: Test sources for validating the algorithms.
- `papers/`: Documentation and reference papers.
