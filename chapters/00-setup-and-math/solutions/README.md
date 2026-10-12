# Chapter 00: Exercise Solutions

This folder contains the complete solutions for the Chapter 00 exercises.

### Running the Python Solutions
```bash
python chapters/00-setup-and-math/solutions/exercise_solutions.py
```

### Summary of Key Results

1. **Exercise 1 (Broadcasting Bug)**:
   - Adding shapes `(5,)` and `(5, 1)` yields `(5, 5)` because NumPy's broadcasting rules align trailing dimensions and stretch singleton dimensions.
   - Always be intentional with shapes using `a.reshape(-1, 1)`.

2. **Exercise 2 (Rational Function)**:
   - Analytical derivative: $f'(x) = \frac{x^2 - 6x - 1}{(x - 3)^2}$.
   - At $x = 5.0$, $f'(5) = -1.5$.
   - Numerical derivative error $< 10^{-10}$.

3. **Exercise 3 (Multivariate Chain Rule)**:
   - $\nabla L(2, -1) = [39.0, 24.0]$.
   - Hand derivation verified exactly with central difference checking.

4. **Exercise 4 (Paraboloid Gradient Descent)**:
   - Minimum is at $(x^*, y^*) = (3.0, -4.0)$.
   - 50 iterations with learning rate $\eta = 0.1$ converges smoothly to the global minimum.

5. **Exercise 5 (PyTorch Autograd)**:
   - PyTorch `L.backward()` yields exactly $1188.0$, matching our hand derivative.
