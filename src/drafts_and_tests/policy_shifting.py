import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def objective_function(shifted_points, fixed_points, max_derivative):
    x = np.concatenate((fixed_points[:, 0], shifted_points[:, 0]))
    y = np.concatenate((fixed_points[:, 1], shifted_points[:, 1]))
    cs = CubicSpline(x, y)

    # Calculate derivatives
    dx = cs.derivative(nu=1)(x)

    # Calculate the objective function (minimize the maximum derivative violation)
    return -np.min(np.maximum(dx, -max_derivative))


def smooth_curve(coords, max_derivative):
    # Fixed points are the first and last points of the curve
    fixed_points = np.array([coords[0], coords[-1]])

    # Initial guess for shifted points (excluding fixed points)
    init_guess = coords[1:-1]
    print(init_guess)
    # Optimization to find the best position for shifted points
    result = minimize(objective_function, init_guess, args=(fixed_points, max_derivative), method='Nelder-Mead')

    # Concatenate fixed points and shifted points
    shifted_points = np.vstack((coords[0], result.x, coords[-1]))

    return shifted_points


# Example usage
original_curve = np.array([(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
max_derivative = 1.0  # Maximum allowed derivative

shifted_curve = smooth_curve(original_curve, max_derivative)

# Plot original and modified curves
plt.plot(original_curve[:, 0], original_curve[:, 1], 'o-', label='Original Curve')
plt.plot(shifted_curve[:, 0], shifted_curve[:, 1], 'o-', label='Shifted Curve')
plt.legend()
plt.title('Smoothly Modified Curve with Max Derivative Constraint')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
