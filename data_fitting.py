import numpy as np
import matplotlib.pyplot as plt

# Step 1: Generate a random function
def random_function(x):
    return np.sin(2 * np.pi * x / 10) + 0.5 * np.sin(4 * np.pi * x / 15) + np.random.normal(0, 0.2, x.shape)

# Step 2: Sample the function at a few dozen random locations
np.random.seed(42)  # Setting seed for reproducibility
num_samples = 30
x_samples = np.sort(np.random.uniform(0, 30, num_samples))
y_samples = random_function(x_samples)

# Step 3: Fit a function using least squares polynomial fit
degree = 3  # Choose the degree of the polynomial
X_matrix = np.vander(x_samples, degree + 1, increasing=True)
weights = np.linalg.lstsq(X_matrix, y_samples, rcond=None)[0]

# Step 4: Measure fitting error
x_grid = np.linspace(0, 30, 1000)  # Grid for evaluating the fitted function
X_grid = np.vander(x_grid, degree + 1, increasing=True)
y_fit = np.dot(X_grid, weights)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(x_samples, y_samples, label='Sampled Data', color='blue')
plt.plot(x_grid, random_function(x_grid), label='True Function', linestyle='--', color='green')
plt.plot(x_grid, y_fit, label='Fitted Polynomial', color='red')
plt.legend()
plt.title('Least Squares Polynomial Fit')
plt.show()

# Measure fitting error
fitting_error = np.mean((random_function(x_grid) - y_fit)**2)
print(f"Fitting Error: {fitting_error}")