import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from PIL import Image

try:
    # Load an image 
    image_path = input("Enter the path to your image: ").strip('"')
    image = np.array(Image.open(image_path).convert('L'))  # Convert to grayscale

except FileNotFoundError:
    print(f"Error: The file '{image_path}' was not found.")
    exit()

except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()

# Perform 1D FFT using numpy
fft_result_np = fft(image[0, :])

# Naive implementation of 1D DFT
def naive_dft(signal):
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    return np.dot(e, signal)

# Naive implementation of 1D IDFT
def naive_idft(spectrum):
    N = len(spectrum)
    k = np.arange(N)
    n = k.reshape((N, 1))
    e = np.exp(2j * np.pi * k * n / N)
    return np.real(np.dot(e, spectrum) / N)

# Perform naive 1D DFT
dft_result_naive = naive_dft(image[0, :])

# Perform simple filtering (attenuate one component by half)
dft_result_naive[10] = dft_result_naive[10] / 2

# Perform naive 1D IDFT
filtered_signal_naive = naive_idft(dft_result_naive)

# Plotting
plt.figure(figsize=(12, 6))

# Plot original image
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')

# Plot numpy FFT result
plt.subplot(2, 2, 2)
plt.title('NumPy FFT Result')
plt.plot(np.abs(fft_result_np))
plt.xlabel('Frequency')

# Plot naive DFT result
plt.subplot(2, 2, 3)
plt.title('Naive DFT Result')
plt.plot(np.abs(dft_result_naive))
plt.xlabel('Frequency')

# Plot filtered image using naive IDFT
plt.subplot(2, 2, 4)
plt.title('Filtered Image (Naive)')
# Reshape to 2D array
filtered_signal_2d = np.reshape(filtered_signal_naive.real, (1, -1))
plt.imshow(filtered_signal_2d, cmap='gray')

plt.tight_layout()
plt.show()