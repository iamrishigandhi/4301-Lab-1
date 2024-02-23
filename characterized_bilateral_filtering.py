import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = input("Enter the path to your image: ").strip('"')
original_image = cv2.imread(image_path)

# Check if the image is loaded successfully
if original_image is None:
    print("Error: Unable to load the image. Please check the file path.")
    exit()

# Set the size of the Matplotlib plot
plt.figure(figsize=(10, 3))

# Display the original image
plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Bilateral Filter
bilateral_filtered = cv2.bilateralFilter(original_image, d=9, sigmaColor=75, sigmaSpace=75)
plt.subplot(1, 4, 2)
plt.imshow(cv2.cvtColor(bilateral_filtered, cv2.COLOR_BGR2RGB))
plt.title('Bilateral Filtered')

# Gaussian Filter
gaussian_filtered = cv2.GaussianBlur(original_image, (9, 9), 2)
plt.subplot(1, 4, 3)
plt.imshow(cv2.cvtColor(gaussian_filtered, cv2.COLOR_BGR2RGB))
plt.title('Gaussian Filtered')

# this is set according to the test image provided, change it accordingly if using a different image
window1 = original_image[50:150, 100:200]

# Convert the window to float32 before computing FFT
window1_float32 = np.float32(window1)

# Compute FFT magnitude for the original image and filtered versions within the selected window
def compute_fft_magnitude(image):
    f_transform = np.fft.fft2(image)
    spectrum_magnitude = np.abs(np.fft.fftshift(f_transform))
    return spectrum_magnitude

# Compute FFT for the windows
spectrum_original = compute_fft_magnitude(window1_float32)
spectrum_bilateral = compute_fft_magnitude(cv2.bilateralFilter(window1_float32, d=9, sigmaColor=75, sigmaSpace=75))
spectrum_gaussian = compute_fft_magnitude(cv2.GaussianBlur(window1_float32, (9, 9), 2))

# Display the spectral views
plt.subplot(1, 4, 4)
plt.imshow(np.log1p(spectrum_original), cmap='gray')
plt.title('Original Spectrum')

plt.show()