import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread('images/image_2.jpg')
if image is None:
    print("Error: Image not found!")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Function to add salt & pepper noise
def add_sp_noise(image, prob):
    noisy = np.copy(image)
    h, w, c = image.shape
    
    # Salt noise (white)
    num_salt = int(prob * h * w / 2)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 255

    # Pepper noise (black)
    num_pepper = int(prob * h * w / 2)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy[coords[0], coords[1]] = 0

    return noisy

# Add noise
noise_10 = add_sp_noise(image, 0.1)
noise_20 = add_sp_noise(image, 0.2)

#Apply median filtering for 10% noise
median_3_10 = cv2.medianBlur(noise_10, 3)
median_5_10 = cv2.medianBlur(noise_10, 5)
median_11_10 = cv2.medianBlur(noise_10, 11)

# Apply median filtering for 20% noise
median_3_20 = cv2.medianBlur(noise_20, 3)
median_5_20 = cv2.medianBlur(noise_20, 5)
median_11_20 = cv2.medianBlur(noise_20, 11)

# Display
plt.figure(figsize=(12,6))

plt.subplot(3,3,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(3,3,4)
plt.imshow(noise_10)
plt.title("10% Salt and Pepper Noise Image")
plt.axis('off')

plt.subplot(3,3,7)
plt.imshow(noise_20)
plt.title("20% Salt and Pepper Noise Image")
plt.axis('off')

plt.subplot(3,3,2)
plt.imshow(median_3_10)
plt.title("Median 3x3 for 10% Noise")
plt.axis('off')

plt.subplot(3,3,5)
plt.imshow(median_5_10)
plt.title("Median 5x5 for 10% Noise")
plt.axis('off')

plt.subplot(3,3,8)
plt.imshow(median_11_10)
plt.title("Median 11x11 for 10% Noise")
plt.axis('off')

plt.subplot(3,3,3)
plt.imshow(median_3_20)
plt.title("Median 3x3 for 20% Noise")
plt.axis('off')

plt.subplot(3,3,6)
plt.imshow(median_5_20)
plt.title("Median 5x5 for 20% Noise")
plt.axis('off')

plt.subplot(3,3,9)
plt.imshow(median_11_20)
plt.title("Median 11x11 for 20% Noise")
plt.axis('off')

plt.tight_layout()
plt.show()