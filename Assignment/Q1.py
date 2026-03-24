import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("images/image_1.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

if image is None:
    print("Error: Image not found!")
    exit()

#apply avaerage filtering
blur_3 = cv2.blur(image, (3, 3))
blur_5 = cv2.blur(image, (5, 5))
blur_11 = cv2.blur(image, (11, 11))
blur_15 = cv2.blur(image, (15, 15))


# Display results
plt.figure(figsize=(10,8))

plt.subplot(2,3,1)
plt.imshow(image)
plt.title("Original Image")
plt.axis('off')

plt.subplot(2,3,2)
plt.imshow(blur_3)
plt.title("Filtered Image with 3x3 Kernel")
plt.axis('off')

plt.subplot(2,3,3)
plt.imshow(blur_5)
plt.title("Filtered Image with 5x5 Kernel")
plt.axis('off')

plt.subplot(2,3,4)
plt.imshow(blur_11)
plt.title("Filtered Image with 11x11 Kernel")
plt.axis('off')

plt.subplot(2,3,5)
plt.imshow(blur_15)
plt.title("Filtered Image with 15x15 Kernel")
plt.axis('off')

plt.tight_layout()
plt.show()



#answer to question 1:
# As the kernel size increases, the image becomes more blurred. 
# This is because a larger kernel averages more neighboring pixels, which results in a smoother and less detailed image. 
# The 3x3 kernel retains more details and edges compared to the 15x15 kernel, which produces a much softer and more blurred image. 
# The 5x5 and 11x11 kernels show intermediate levels of blurring, with the 11x11 kernel producing a more noticeable blur than the 5x5 kernel. 
# Overall, the choice of kernel size depends on the desired level of blurring and the specific application, with larger kernels being more effective for reducing noise but also potentially losing important details.

# Effect of Increasing Kernel Size in Average Filtering

# Average filtering replaces each pixel value with the average of its neighboring pixels within the kernel.

# When the kernel size is small (3×3), only a small neighborhood is considered, resulting in slight smoothing while preserving most image details.
# As the kernel size increases (5×5, 11×11), more neighboring pixels are included, leading to stronger blurring and noise reduction.
# With a very large kernel (15×15), the image becomes highly blurred, causing loss of edges and fine details.
# 🔹 Conclusion:

# Increasing the kernel size:

# ✔ Increases smoothing effect
# ✔ Reduces noise
# ❌ Reduces sharpness and details
