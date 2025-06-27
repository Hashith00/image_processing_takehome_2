import numpy as np
import cv2
import matplotlib.pyplot as plt

def apply_otsu_threshold(image):
    """
    Implements Otsu's algorithm to find the optimal threshold.
    """
    # Convert image to 8-bit integer type
    image = image.astype(np.uint8)

    # Calculate histogram
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()

    # Calculate cumulative sums
    Q = hist_norm.cumsum()

    # Initialize bins
    bins = np.arange(256)

    # Initialize intermediate variables
    fn_min = np.inf
    thresh = -1

    # Iterate through all possible thresholds
    for i in range(1, 256):
        # Probabilities of background and foreground
        p1, p2 = np.hsplit(hist_norm, [i])
        q1, q2 = Q[i-1], Q[255] - Q[i-1]
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue

        # Bins for background and foreground
        b1, b2 = np.hsplit(bins, [i])

        # Means and variances of background and foreground
        m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
        v1, v2 = np.sum(((b1 - m1)**2) * p1) / q1, np.sum(((b2 - m2)**2) * p2) / q2

        # Calculate within-class variance
        fn = v1 * q1 + v2 * q2
        if fn < fn_min:
            fn_min = fn
            thresh = i

    # Apply the found threshold to the image
    ret, otsu_img = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    return otsu_img, thresh

# 1. Create a simple image with 2 objects and a background
# Define pixel values for background and two objects
background_val = 20
object1_val = 120  # A mid-gray ellipse
object2_val = 210  # A bright line

# Create a blank image
image = np.full((300, 300), background_val, dtype=np.uint8)

# Add an ellipse to the image
# cv2.ellipse(image, center, axes, angle, startAngle, endAngle, color, thickness)
cv2.ellipse(image, (150, 100), (80, 40), 0, 0, 360, object1_val, -1)

# Add a thick line to the image
# cv2.line(image, start_point, end_point, color, thickness)
cv2.line(image, (50, 250), (250, 250), object2_val, 15)


# 2. Add Gaussian noise to the image
mean = 0
var = 70
sigma = var**0.5
gaussian_noise = np.random.normal(mean, sigma, image.shape)
noisy_image = np.clip(image + gaussian_noise, 0, 255).astype(np.uint8)

# 3. Implement and test Otsu's algorithm
segmented_image, threshold = apply_otsu_threshold(noisy_image.copy())

# 4. Displaying the results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(noisy_image, cmap='gray')
plt.title('Image with Gaussian Noise')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(segmented_image, cmap='gray')
plt.title(f"Otsu's Segmentation (Threshold={threshold})")
plt.axis('off')

plt.show()

print(f"Optimal threshold found by Otsu's algorithm: {threshold}")