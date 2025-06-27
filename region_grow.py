import cv2
import numpy as np
import matplotlib.pyplot as plt

def region_growing(image, seeds, threshold):
    """
    Performs image segmentation using the region growing algorithm.
    """
    height, width = image.shape
    # Create an output image initialized to zeros
    segmented_mask = np.zeros((height, width, 1), np.uint8)

    # List to store pixels to be checked
    points_to_check = []
    for seed in seeds:
        points_to_check.append(seed)
        # Mark the seed point itself in the mask
        segmented_mask[seed[0], seed[1]] = 255


    # Get the intensity of the first seed point to compare against
    # It's better to average if multiple seeds are used, but for one seed this is fine.
    seed_intensity = image[seeds[0][0], seeds[0][1]]

    # Array to keep track of processed pixels
    processed = np.zeros_like(image, dtype=bool)

    # Loop as long as there are points to check
    while points_to_check:
        # Get the next point to process
        current_point = points_to_check.pop(0)
        x, y = current_point

        # Skip if the point has already been processed
        if processed[x, y]:
            continue

        # Mark the current point as processed
        processed[x, y] = True

        # Check its 8-connected neighbors
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                
                neighbor_x, neighbor_y = x + i, y + j

                # Check if the neighbor is within the image boundaries
                if 0 <= neighbor_x < height and 0 <= neighbor_y < width:
                    # Check if the neighbor has been processed
                    if not processed[neighbor_x, neighbor_y]:
                        # Check if the neighbor's intensity is within the threshold
                        if abs(int(image[neighbor_x, neighbor_y]) - int(seed_intensity)) <= threshold:
                            # Add the pixel to the segmented region
                            segmented_mask[neighbor_x, neighbor_y] = 255
                            # Add the neighbor to the list to check
                            points_to_check.append((neighbor_x, neighbor_y))
                            # Mark as processed immediately to avoid re-adding
                            processed[neighbor_x, neighbor_y] = True

    return segmented_mask

# --- Main execution ---
# Create a new sample image for demonstration
# A dark background with two separate objects of different intensities
sample_image = np.full((250, 250), 30, dtype=np.uint8) # Dark background

# Add a mid-gray diamond shape
pts = np.array([[125, 40], [190, 125], [125, 210], [60, 125]], np.int32)
pts = pts.reshape((-1, 1, 2))
cv2.fillPoly(sample_image, [pts], 100) # Mid-gray diamond

# Add a bright triangle
pts2 = np.array([[50, 50], [90, 20], [90, 80]], np.int32)
pts2 = pts2.reshape((-1, 1, 2))
cv2.fillPoly(sample_image, [pts2], 200) # Bright triangle


# 1. Start from a set of points (seeds) inside the object of interest
# Let's try to segment the diamond shape
seed_points = [(125, 125)]

# 2. Define a pre-defined range (threshold)
# The diamond has intensity 100. A threshold of 40 will not jump to the
# background (30) or the triangle (200).
intensity_threshold = 40

# 3. Run the region growing algorithm
segmented_image = region_growing(sample_image, seed_points, intensity_threshold)

# Displaying the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(sample_image, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image, cmap='gray')
plt.title('Segmented by Region Growing')
plt.axis('off')

plt.show()