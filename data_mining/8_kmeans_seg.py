from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

image_path = "image.jpg"
image = Image.open(image_path).convert("RGB")
image = np.array(image)

pixels = image.reshape(-1, 3)

k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(pixels)

segmented_pixels = kmeans.cluster_centers_[kmeans.labels_]
segmented_image = segmented_pixels.reshape(image.shape).astype(np.uint8)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Segmented Image")
plt.imshow(segmented_image)
plt.axis("off")

plt.show()
