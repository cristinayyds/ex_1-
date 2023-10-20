import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity


def detect(img_corrupted, img_inpainted):
    # Convert images to grayscale
    img_corrupted_gray = cv2.cvtColor(img_corrupted, cv2.COLOR_BGR2GRAY)
    img_inpainted_gray = cv2.cvtColor(img_inpainted, cv2.COLOR_BGR2GRAY)
    # Compute SSIM between two images
    (score, diff) = structural_similarity(img_corrupted_gray, img_inpainted_gray, full=True)
    print("Image similarity", score)
    # The diff image contains the actual image differences between the two images
    # and is represented as a floating point data type in the range [0,1]
    # so we must convert the array to 8-bit unsigned integers in the range
    # [0,255] before we can use it with OpenCV
    diff = (diff * 255).astype("uint8")
    # Threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    detection = img_corrupted.copy()
    for c in contours:
        area = cv2.contourArea(c)
        if area > 50:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(detection, (x, y), (x + w, y + h), (36, 255, 12), 2)
            # cv2.drawContours(mask, [c], 0, (0, 255, 0), -1)
            # cv2.drawContours(filled_after, [c], 0, (0, 255, 0), -1)
    return detection


img1_original = cv2.imread('../outputs(win_size7)/original_0.png')
title1 = "original"
plt.subplot(1, 5, 1)
plt.title(title1, fontsize=8)
plt.xticks([])
plt.yticks([])
plt.imshow(img1_original)
img1_corrupted = cv2.imread('../outputs(win_size7)/corrupted_0.png')
title2 = "corrupted"
plt.subplot(1, 5, 2)
plt.title(title2, fontsize=8)
plt.xticks([])
plt.yticks([])
plt.imshow(img1_corrupted)
img1_inpaint = cv2.imread('../outputs(win_size7)/output_0.png')
title3 = "inpaint"
plt.subplot(1, 5, 3)
plt.title(title3, fontsize=8)
plt.xticks([])
plt.yticks([])
plt.imshow(img1_inpaint)
img1_blended = cv2.imread('../outputs(win_size7)/blended_0.png')
title4 = "blended"
plt.subplot(1, 5, 4)
plt.title(title4, fontsize=8)
plt.imshow(img1_blended)
plt.xticks([])
plt.yticks([])
result = detect(img1_corrupted, img1_blended)
title5 = "detection"
plt.subplot(1, 5, 5)
plt.title(title5, fontsize=8)
plt.imshow(result)
plt.xticks([])
plt.yticks([])
plt.show()


