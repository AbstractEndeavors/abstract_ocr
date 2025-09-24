import cv2
import numpy as np
import pytesseract
from PIL import Image
import os

# Path to the image file
image_path = "/home/computron/Pictures/linetest.png"
output_dir = "output_lines"
os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

# Step 1: Load the image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# Step 2: Preprocess the image to improve OCR accuracy
# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Binarize the image using adaptive thresholding for better contrast
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Step 3: Use pytesseract to detect words with high confidence
# Convert OpenCV image to PIL image for pytesseract
pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Get detailed OCR data with bounding boxes and confidence scores
custom_config = r'--oem 3 --psm 6'
ocr_data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT, config=custom_config)

# Step 4: Identify rows containing high-confidence words
height, width = binary.shape
word_rows = []  # Store (start_row, end_row, height) for each word
confidence_threshold = 80  # Confidence threshold for word detection

# Debug: Print detected words and their confidence scores
print("Detected words and confidence scores:")
for i in range(len(ocr_data['text'])):
    text = ocr_data['text'][i].strip()
    conf = float(ocr_data['conf'][i])
    # A "word" should have at least 2 characters and contain at least one letter
    if (text and conf >= confidence_threshold and len(text) >= 2 and 
        any(c.isalpha() for c in text)):
        y = ocr_data['top'][i]
        h = ocr_data['height'][i]
        print(f"Word: '{text}', Confidence: {conf}, Top: {y}, Height: {h}")
        # Store the start and end rows for this word
        start_row = max(0, y)
        end_row = min(height, y + h)
        word_rows.append((start_row, end_row, h))

# Step 5: Fallback if no high-confidence words are detected
if not word_rows:
    print("No high-confidence words detected. Falling back to projection-based detection.")
    projection = np.sum(binary, axis=1)
    text_row_threshold = 255 * width * 0.1  # Rows with at least 10% black pixels are considered text
    for y in range(height):
        if projection[y] < text_row_threshold:  # Row contains significant black pixels (text)
            word_rows.append((y, y + 1, 1))  # Treat each row as a "word" with height 1

# Step 6: Compute the vertical projection profile to find white space gaps
projection = np.sum(binary, axis=1)
white_space_threshold = 255 * width * 0.9  # 90% of the maximum possible sum for a row
gaps = []
current_gap_start = None

for y in range(height):
    if projection[y] >= white_space_threshold:  # Row is mostly white
        if current_gap_start is None:
            current_gap_start = y
    else:
        if current_gap_start is not None:
            gap_end = y
            gaps.append((current_gap_start, gap_end))
            current_gap_start = None

if current_gap_start is not None:
    gaps.append((current_gap_start, height))

# Step 7: Group word rows into lines based on strict proximity
word_rows = sorted(word_rows, key=lambda x: x[0])  # Sort by start_row
line_groups = []
current_group = [word_rows[0]]

for i in range(1, len(word_rows)):
    prev_start, prev_end, prev_height = current_group[-1]
    curr_start, curr_end, curr_height = word_rows[i]

    # Check if the current word row is part of the same line
    # They should overlap or be within the height of the previous word
    if curr_start <= prev_end or curr_start <= prev_end + prev_height:
        current_group.append(word_rows[i])
    else:
        # Check if there's a significant gap between the previous end and current start
        gap_found = False
        for gap_start, gap_end in gaps:
            if prev_end < gap_start < curr_start:
                gap_found = True
                break
        if gap_found:
            line_groups.append(current_group)
            current_group = [word_rows[i]]
        else:
            current_group.append(word_rows[i])

# Add the last group
line_groups.append(current_group)

# Step 8: Define line boundaries by expanding around each group
line_boundaries = []
for group in line_groups:
    # Get the start and end rows of the group
    group_start = min(start for start, _, _ in group)
    group_end = max(end for _, end, _ in group)

    # Expand the line boundaries to the nearest gap or image edge
    upper_bound = 0
    lower_bound = height

    # Find the nearest gap above group_start
    for gap_start, gap_end in gaps:
        if gap_end <= group_start:
            upper_bound = max(upper_bound, gap_end)
        if gap_start >= group_end:
            lower_bound = min(lower_bound, gap_start)
            break

    line_boundaries.append((upper_bound, lower_bound))

# Step 9: Crop and save each line
for i, (start_y, end_y) in enumerate(line_boundaries):
    if end_y - start_y < 5:  # Skip very small lines
        continue

    # Crop the line from the original image
    line_image = image[start_y:end_y, :]

    # Save the cropped line
    output_path = os.path.join(output_dir, f"line_{i+1}.png")
    cv2.imwrite(output_path, line_image)

    # Draw a line on the original image to mark the boundary
    cv2.line(image, (0, start_y), (image.shape[1], start_y), (0, 255, 0), 1)
    cv2.line(image, (0, end_y), (image.shape[1], end_y), (0, 0, 255), 1)

# Step 10: Save the annotated image
cv2.imwrite("annotated_image.png", image)

print(f"Separated {len(line_boundaries)} lines. Check the '{output_dir}' directory for the cropped images.")
