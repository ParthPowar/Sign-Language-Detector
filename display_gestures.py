import cv2, os, random
import numpy as np

def get_image_size():
    # Try to read an image, if it doesn't exist, return default size
    try:
        img = cv2.imread('gestures/0/100.jpg', 0)
        if img is None:
            return (50, 50)  # Default size
        return img.shape
    except:
        return (50, 50)  # Default size if error occurs

# Check if gestures folder exists
if not os.path.exists('gestures/'):
    print("Gestures folder not found. Please create gesture data first.")
    exit()

gestures = os.listdir('gestures/')
if not gestures:
    print("No gesture folders found. Please create gesture data first.")
    exit()

# Sort gestures numerically
gestures.sort(key=int)
begin_index = 0
end_index = min(5, len(gestures))
image_x, image_y = get_image_size()

if len(gestures)%5 != 0:
	rows = int(len(gestures)/5)+1
else:
	rows = int(len(gestures)/5)

full_img = None
for i in range(rows):
	col_img = None
	for j in range(begin_index, end_index):
		if j >= len(gestures):
			break
		
		# Get a random image from gesture folder
		try:
			files = os.listdir(f"gestures/{j}")
			if files: