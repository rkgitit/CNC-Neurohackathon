import cv2
import numpy as np
from PIL import ImageGrab
from pyautogui import press, typewrite, pixel
from time import sleep

# Bounding boxes for cactus and bird (x1, y1, x2, y2)
CACTUS_BBOX = (900, 900, 950, 940)  # y1 ≠ y2 — fix zero height
BIRD_BBOX = (670, 660, 820, 685)

def draw_bounding_boxes():
    while True:
        # Grab screen (full screen or region)
        screenshot = np.array(ImageGrab.grab())

        # Draw rectangles (BGR colors)
        cv2.rectangle(screenshot, (CACTUS_BBOX[0], CACTUS_BBOX[1]), (CACTUS_BBOX[2], CACTUS_BBOX[3]), (0, 255, 0), 2)
        cv2.rectangle(screenshot, (BIRD_BBOX[0], BIRD_BBOX[1]), (BIRD_BBOX[2], BIRD_BBOX[3]), (255, 0, 0), 2)

        # Show the frame
        cv2.imshow("Bounding Boxes Preview - Press 'q' to close", cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR))

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# === Call the preview function ===
draw_bounding_boxes()
