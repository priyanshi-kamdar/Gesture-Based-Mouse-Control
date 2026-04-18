import pyautogui
import time
import numpy as np
from collections import deque

pyautogui.PAUSE = 0  # Disable PyAutoGUI delay for much smoother movement
pyautogui.FAILSAFE = False

buffer = deque(maxlen=5)
last_action_time = 0
cooldown = 1  # seconds

# Add a smoothing factor for the cursor
smooth_x, smooth_y = 0, 0
smoothing_factor = 5

def get_stable_prediction(pred):
    buffer.append(pred)
    return max(set(buffer), key=buffer.count)

def perform_action(pred, original_landmarks=None):
    global last_action_time, smooth_x, smooth_y

    # Pointing action: cursor movement
    if pred == 4 and original_landmarks is not None:
        index_tip = original_landmarks[8] # index finger tip
        screen_w, screen_h = pyautogui.size()
        
        # 1. Flip X coordinate (1 - x) so it mirrors your direct hand movement
        # 2. Add an amplification factor (1.5x) so you don't have to reach the far edges of the camera
        target_x = np.clip((1.0 - index_tip[0]) * 1.5 - 0.25, 0, 1) * screen_w
        target_y = np.clip(index_tip[1] * 1.5 - 0.25, 0, 1) * screen_h
        
        # 3. Apply smoothing logic to avoid jittering
        if smooth_x == 0 and smooth_y == 0:
            smooth_x, smooth_y = target_x, target_y
        else:
            smooth_x += (target_x - smooth_x) / smoothing_factor
            smooth_y += (target_y - smooth_y) / smoothing_factor

        try:
            pyautogui.moveTo(int(smooth_x), int(smooth_y))
        except Exception:
            pass
        return

    if time.time() - last_action_time < cooldown:
        return

    if pred == 0:
        pyautogui.screenshot(f"screenshot_{int(time.time())}.png")
    elif pred == 1:
        pyautogui.click(button='right')
    elif pred == 2:
        pyautogui.scroll(200)
    elif pred == 3:
        pyautogui.scroll(-200)

    last_action_time = time.time()