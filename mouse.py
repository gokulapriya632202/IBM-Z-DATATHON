import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands and drawing utility
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)  # Detect a single hand for simplicity
mp_drawing = mp.solutions.drawing_utils

# Get screen dimensions for mouse movement scaling
screen_width, screen_height = pyautogui.size()

# Start capturing video from the webcam
camera = cv2.VideoCapture(0)

# Variables to track clicking and dragging state
clicking = False
dragging = False
last_click_time = 0  # To track the last click time
click_delay = 0.5  # Minimum delay between clicks to avoid double clicks

try:
    while True:
        # Capture frame from webcam
        ret, image = camera.read()
        if not ret:
            print("Failed to grab frame.")
            break
        image = cv2.flip(image, 1)  # Flip the image horizontally
        image_height, image_width, _ = image.shape

        # Convert the frame to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_image)
        landmarks = result.multi_hand_landmarks

        if landmarks:
            for hand_landmarks in landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get landmark coordinates for fingers
                index_finger_tip = hand_landmarks.landmark[8]  # Index finger tip
                thumb_tip = hand_landmarks.landmark[4]  # Thumb tip
                middle_finger_tip = hand_landmarks.landmark[12]  # Middle finger tip

                # Move mouse cursor
                mouse_x = int(index_finger_tip.x * screen_width)
                mouse_y = int(index_finger_tip.y * screen_height)
                pyautogui.moveTo(mouse_x, mouse_y)

                # Calculate distance between index and middle fingers for scrolling
                index_middle_dist = abs(int(middle_finger_tip.y * image_height) - int(index_finger_tip.y * image_height))

                # Task: Scrolling (Up/Down)
                if index_middle_dist < 30:
                    pyautogui.scroll(-20)  # Scroll down
                elif index_middle_dist > 50:
                    pyautogui.scroll(20)  # Scroll up

                # Task: Click - Detect pinch between thumb and index finger
                thumb_index_dist = abs(int(thumb_tip.y * image_height) - int(index_finger_tip.y * image_height))

                # Check for pinch gesture for clicking and dragging
                if thumb_index_dist < 40:  # Check if thumb and index are close enough
                    current_time = time.time()
                    if not clicking:  # If not already clicking
                        clicking = True
                        # Register a click if the delay since the last click is sufficient
                        if current_time - last_click_time > click_delay:
                            pyautogui.click()  # Perform click
                            last_click_time = current_time  # Update the last click time

                    if not dragging:  # Start dragging if not already dragging
                        dragging = True
                        pyautogui.mouseDown()  # Begin dragging
                else:
                    if dragging:  # Stop dragging when pinch is released
                        dragging = False
                        pyautogui.mouseUp()  # End dragging
                    clicking = False  # Reset clicking state when fingers are apart

        # Display the video feed with annotations
        cv2.imshow("Hand Gesture Scrolling and Clicking Control", image)

        # Exit when ESC key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Release the camera and close OpenCV windows
    camera.release()
    cv2.destroyAllWindows()
