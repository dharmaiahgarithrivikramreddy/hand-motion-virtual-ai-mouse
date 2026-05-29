import cv2
import mediapipe as mp
import pyautogui
import math
import time

pyautogui.FAILSAFE = False

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

def dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        thumb = lm[4]
        index = lm[8]
        middle = lm[12]
        ring = lm[16]

        # MOVE MOUSE (Only index)
        x = int(index.x * screen_w)
        y = int(index.y * screen_h)
        pyautogui.moveTo(x, y)

        # LEFT CLICK (Index + Thumb)
        if dist(index, thumb) < 0.03:
            pyautogui.click()
            time.sleep(0.3)

        # RIGHT CLICK (Index + Middle)
        elif dist(index, middle) < 0.03:
            pyautogui.rightClick()
            time.sleep(0.3)

        # DOUBLE CLICK (Index + Middle + Thumb)
        elif dist(index, thumb) < 0.03 and dist(index, middle) < 0.03:
            pyautogui.doubleClick()
            time.sleep(0.4)

        # SCROLL (Index + Middle up/down)
        if index.y < lm[6].y and middle.y < lm[10].y:
            pyautogui.scroll(40)
        elif index.y > lm[6].y and middle.y > lm[10].y:
            pyautogui.scroll(-40)

        # MINIMIZE WINDOW (FIST)
        fingers_closed = all(lm[i].y > lm[i - 2].y for i in [8, 12, 16, 20])
        if fingers_closed:
            pyautogui.hotkey('win', 'down')
            time.sleep(0.6)

        # CLOSE WINDOW (Thumb + Ring)
        if dist(thumb, ring) < 0.03:
            pyautogui.hotkey('alt', 'f4')
            time.sleep(0.6)

        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("AI Virtual Mouse", img)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
