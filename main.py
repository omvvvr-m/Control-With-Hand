# Programmed by Omar Mohamed

import cv2
import mediapipe as mp
import math as m
import mouse

cap = cv2.VideoCapture(0)


lclicked = False
rclicked = False
sensitivity = 3
smoothTime = 0.25
moveDir = [0,0]
deadzone = 1
prev_x, prev_y = None, None
fx, fy = None, None
smoothed_x, smoothed_y = 0,0
screen_w, screen_h = 1920,1080


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands = 1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def lerp(a, b, t):
    return a + (b - a) * t

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, c = frame.shape
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            for connection in mp_hands.HAND_CONNECTIONS:
                start_idx, end_idx = connection
                x1, y1 = int(hand_landmarks.landmark[start_idx].x * w), int(hand_landmarks.landmark[start_idx].y * h)
                x2, y2 = int(hand_landmarks.landmark[end_idx].x * w), int(hand_landmarks.landmark[end_idx].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            for hand_landmark in hand_landmarks.landmark:
                cx, cy = int(hand_landmark.x * w),int(hand_landmark.y*h)
                cv2.circle(frame,(cx,cy),1,(255,255,255),2)


            control = hand_landmarks.landmark[0]
            index = hand_landmarks.landmark[8]
            thumb = hand_landmarks.landmark[4]
            middle = hand_landmarks.landmark[12]
            ring = hand_landmarks.landmark[16]

            control_x = int(control.x * w)
            control_y = int(control.y * h)

            thumb_x = int(thumb.x * w)
            thumb_y = int(thumb.y * h)

            middle_x = int(middle.x * w)
            middle_y = int(middle.y * h)

            ring_x = int(ring.x * w)
            ring_y = int(ring.y * h)


            index_x = int(index.x * w)
            index_y = int(index.y * h)

            cursor_x = int(control.x* screen_w)
            cursor_y = int(control.y* screen_h)


            cv2.line(frame, (middle_x,middle_y), (thumb_x,thumb_y),(0,255,255),2)
            cv2.line(frame, (index_x,index_y), (thumb_x,thumb_y),(0,255,255),2)
            thumbMiddleDistance = m.dist((thumb_x,thumb_y), (middle_x,middle_y))
            leftDistance = m.dist((thumb_x,thumb_y), (index_x,index_y))

            cv2.putText(frame,str(round(thumbMiddleDistance,2)),(int((thumb_x + control_x)/2) + 20,int((thumb_y + control_y) / 2)),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            

        clickThreshold = 25
        if leftDistance <= clickThreshold:
            if not lclicked:
                mouse.press("left")
                lclicked = True
        else:
            if lclicked:
                mouse.release("left")
                lclicked = False

        if thumbMiddleDistance <= clickThreshold:
            if not rclicked:
                mouse.press("right")
                rclicked = True
        else:
            if rclicked:
                mouse.release("right")
                rclicked = False
             


            smoothed_x, smoothed_y = lerp(smoothed_x,cursor_x,smoothTime),lerp(smoothed_y,cursor_y,smoothTime)
            if prev_x is not None and prev_y is not None:
                dx = (smoothed_x - prev_x) * sensitivity
                dy = (smoothed_y - prev_y) * sensitivity
                if fx is not None and fy is not None:
                    dxx = (smoothed_x - fx)
                    dyy = (smoothed_y - fy)
                    mag = m.sqrt(pow(abs(dxx),2) + pow(abs(dyy),2))
                    if mag > deadzone:
                        moveDir = [dx,dy]
                    mouse.move(int(moveDir[0]), int(moveDir[1]), absolute=False)
                    moveDir = [0,0]

            prev_x, prev_y = smoothed_x, smoothed_y
            fx, fy = smoothed_x, smoothed_y             
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()