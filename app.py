import cv2
import mediapipe as mp
import numpy as np
import winsound

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    frame_height, frame_width, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            left_eye = face_landmarks.landmark[159]
            right_eye = face_landmarks.landmark[145]

            left_y = int(left_eye.y * frame_height)
            right_y = int(right_eye.y * frame_height)

            difference = abs(left_y - right_y)

            cv2.putText(
                frame,
                f"Eye Diff: {difference}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            if difference < 4:
                cv2.putText(
                    frame,
                    "DROWSINESS DETECTED!",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3
                )

                winsound.Beep(1000, 500)

    cv2.imshow("Drowsiness Detector", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()