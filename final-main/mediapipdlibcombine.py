import cv2
import mediapipe as mp
import numpy as np
import dlib
from imutils import face_utils
radius = 0.0
string_number = ""
# Initializing objects
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)

# Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to compute distance between two points
def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

# Function to determine eye blinking status
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2
    elif ratio > 0.21 and ratio <= 0.25:
        return 1
    else:
        return 0
    radius = float(ratio)
    string_number = str(radius)
    
# Status marking for current state
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

# Detect the face landmarks
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while True:
        success, image = cap.read()

        # Flip the image horizontally and convert the color space from BGR to RGB
        image_rgb = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image_rgb.flags.writeable = False

        # Detect the face landmarks using dlib
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)
            landmarks = face_utils.shape_to_np(landmarks)

            # The numbers are actually the landmarks which will show eye
            left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
            right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

            # Now judge what to do for the eye blinks
            if left_blink == 0 or right_blink == 0:
                sleep += 1
                drowsy = 0
                active = 0
                if sleep > 6:
                    status = "SLEEPING !!!"
                    color = (255, 0, 0)

            elif left_blink == 1 or right_blink == 1:
                sleep = 0
                active = 0
                drowsy += 1
                if drowsy > 6:
                    status = "Drowsy !"
                    color = (0, 0, 255)

            else:
                drowsy = 0
                sleep = 0
                active += 1
                if active > 6:
                    status = "Active :)"
                    color = (0, 255, 0)

        # To improve performance
        image_rgb.flags.writeable = True

        # Convert back to the BGR color space
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw the face mesh annotations on the image.     FACIAL LANDMAK MESH SHOW
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image_bgr,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

        # Draw status text based on eye blinking status
        cv2.putText(image_bgr, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
       # cv2.putText(image_bgr, string_number, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Display the image with Mediapipe FaceMesh annotations
        cv2.imshow('Mediapipe FaceMesh', image_bgr)

        # Preserve aspect ratio while resizing for side-by-side display
        image_resized = cv2.resize(image, (int(image.shape[1] * 0.7), int(image.shape[0] * 0.7)))

        # Draw status text based on eye blinking status for the second frame
        cv2.putText(image_resized, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Display the original image with the aspect ratio preserved
        cv2.imshow('Original Image', image_resized)

        # Terminate the process
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
