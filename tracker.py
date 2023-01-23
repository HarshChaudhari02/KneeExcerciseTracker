#Importing Libraries
import cv2
import mediapipe as mp
import numpy as np
import time

#Setting some features
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

#Initializing the counter
TIMER = int(0)

#Calculate Angles
def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360-angle

    return angle

#Getting the video
cap = cv2.VideoCapture('KneeBendVideo.mp4')
cap.set(3,800)
cap.set(4,800)

mes = ""
#Rep Counter
counter = 0

#Stage
stage = "up"

#Fluctuation Checker
checker = 0

#set the font
font = cv2.FONT_HERSHEY_SIMPLEX

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    prev = time.time()
    while cv2.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv2.waitKey()
            break

        #Detect stuff and render
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            #Get Landmarks

            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            #Calculate Angle
            angle = calculate_angle(hip, knee, ankle)

            #Rep counter & Timer logic
            if angle > 140:
                stage = "down"
                check = "yes"
                if TIMER > 0:
                    checker = 1
                    cv2.putText(image, str("Keep your knee bent"), (200, 250), font, 1, (0, 255, 255), 4, cv2.LINE_AA)

            if angle <= 140:
                #Check for Fluctuations
                if checker == 1:
                    counter -= 1
                    checker = 0

                if check == "yes":
                    stage = "up"
                    counter += 1
                    check = "no"

                #set the timer
                if flag == 1:
                    TIMER = int(8)

                cur = time.time()

                #Reset the timer
                if cur - prev >= 1:
                    prev = cur
                    TIMER = TIMER - 1

        except:
            pass

        #set the flag
        if stage == "down":
            flag = 1
        else:
            flag = 0

        #Setup status box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

        #Rep data
        cv2.putText(image, 'REPS', (15,12), font, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        #Stage
        cv2.putText(image, 'STAGE', (100, 12), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (90, 60), font, 1.5, (255, 255, 255), 2, cv2.LINE_AA)

        #Render Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))


        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()










