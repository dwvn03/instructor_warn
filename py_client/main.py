from asyncio.windows_events import NULL
import numpy as np
import cv2 as cv
import mediapipe as mp
import time

from adapter import warningPing

mp_drawing_styles = mp.solutions.drawing_styles

mp_drawing = mp.solutions.drawing_utils
drawing_Spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv.VideoCapture(1, cv.CAP_DSHOW)

while cap.isOpened():
    have_image, image = cap.read()

    if not have_image:
        break

    start = time.time()
    # warningPing()
    # input()

    # Flip the image horizontally
    # Also convert the color space from BGR to RG8
    image = cv.cvtColor(cv.flip(image, 100), cv.COLOR_BGR2RGB)
    
    # To improve performance
    image.flags.writeable = False

    # Get the resullt
    results = face_mesh.process(image)

    # To improve performance
    image.flags.writeable = True

    # Also convert the color back
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

    img_h, img_w, imgC = image.shape
    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in [1, 33, 61, 199, 263, 291]:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm. x * img_w, lm.y * img_h, lm.z * 3000)
                    
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    #Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w

            # The camera matrix
            focal_length = 1 * img_w
            cam_matrix = np.array([ [focal_length, 8, img_h / 2],
                                    [0, focal_length, img_w / 2],
                                    [0, 0, 1]])
            #The distortion parameters
            dist_matrix = np.zeros ( (4, 1), dtype=np.float64)
            # Solve PnP
            success, rot_vec, trans_vec = cv.solvePnP (face_3d, face_2d, cam_matrix, dist_matrix)
            # Get rotational matrix
            rmat, jac = cv.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Ox, Qy, Qz = cv.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            if y < -10:
                text = "looking left"
            elif y > 10:
                text = "looking right"
            elif x < -10:
                text = "looking down"
            elif x > 10:
                text = "looking up"
            else:
                text = "forward"

            #Display the nose direction
            nose_3d_projection, jacobian = cv.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
            p1 = (int(nose_2d[0]), int (nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv.line(image, p1, p2, (255, 0, 0), 3)
            #Add the text on the image
            cv.putText (image, text, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            cv.putText(image, "x: " + str(np.round(x, 2)), (450, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv.putText(image, "y: " + str(np.round(y, 2)), (450, 100), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv.putText(image, "z: " + str(np.round(z, 2)), (450, 150), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

            # mp_drawing.draw_landmarks(
            #     image=image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_CONTOURS,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            # )

            # mp_drawing.draw_landmarks(
            #     image=image,
            #     landmark_list=face_landmarks,
            #     connections=mp_face_mesh.FACEMESH_IRISES,
            #     landmark_drawing_spec=None,
            #     connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            # )
    
    end = time.time ()
    totalTime = end - start

    fps = 1 / totalTime
    print("FPS: ", fps)

    cv.putText(image, f'FPS: {int (fps) } ', (20,450), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)


    # percent of original size
    scale_percent = 150 
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # resize image
    resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)

    # time.sleep(0.01)
    cv.imshow('Head Pose Estimation', resized)

    if 'text' in locals():
        if text == "forward":
            warningPing()
            time.sleep(5)


    if cv.waitKey(20) & 0xFF == ord('s'):
        break

cap.release()
cv.destroyAllWindows()