import cv2 as cv
import mediapipe as mp

mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def resizeThenShow(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    
    # resize image
    resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)

    # time.sleep(0.01)
    cv.imshow('Head Pose Estimation', resized)

def draw_landmarks(image, landmark_list, tesselation=False, contours=False, irises=False):
    if tesselation:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=landmark_list,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )

    if contours:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=landmark_list,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
        )

    if irises:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=landmark_list,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
        )