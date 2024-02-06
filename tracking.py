import sys
import numpy as np
import cv2
import cv2.aruco as aruco
import numpy as np
from matrices import translation, rotation, scale, vec2m, m2vec
from typing import Tuple
import redis
import json
from relative import marker_w
import time
import typer
import random


# Define the type of ArUco markers
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
# parameters = aruco.DetectorParameters_create()

# Camera calibration parameters (replace with your camera's parameters)
camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
camera_matrix = np.array(
    [
        9.4425079031444568e02,
        0.0,
        6.3411861938672507e02,
        0.0,
        9.4425079031444568e02,
        3.7267119773844064e02,
        0.0,
        0.0,
        1.0,
    ]
).reshape(3, 3)
dist_coeffs = np.array(
    [
        0.0,
        1.0032252215814509e-02,
        6.4898498787460228e-03,
        -1.7873876136562906e-03,
        -1.7555284078852795e-01,
    ]
).reshape(1, 5)
# @profile
def main(source: str, preview: bool = True):
    r = redis.Redis(decode_responses=True)
    cam_id = random.randint(0, 9999999)
    # Initialize the webcam
    try:
        source = int(source)
    except ValueError:
        pass

    cap = cv2.VideoCapture(source)
    while True:
        start = time.time()
        ret, frame = cap.read()
        if ret:
            # Detect ArUco markers
            # corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict)
            corners, ids, _ = aruco.detectMarkers(frame, aruco_dict)
            if ids is None:
                ids = np.array([])
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, marker_w, camera_matrix, dist_coeffs
            )
            aruco.drawDetectedMarkers(frame, corners, ids)

            def get_affine_transform(aruco_id: int) -> np.ndarray:
                """Returns the 4x4 matrix of the transformation from the camera frame to the aruco frame"""
                result = np.zeros((4, 4))
                result[3, 3] = 1
                index = np.where(ids == aruco_id)[0][0]

                R, _ = cv2.Rodrigues(rvecs[index])
                result[:3, :3] = R
                result[:3, 3] = tvecs[index]
                return result

            payload = {
                int(id[0]): list(get_affine_transform(int(id[0])).flatten())
                for id in ids
            }
            fps = 1 / (time.time() - start)

            r.xadd(
                "aruco",
                {
                    "fps": fps,
                    "camera_id": cam_id,
                    "transforms": json.dumps(payload),
                },
            )

            if preview:
                # Display the frame
                cv2.imshow("Frame", frame)

                # Break the loop
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    typer.run(main)
    main(r)
