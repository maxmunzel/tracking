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
import depthai as dai
import random


# Define the type of ArUco markers
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
# parameters = aruco.DetectorParameters_create()

# Camera calibration parameters (replace with your camera's parameters)
camera_matrix = np.array(
    [
        1.7907479311571933e03,
        0.0,
        9.3152486270890222e02,
        0.0,
        1.7907479311571933e03,
        5.3473987232637899e02,
        0.0,
        0.0,
        1.0,
    ],
    dtype=np.float32,
).reshape(3, 3)

dist_coeffs = np.array(
    [1.7036472746498343e-01, 0.0, 0.0, 0.0, -1.4241571045237869e00]
).reshape(5, 1)
# @profile
def main(preview: bool = True, slow: bool = False):
    r = redis.Redis(decode_responses=True)
    cam_id = random.randint(0, 9999999)
    # Initialize the webcam

    # depthai magic
    pipeline = dai.Pipeline()
    colorCamera = pipeline.create(dai.node.ColorCamera)
    colorCamera.setBoardSocket(dai.CameraBoardSocket.RGB)
    colorCamera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    colorCamera.video.link(xoutRgb.input)
    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)

        while True:
            start = time.time()
            frame = qRgb.get().getCvFrame()
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
            if slow:
                time.sleep(1 / 10)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    typer.run(main)
    # main(r)
