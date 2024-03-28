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
from load_calib import load_calib

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)


def main(calibration: str, preview: bool = True, slow: bool = False):
    camera_matrix, dist_coeffs = load_calib(calibration)
    r = redis.Redis(decode_responses=True)
    cam_id = random.randint(0, 9999999)
    # Initialize the webcam

    # depthai magic
    pipeline = dai.Pipeline()
    colorCamera = pipeline.create(dai.node.ColorCamera)
    colorCamera.setBoardSocket(dai.CameraBoardSocket.RGB)
    colorCamera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cameraId = colorCamera.getCamId()
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    colorCamera.video.link(xoutRgb.input)
    with dai.Device(pipeline) as device:
        camera_matrix = np.array(
            device.readCalibration().getCameraIntrinsics(
                colorCamera.getBoardSocket(), 1920, 1080
            )
        )
        qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)

        while True:
            start = time.time()
            frame = qRgb.get().getCvFrame()
            # Detect ArUco markers
            # corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict)

            # gamma correction a la https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
            gamma = 0.4
            frame = (frame / 255) ** gamma * 255
            frame = frame.astype(np.uint8)

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
