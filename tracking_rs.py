import sys
import numpy as np
import cv2
import cv2.aruco as aruco
import redis
import json
import pyrealsense2 as rs
from typing import Tuple
import time
import typer
from relative import marker_w

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)


def load_realsense_calibration(profile, stream_type=rs.stream.color):
    """Load camera matrix and distortion coefficients for the RealSense D435i."""
    intrinsics = (
        profile.get_stream(stream_type).as_video_stream_profile().get_intrinsics()
    )
    camera_matrix = np.array(
        [
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1],
        ]
    )
    dist_coeffs = np.array(intrinsics.coeffs)
    return camera_matrix, dist_coeffs


def main(
    redis_ip: str = "10.10.20.142",
    preview: bool = True,
    slow: bool = False,
):
    r = redis.Redis(redis_ip, decode_responses=True)

    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color, 1920, 1080, rs.format.bgr8, 30
    )  # RGB camera at max resolution

    profile = pipeline.start(config)

    # Load calibration data
    camera_matrix, dist_coeffs = load_realsense_calibration(profile)

    try:
        # gamma correction
        table = np.array(
            [((i / 255.0) ** 0.4) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")

        while True:
            start = time.time()

            # Wait for a frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()  # RGB camera frame
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())

            # Apply gamma correction
            frame = cv2.LUT(frame, table)

            # Detect ArUco markers
            corners, ids, _ = aruco.detectMarkers(frame, aruco_dict)
            if ids is None:
                ids = np.array([])
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, marker_w, camera_matrix, dist_coeffs
            )
            aruco.drawDetectedMarkers(frame, corners, ids)

            def get_affine_transform(aruco_id: int) -> np.ndarray:
                """Returns the 4x4 matrix of the transformation from the camera frame to the aruco frame."""
                result = np.zeros((4, 4))
                result[3, 3] = 1
                index = np.where(ids == aruco_id)[0][0]

                R, _ = cv2.Rodrigues(rvecs[index])
                result[:3, :3] = R
                result[:3, 3] = tvecs[index].flatten()
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
                    "camera_id": 1,
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
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    typer.run(main)
