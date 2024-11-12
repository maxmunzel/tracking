from collections import defaultdict
import cv2
import numpy as np
import typer
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from load_calib import load_calib
from typing import List, Tuple


def calculate_marker_distance(rvec1, tvec1, rvec2, tvec2):
    # Calculate Euclidean distance between two translation vectors
    distance = np.linalg.norm(tvec1 - tvec2)
    return distance


def generate_video(input_video: str, output_video: str, calibration: str):
    marker_pairs = [
        ("h", 3, 4),
        ("h", 5, 6),
        ("h", 10, 11),
        ("h", 12, 13),
        ("v", 0, 7),
        ("v", 1, 8),
        ("v", 2, 9),
    ]
    marker_pairs: List[Tuple[str, int, int]]
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError("Could not open input video file.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width * 2, frame_height),
    )

    # Load calibration data
    camera_matrix, dist_coeffs = load_calib(calibration)

    # Initialize the ArUco dictionary and parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()

    distances = defaultdict(list)
    times = defaultdict(list)
    frame_count = 0
    marker_length = 0.05625  # Set the actual marker side length in meters

    t = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        t += 1 / cap.get(cv2.CAP_PROP_FPS)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=aruco_params
        )

        if ids is not None:
            ids = ids.flatten()
            for axis, marker_id1, marker_id2 in marker_pairs:
                key = (axis, marker_id1, marker_id2)
                # Only proceed if both markers are detected
                if marker_id1 in ids and marker_id2 in ids:
                    idx1 = np.where(ids == marker_id1)[0][0]
                    idx2 = np.where(ids == marker_id2)[0][0]

                    # Estimate pose for each marker
                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        corners, marker_length, camera_matrix, dist_coeffs
                    )

                    # Calculate the distance between the two markers' translation vectors
                    distance = calculate_marker_distance(
                        rvecs[idx1], tvecs[idx1], rvecs[idx2], tvecs[idx2]
                    )
                    distances[key].append(distance)
                    times[key].append(t)

        # Create plot image
        fig = Figure(figsize=(4, 3))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        delta = 0.2
        # ax.set_ylim(.15 -delta, .15 + delta)
        for key in marker_pairs:
            axis = key[0]
            color = "tab:orange" if axis == "v" else "tab:blue"
            ax.plot(times[key], distances[key], color=color)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Distance (meters)")
        ax.set_title("Distance between ArUco Markers")

        canvas.draw()
        buf = canvas.buffer_rgba()
        plot_img = np.asarray(buf)

        # Convert plot to BGR format and resize to match video dimensions
        plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGBA2BGR)
        plot_img = cv2.resize(plot_img, (frame_width, frame_height))

        # Stack original frame and plot image horizontally
        combined_frame = np.hstack((frame, plot_img))
        out.write(combined_frame)

        frame_count += 1

    cap.release()
    out.release()


if __name__ == "__main__":
    typer.run(generate_video)
