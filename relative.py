import sys
import numpy as np
import cv2
import cv2.aruco as aruco
import numpy as np
from matrices import translation, rotation, scale

# @profile
def main():
    # Initialize the webcam
    source = sys.argv[1]
    try:
        source = int(source)
    except ValueError:
        pass

    cap = cv2.VideoCapture(source)

    # Define the type of ArUco markers
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    # parameters = aruco.DetectorParameters_create()

    # Camera calibration parameters (replace with your camera's parameters)
    camera_matrix = np.array(
        [[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32
    )
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    trans_from_X_to_box = dict()
    w = 0.07 # box width in m
    marker_w = 0.053  # marker width in m
    trans_from_X_to_box[1] = (
        rotation("x", -90) @ translation("y", w/2) @ translation("z", -6) 
    )
    trans_from_X_to_box[2] = (
        rotation("x", -90)
        @ translation("y", w/2)
        @ rotation("z", 90)
        @ translation("z", -6)
    )
    trans_from_X_to_box[3] = (
        rotation("x", -90)
        @ translation("y", w/2)
        @ rotation("z", 180)
        @ translation("z", -6)
    )
    trans_from_X_to_box[4] = (
        rotation("x", -90)
        @ translation("y", w/2)
        @ rotation("z", 270)
        @ translation("z", -6)
    )

    anchor = 0

    while True:
        ret, frame = cap.read()
        if ret:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers
            # corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict)
            corners, ids, rejected_img_points = aruco.detectMarkers(frame, aruco_dict)
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

            if anchor in ids:
                M0 = get_affine_transform(0)

                for id in ids:
                    if id == anchor:
                        continue

                    M = get_affine_transform(id)
                    Z = np.array(
                        [
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ]
                    ).T

                    target = np.linalg.inv(M0) @ M @ Z

                    x, y, z = target[:3, 3].flatten()

                    # Print the relative position
                    print(
                        f"Relative Position of Marker {id} to Marker 0: {x:1.2f} {y:1.2f} {z:1.2f}"
                    )

            # Display the frame
            cv2.imshow("Frame", frame)

            # Break the loop
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    # Release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
