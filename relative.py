import cv2
import cv2.aruco as aruco
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define the type of ArUco markers
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
# parameters = aruco.DetectorParameters_create()

# Camera calibration parameters (replace with your camera's parameters)
camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

while True:
    ret, frame = cap.read()
    if ret or True:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        # corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict)
        corners, ids, rejected_img_points = aruco.detectMarkers(frame, aruco_dict)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners, 0.05, camera_matrix, dist_coeffs
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

        # Check if both markers are detected
        if ids is not None and len(ids) >= 2 and 0 in ids and 1 in ids:
            # Estimate pose of each marker

            M0 = get_affine_transform(0)
            M1 = get_affine_transform(1)
            Z = np.array([[0, 0, 0, 1]]).T

            tvec_rel = np.linalg.inv(M0) @ M1 @ Z

            x, y, z, _ = tvec_rel.flatten()

            # Print the relative position
            print(
                f"Relative Position of Marker 1 to Marker 0: {x:1.2f} {y:1.2f} {z:1.2f}"
            )

        # Display the frame
        cv2.imshow("Frame", frame)

        # Break the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Release the capture
cap.release()
cv2.destroyAllWindows()
