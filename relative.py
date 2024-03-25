import numpy as np
import numpy as np
from scipy.spatial.transform import Rotation
from matrices import translation, rotation, m2vec
import redis
import json

PROJECT_TO_TABLE = True


def project_transform_onto_table(transform):  # chatgpt
    """
    Projects a quaternion to the space of rotations around the Z-axis (yaw only).

    Parameters:
    quaternion (array-like): The input quaternion in the form [x, y, z, w].

    Returns:
    numpy.ndarray: The output quaternion representing a rotation around the Z-axis.
    """
    res = transform.copy()
    # Convert the input rotation to Euler angles with 'ZYX' convention
    euler_angles = Rotation.from_matrix(transform[:3, :3]).as_euler("ZYX")

    # Zero out the roll and pitch components (X and Y axes rotations)
    euler_angles[0] = 0  # Roll
    euler_angles[1] = 0  # Pitch

    # Convert the modified Euler angles back to the result
    res[:3, :3] = Rotation.from_euler("ZYX", euler_angles).as_matrix()

    # Set z height correctly
    res[2, 3] = 0
    return res


trans_from_X_to_box = dict()
w = 0.1278  # box width in m
marker_w = 0.08  # marker width in m
marker_h = 0.02 + marker_w / 2  # marker hight over the bottom of the box
trans_from_X_to_box[1] = (
    rotation("x", -90)
    @ translation("y", w / 2)
    @ translation("z", -marker_h)
    @ rotation("z", -90)
)
trans_from_X_to_box[2] = (
    rotation("x", -90)
    @ translation("y", w / 2)
    @ rotation("z", 90)
    @ translation("z", -marker_h)
    @ rotation("z", -90)
)
trans_from_X_to_box[3] = (
    rotation("x", -90)
    @ translation("y", w / 2)
    @ rotation("z", 180)
    @ translation("z", -marker_h)
    @ rotation("z", -90)
)
trans_from_X_to_box[4] = (
    rotation("x", -90)
    @ translation("y", w / 2)
    @ rotation("z", 270)
    @ translation("z", -marker_h)
    @ rotation("z", -90)
)

unit_x = np.array([1, 0, 0, 1]).T
unit_y = np.array([0, 1, 0, 1]).T


def score(M: np.ndarray) -> float:
    # estimates the area of the aruco marker by projecting
    # the x/y unit vectors from the marker frame to the camera
    # and calculating the area of the parallelogram they span.
    assert M.shape == (4, 4)
    a = (M @ unit_x)[:2]
    b = (M @ unit_y)[:2]
    return np.linalg.norm(np.cross(a, b))


def main():
    r = redis.Redis(decode_responses=True)
    anchor = "0"
    M0 = dict()
    M0_INV = dict()
    while True:
        res = r.xread({"aruco": "$"}, block=1000, count=1)

        if res:
            _, payload = res[0][1][-1]
            # [('1704280318147-0', {'fps': '78.59653330834817', 'transforms': '{}'})]
            transforms = json.loads(payload["transforms"])
            cam_id = payload["camera_id"]
            if anchor in transforms.keys():
                M = np.array(transforms[anchor]).reshape(4, 4)
                if cam_id not in M0:
                    M0[cam_id] = M
                    M0_INV[cam_id] = np.linalg.inv(M)
                else:
                    alpha = 0.98
                    M0[cam_id] *= alpha
                    M0[cam_id] += (1 - alpha) * M
                    M0_INV[cam_id] = np.linalg.inv(M0[cam_id])

            if cam_id in M0:
                assert M0[cam_id] is not None
                assert M0_INV[cam_id] is not None
                results = []
                for id, transform in transforms.items():
                    if id == anchor:
                        continue
                    M = np.array(transform).reshape(4, 4)
                    try:
                        target = M0_INV[cam_id] @ M @ trans_from_X_to_box[int(id)]
                    except KeyError:
                        # we have detected a spurious marker, skip it
                        continue

                    # Add x offset, as the aruco marker is at (offset, 0, 0) in the world coordinate frame.
                    # Conveniently, the robot and world coordinate frames are aligned, so we just need to adjust
                    # the x component of the 4x4 matrix.
                    x = 0.12 + target[0, 3]

                    x0 = 0.15
                    err0 = 0.01
                    x1 = 0.650
                    err1 = 0.04
                    slope = (err1 - err0) / (x0 - x1)
                    err0 = -x1 * slope - err1
                    err = err0 + slope * x

                    target[0, 3] = x - err

                    # x, y, z = m2vec(target)
                    # Print the relative position
                    # print(
                    #    f"Relative Position of Marker {id} to Marker {anchor}: {x:1.2f} {y:1.2f} {z:1.2f}"
                    # )
                    results.append((score(M), id, target.copy()))
                if results:
                    _, id, target = max(results)
                    if PROJECT_TO_TABLE:
                        yield id, project_transform_onto_table(target)
                    else:
                        yield id, target


if __name__ == "__main__":
    for id, target in main():
        x, y, z = target[:3, 3]
        dist = score(target)
        print(f"marker {id} at xyz {x:.2f} {y:.2f} {z:.2f}")
