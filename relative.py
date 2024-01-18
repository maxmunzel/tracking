import numpy as np
import numpy as np
from matrices import translation, rotation, m2vec
import redis
import json

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
                M0[cam_id] = np.array(transforms[anchor]).reshape(4, 4)
                M0_INV[cam_id] = np.linalg.inv(M0[cam_id])

            if cam_id in M0:
                assert M0[cam_id] is not None
                assert M0_INV[cam_id] is not None
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
                    x_offset = 0.13
                    target[0, 3] += x_offset

                    # x, y, z = m2vec(target)
                    # Print the relative position
                    # print(
                    #    f"Relative Position of Marker {id} to Marker {anchor}: {x:1.2f} {y:1.2f} {z:1.2f}"
                    # )
                    yield id, target.copy()


if __name__ == "__main__":
    for id, target in main():
        x, y, z = target[:3, 3]
        print(f"marker {id} at xyz {x:.2f} {y:.2f} {z:.2f}")
