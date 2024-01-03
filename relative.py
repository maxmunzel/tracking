import sys
import numpy as np
import cv2
import cv2.aruco as aruco
import numpy as np
from matrices import translation, rotation, scale, vec2m, m2vec
from typing import Tuple
import redis
import json

trans_from_X_to_box = dict()
w = 0.1278  # box width in m
marker_w = 0.08  # marker width in m
marker_h = 0.02 + marker_w / 2  # marker hight over the bottom of the box
trans_from_X_to_box[1] = (
    rotation("x", -90) @ translation("y", w / 2) @ translation("z", -marker_h)
)
trans_from_X_to_box[2] = (
    rotation("x", -90)
    @ translation("y", w / 2)
    @ rotation("z", 90)
    @ translation("z", -marker_h)
)
trans_from_X_to_box[3] = (
    rotation("x", -90)
    @ translation("y", w / 2)
    @ rotation("z", 180)
    @ translation("z", -marker_h)
)
trans_from_X_to_box[4] = (
    rotation("x", -90)
    @ translation("y", w / 2)
    @ rotation("z", 270)
    @ translation("z", -marker_h)
)


def main():
    r = redis.Redis(decode_responses=True)
    anchor = "0"
    while True:
        res = r.xrevrange("aruco", "+", "-", count=1)
        if res:
            payload = res[0][1]
            # [('1704280318147-0', {'fps': '78.59653330834817', 'transforms': '{}'})]
            transforms = json.loads(payload["transforms"])
            if anchor in transforms.keys():
                M0 = np.array(transforms[anchor]).reshape(4, 4)
                M0_INV = np.linalg.inv(M0)
                for id, transform in transforms.items():
                    if id == anchor:
                        continue
                    M = np.array(transform).reshape(4, 4)
                    try:
                        target = M0_INV @ M @ trans_from_X_to_box[int(id)]
                    except KeyError:
                        # we have detected a spurious marker, skip it
                        continue

                    x, y, z = m2vec(target)

                    # Print the relative position
                    print(
                        f"Relative Position of Marker {id} to Marker {anchor}: {x:1.2f} {y:1.2f} {z:1.2f}"
                    )
                    yield id, target.copy()

    # Release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    for _ in main():
        pass