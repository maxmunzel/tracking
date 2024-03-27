import re
import numpy as np
from typing import Tuple


def load_calib(xml_file: str) -> Tuple[np.ndarray, np.ndarray]:
    with open("oakd_bot3.xml") as f:
        xml = f.read()

    camera_matrix_str = (
        re.search(
            r"cameraMatrix.*?<data>(.*?)</data>.*?dist_coeffs ",
            xml,
            re.MULTILINE | re.DOTALL,
        )
        .group(1)
        .strip()
    )
    camera_matrix = np.array(list(map(float, camera_matrix_str.split()))).reshape(3, 3)
    dist_coeffs_str = (
        re.search(
            r"<dist_coeffs .*?<data>(.*?)</data>.*?</dist_coeffs>",
            xml,
            re.DOTALL | re.MULTILINE,
        )
        .group(1)
        .strip()
    )
    dist_coeffs = np.array(list(map(float, dist_coeffs_str.split()))).reshape(5, 1)
    return camera_matrix, dist_coeffs
