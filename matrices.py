import numpy as np
def scale(alpha):
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, alpha]
        ])

def rotation(axis, angle_deg):
    """
    Creates a 4x4 homogenous rotation matrix for a given axis and angle.

    :param axis: The axis of rotation ('x', 'y', or 'z').
    :param angle_deg: The angle of rotation in degrees.
    :return: 4x4 numpy rotation matrix.
    """
    angle_rad = np.radians(angle_deg)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0,         0,          0],
            [0, cos_angle, -sin_angle, 0],
            [0, sin_angle, cos_angle,  0],
            [0, 0,         0,          1]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [cos_angle,  0, sin_angle, 0],
            [0,          1, 0,         0],
            [-sin_angle, 0, cos_angle, 0],
            [0,          0, 0,         1]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [cos_angle, -sin_angle, 0, 0],
            [sin_angle, cos_angle,  0, 0],
            [0,         0,         1, 0],
            [0,         0,         0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")

    return rotation_matrix

def translation(axis, distance):
    """
    Creates a 4x4 homogenous translation matrix for a given axis and distance.

    :param axis: The axis of translation ('x', 'y', or 'z').
    :param distance: The distance of translation.
    :return: 4x4 numpy translation matrix.
    """
    if axis == 'x':
        translation_matrix = np.array([
            [1, 0, 0, distance],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'y':
        translation_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, distance],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    elif axis == 'z':
        translation_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, distance],
            [0, 0, 0, 1]
        ])
    else:
        raise ValueError("Invalid axis. Choose 'x', 'y', or 'z'.")

    return translation_matrix


