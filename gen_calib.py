# coding: utf-8
import cv2
import cv2.aruco

ARUCO_DICT = cv2.aruco.DICT_4X4_100  # Dictionary ID
SQUARES_VERTICALLY = 7  # Number of squares vertically
SQUARES_HORIZONTALLY = 5  # Number of squares horizontally
SQUARE_LENGTH = 300  # Square side length (in pixels)
MARKER_LENGTH = 150  # ArUco marker side length (in pixels)
MARGIN_PX = 20  # Margins size (in pixels)

IMG_SIZE = tuple(
    i * SQUARE_LENGTH + 2 * MARGIN_PX
    for i in (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY)
)
OUTPUT_NAME = "ChArUco_Marker.png"


def create_and_save_new_board():
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary,
    )
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    img = cv2.aruco.CharucoBoard.generateImage(board, IMG_SIZE, marginSize=MARGIN_PX)
    cv2.imwrite(OUTPUT_NAME, img)


create_and_save_new_board()
