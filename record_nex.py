import cv2
import typer


def main(
    file: str,
    preview: bool = True,
    cam_id: int = 1,
    crop: bool = True,
    mirror: bool = False,
):
    cam = cv2.VideoCapture(cam_id)
    out = cv2.VideoWriter(file, cv2.VideoWriter_fourcc(*"mp4v"), 3, (1620, 1080))
    assert out.isOpened()

    try:
        i = 0
        while True:
            i += 1
            ret, frame = cam.read()
            assert ret
            if crop:
                frame = frame[:, :1620, :]  # crop away menu
            if not i % 10:
                out.write(frame)

            if mirror:
                frame = frame[:, ::-1, :]

            if preview:
                # Display the frame
                cv2.imshow("Frame", frame)

                # Break the loop
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    typer.run(main)
