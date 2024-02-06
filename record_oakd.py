import sys
import numpy as np
import cv2
import cv2.aruco as aruco
import numpy as np
import time
import typer
import depthai as dai


def main(file: str, preview: bool = True):

    # depthai magic
    pipeline = dai.Pipeline()
    colorCamera = pipeline.create(dai.node.ColorCamera)
    colorCamera.setBoardSocket(dai.CameraBoardSocket.RGB)
    colorCamera.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    colorCamera.video.link(xoutRgb.input)
    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        out = cv2.VideoWriter(file, cv2.VideoWriter_fourcc(*"MJPG"), 30, (1920, 1080))

        while True:
            frame = qRgb.get().getCvFrame()
            out.write(frame)

            if preview:
                # Display the frame
                cv2.imshow("Frame", frame)

                # Break the loop
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cv2.destroyAllWindows()
    out.release()


if __name__ == "__main__":
    typer.run(main)
    # main(r)
