from abc import ABC, abstractmethod
import ctypes
from threading import Thread
from timeit import default_timer as timer

import cv2
import numpy as np
import pyrealsense2 as rs

from yolov5_trt_RS import YoLov5TRT


class CVThread(ABC):

    def stop(self):
        self.stopped = True

    def _setup(self, tname, args):
        self.stopped = False
        self._next()

        thread = Thread(target=self._update, name=tname, args=args)
        thread.setDaemon(True)
        thread.start()

    def _update(self):
        while True:
            if self.stopped:
                self._cleanup()
                return
            self._next()

    @abstractmethod
    def _next(self):
        """Sets nexts frames"""

    @abstractmethod
    def _cleanup(self):
        """Cleans up on termination"""

    @abstractmethod
    def read(self):
        """Returns next frames"""


class RSCapture(CVThread):

    def __init__(self, uid, *args):
        self.uid = uid
        self.stream = rs.pipeline()

        rs_cfg = rs.config()
        rs_cfg.enable_stream(rs.stream.color)
        rs_cfg.enable_stream(rs.stream.depth)

        self.align = rs.align(rs.stream.color)
        self.stream.start(rs_cfg)

        self._setup(f"RS-{uid}", args)

    def _next(self):
        frames = self.stream.wait_for_frames()
        aligned = self.align.process(frames)

        self.frame = np.asanyarray(aligned.get_color_frame().get_data())
        self.depth = aligned.get_depth_frame()
        self.intrin = self.depth.profile.as_video_stream_profile().intrinsics

    def _cleanup(self):
        self.stream.stop()

    def read(self):
        return self.frame, self.depth, self.intrin


class YOLOThread(CVThread):

    def __init__(self, yolo, cap, *args):
        self.yolo = yolo
        self.cap = cap
        self._setup(f"YOLO-{cap.uid}", args)

    def _next(self):
        frame, _, _, = self.cap.read()
        self.frame = self.yolo.infer(frame[..., ::-1])[0]

    def _cleanup(self):
        self.cap._cleanup()
        self.yolo.destroy()

    def read(self):
        return self.frame


if __name__ == "__main__":
    ctypes.CDLL("build/libmyplugins.so")

    yolo = YoLov5TRT("build/yolov5s.engine")
    cap = RSCapture(0)
    yolo_thread = YOLOThread(yolo, cap)

    fpses = []
    while True:
        t = timer()
        frame = yolo_thread.read()

        fps = 1 / (timer() - t)
        fpses.append(fps)
        # print("FPS:", fps)

        cv2.imshow("test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.stop()
            break

    print(f"Average fps: {sum(fpses) / len(fpses)}")
