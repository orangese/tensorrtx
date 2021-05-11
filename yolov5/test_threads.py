from abc import ABC, abstractmethod
from threading import Thread
from timeit import default_timer as timer

import cv2
import numpy as np
import pyrealsense2 as rs


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

        self._setup(f"RS::{uid}", args)

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


class Blur(CVThread):

    def __init__(self, cap, *args):
        self.cap = cap
        self._setup(f"CV::{cap.uid}", args)

    def _next(self):
        frame, _, _, = self.cap.read()
        self.frame = cv2.blur(frame, (15, 15))

    def _cleanup(self):
        self.cap._cleanup()

    def read(self):
        return self.frame


if __name__ == "__main__":
    cap = RSCapture(0)
    blur = Blur(cap)
    fpses = []

    while True:
        t = timer()
        frame = blur.read()

        fps = 1 / (timer() - t)
        fpses.append(fps)
        print("FPS:", fps)

        cv2.imshow("test", frame[..., ::-1])
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cap.stop()
            break

    print(f"Average fps: {sum(fpses) / len(fpses)}")
