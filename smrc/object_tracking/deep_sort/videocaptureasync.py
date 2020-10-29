import threading
import cv2


# code from http://blog.blitzblit.com/2017/12/24/
# asynchronous-video-capture-in-python-with-opencv/
# Capturing a video stream from a camera in python
# can be done with OpenCV. However, when doing this operation on the main
# thread, performance won’t be great, especially when capturing
# in HD quality. In this blog post, a solution is shown by
# running the video capture operation in a separate (green) thread.
# The performance increases dramatically
# as shown below (on a MacBook Pro) :
# For 640×480:
#
# [i] Frames per second: 28.71, async=False
# [i] Frames per second: 81.67, async=True
#
# For 1280×720
#
# [i] Frames per second: 15.02, async=False
# [i] Frames per second: 52.04, async=True
class VideoCaptureAsync:
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Asynchroneous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()