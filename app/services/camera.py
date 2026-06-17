import cv2


class Camera:

    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)

    def read(self):
        success, frame = self.cap.read()
        return success, frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
            