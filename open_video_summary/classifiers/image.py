from abc import ABC, abstractmethod

from cv2 import CascadeClassifier


class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, frame) -> bool:
        ...


class CascadeFaceDetector(ObjectDetector):
    def __init__(self, classifier_path: str) -> None:
        self.classifier = CascadeClassifier(classifier_path)

    def detect(self, frame) -> bool:
        faces = self.classifier.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
        return bool(len(faces))
