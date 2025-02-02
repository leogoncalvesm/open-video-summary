from numpy import argmax
from abc import ABC, abstractmethod
from tensorflow import convert_to_tensor
from sentence_transformers import CrossEncoder
from tensorflow.keras.activations import softmax  # type: ignore


class BinaryTextClassifier(ABC):
    @abstractmethod
    def classify(self, content: str) -> bool:
        ...

    @abstractmethod
    def classify_list(self, content_list: list[str]) -> list[bool]:
        ...


class TransformersSubjectivityClassifier(BinaryTextClassifier):
    def __init__(self, model_path: str) -> None:
        self.classifier = CrossEncoder(model_path, num_labels=2)

    def classify(self, content: str) -> bool:
        return bool(
            argmax(
                softmax(
                    convert_to_tensor(self.classifier.predict([[content]]))
                ).numpy(),
                axis=1,
            )[0]
        )

    def classify_list(self, content_list: list[str]) -> list[bool]:
        return list(
            map(
                bool,
                argmax(
                    softmax(
                        convert_to_tensor(
                            self.classifier.predict([[item] for item in content_list])
                        )
                    ).numpy(),
                    axis=1,
                ),
            )
        )
