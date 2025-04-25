from math import log10
from pandas import DataFrame
from collections import Counter
from sklearn.cluster import KMeans  # type: ignore
from numpy import concatenate
from cv2 import (
    calcHist,
    normalize,
    compareHist,
    SIFT_create,
    NORM_L1,
    HISTCMP_INTERSECT,
)

from open_video_summary.utils import log
from open_video_summary.entities.image import Keyframe
from open_video_summary.entities.video import VideoSegment
from open_video_summary.handlers.image import KeyframeHandler


class BagOfVisualWords:
    def __init__(self, items: dict[VideoSegment, list], dict_size: int) -> None:
        self.__items = items
        self.__dict_size = dict_size
        self.__kmeans = None
        self.__bovw_df = None

    def fit_kmeans(self, **kwargs) -> None:
        self.__kmeans = KMeans(n_clusters=self.__dict_size, **kwargs)
        if self.__kmeans is None:
            raise ValueError("KMeans instance is not initialized.")

        log.info("Fitting KMeans...")
        self.__kmeans.fit(concatenate(list(self.__items.values())))

    def generate_bovw_dataframe(self) -> DataFrame:
        if self.__kmeans is None:
            raise ValueError("KMeans has not been fitted.")

        self.__bovw_df = DataFrame(
            self.__items.items(), columns=["segment", "features"]
        )

        tfs = self.__bovw_df.features.apply(self.__kmeans.predict).apply(Counter)
        doc_freq = Counter(key for a in tfs for key in a.keys())

        for key in doc_freq.keys():
            term_freq = tfs.apply(lambda x: x.get(key))
            term_idf = log10(self.__dict_size / doc_freq.get(key))

            new_df = self.__bovw_df.copy()
            new_df[key] = term_freq * term_idf
            self.__bovw_df = new_df

        self.__bovw_df.drop(columns=["features"], inplace=True)
        self.__bovw_df.set_index("segment", drop=True, inplace=True)

        return self.__bovw_df


class ImageProcessor:
    @staticmethod
    def ks_sift(frames: list):
        segment_keyframes: list[Keyframe] = []
        for frame in frames[1:-1]:
            _, descriptor = SIFT_create().detectAndCompute(frame, None)

            if descriptor is None:
                continue

            keyframe = Keyframe(descriptor=descriptor)
            if KeyframeHandler.is_keyframe(keyframe, segment_keyframes):
                segment_keyframes.append(keyframe)

        return concatenate([kf.descriptor for kf in segment_keyframes])

    @staticmethod
    def get_frame_histogram(frame):
        histogram = calcHist([frame], [0], None, [256], [0, 256])
        normalize(histogram, histogram, norm_type=NORM_L1)
        return histogram

    @staticmethod
    def compare_histograms(hist_1, hist_2):
        return compareHist(hist_1, hist_2, HISTCMP_INTERSECT)
