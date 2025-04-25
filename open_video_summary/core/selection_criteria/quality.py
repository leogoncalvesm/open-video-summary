from typing import Callable
from pandas import DataFrame

from open_video_summary.utils import log
from open_video_summary.entities.video import VideoSegment
from open_video_summary.utils.processing.video import VideoProcessor
from open_video_summary.handlers.summary import SummarySegmentHandler
from open_video_summary.core.selection_criteria.base import SelectionCriteria
from open_video_summary.utils.processing.image import BagOfVisualWords, ImageProcessor


class QualityPick(SelectionCriteria):
    def __init__(
        self,
        source_criteria: str,
        top_n_segments: int = 1,
        bovw_dict_size: int = 300,
        features_extractor: Callable = ImageProcessor.ks_sift,
    ) -> None:
        super().__init__(read_from="pick", source_criteria=source_criteria)
        self.top_n_segments = top_n_segments
        self.bovw_dict_size = bovw_dict_size
        self.features_extractor = features_extractor

    def evaluate(self, handler: SummarySegmentHandler) -> SummarySegmentHandler:
        clusters = [
            cluster
            for cluster in self.get_criteria_input(handler)
            if isinstance(cluster, set)
        ]
        log.info(f"Retrieved {len(clusters)} cluster to execute {self.name} criteria.")

        for cluster in clusters:
            seg_features = self.extract_segments_visual_features(cluster)
            df = self.get_bovw_dataframe(seg_features)

            log.info(f"Retrieving top-{self.top_n_segments} segments from cluster.")

            df["histogram_sum"] = df.sum(axis=1)
            top_segments = df.nlargest(
                self.top_n_segments, columns="histogram_sum"
            ).index.to_list()

            # Discarding whole cluster and including only best-quality segment
            map(lambda s: self.discard(handler, s), cluster)
            for segment in top_segments:
                self.include(handler, segment)

        return handler

    def extract_segments_visual_features(
        self, segments: set[VideoSegment]
    ) -> dict[VideoSegment, list]:
        log.info("Extracting visual features from segments.")
        return {
            segment: self.features_extractor(
                VideoProcessor.retrieve_video_frames(
                    segment.video_path,
                    grayscale=True,
                )
            )
            for segment in segments
        }

    def get_bovw_dataframe(
        self, segments_features: dict[VideoSegment, list]
    ) -> DataFrame:
        log.info(
            f"Generating Bag-of-Visual-Words for {len(segments_features)} segments."
        )
        bovw = BagOfVisualWords(
            items=segments_features,
            dict_size=self.bovw_dict_size,
        )
        log.info("Fitting KMeans algorithm for Bag-of-Visual-Words generated.")
        bovw.fit_kmeans()

        return bovw.generate_bovw_dataframe()
