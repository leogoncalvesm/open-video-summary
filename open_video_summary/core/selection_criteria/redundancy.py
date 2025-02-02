from pandas import DataFrame
from numpy import equal, tril
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

from open_video_summary.utils import log
from open_video_summary.entities.video import Video, VideoSegment
from open_video_summary.handlers.summary import SummarySegmentHandler
from open_video_summary.core.selection_criteria.base import SelectionCriteria
from open_video_summary.utils.helpers import custom_cosine


class ContentBasedRedundancy(SelectionCriteria):
    def __init__(
        self,
        reference_time_sec: int = 785,
        base_threshold: float = 0.17,
    ) -> None:
        super().__init__(read_from="source")
        self.reference_time_sec = reference_time_sec
        self.base_threshold = base_threshold

    def evaluate(self, handler: SummarySegmentHandler) -> SummarySegmentHandler:
        videos = [
            video
            for video in self.get_criteria_input(handler)
            if isinstance(video, Video)
        ]
        log.info(f"Found {len(videos)} videos to execute {self.name} criteria.")

        bow_df = self.get_bow_df(videos)
        correlations = self.get_correlations_df(
            bow_df, threshold=self.calc_min_threshold(videos)
        )
        redundancies = self.get_redundancies(correlations)
        redundancy_clusters = self.cluster_segments(handler, redundancies, videos)

        for cluster in redundancy_clusters:
            log.info(
                f"Including cluster with {len(cluster)} elements to be chosen from."
            )
            self.pick(handler, cluster)

        return handler

    def calc_min_threshold(self, videos: list[Video]) -> float:
        set_time = sum(video.segments[-1].end for video in videos)
        diff = (set_time - self.reference_time_sec) / self.reference_time_sec
        return self.base_threshold + self.base_threshold * diff

    def get_bow_df(self, videos: list[Video]) -> DataFrame:
        log.info("Generating bag-of-words DataFrame for videos found.")
        items = {
            (vid_index, seg_index): segment.content
            for vid_index, video in enumerate(videos)
            for seg_index, segment in enumerate(video.segments)
        }

        index_names = ["video_index", "segment_index"]
        index_df = DataFrame(items.keys(), columns=index_names)

        sentences = list(items.values())

        vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=False)
        tfidf_data = vectorizer.fit_transform(sentences)
        tfidf_df = DataFrame(
            tfidf_data.toarray(), columns=vectorizer.get_feature_names_out()
        )

        return index_df.join(tfidf_df).set_index(index_names)

    def get_correlations_df(self, bow_df: DataFrame, threshold: float) -> DataFrame:
        log.info("Calculating correlations DataFrame from bag-of-words.")
        correlations = bow_df.T.corr(custom_cosine)

        # Disregarding same-video comparisons
        is_same_video = equal.outer(
            correlations.index.get_level_values("video_index"),
            correlations.columns.get_level_values("video_index"),
        )

        # Keeping only the upper diagonal of the pairwise comparisons
        is_upper_diagonal = tril(correlations) > 0

        # Keeping only similarities greater than treshold
        is_gt_threshold = correlations.gt(threshold)

        correlations = (
            correlations.mask(is_same_video | is_upper_diagonal | ~is_gt_threshold)
            .dropna(axis="index", how="all")
            .dropna(axis="columns", how="all")
        )
        correlations = correlations.melt(ignore_index=False).dropna(subset=["value"])
        correlations.columns = ["video_index_col", "segment_index_col", "value"]
        correlations.reset_index(inplace=True)

        return correlations

    def get_redundancies(self, correlations: DataFrame) -> list[list[tuple[int, int]]]:
        log.info("Finding redundant segments from correlations.")
        redundancies = correlations[
            correlations.groupby(["video_index", "video_index_col"])["value"].transform(
                max
            )
            == correlations["value"]
        ]

        # Setting video index and segment index as one tuple object
        redundancies["video"] = tuple(
            zip(redundancies["video_index"], redundancies["segment_index"])
        )
        redundancies["match"] = tuple(
            zip(redundancies["video_index_col"], redundancies["segment_index_col"])
        )
        return redundancies[["video", "match"]].values.tolist()

    def cluster_segments(
        self,
        handler: SummarySegmentHandler,
        redundancies: list[list[tuple[int, int]]],
        videos: list[Video],
    ) -> list[set[VideoSegment]]:
        clusters: list[set[VideoSegment]] = []
        locations: dict[VideoSegment, int] = {}

        for item_a, item_b in redundancies:
            segment_a = videos[item_a[0]].segments[item_a[1]]
            segment_b = videos[item_b[0]].segments[item_b[1]]

            # Disregarding redundancies which one of the elements was either discarded or outputted
            if (
                segment_a in handler.discard
                or segment_a in handler.output
                or segment_b in handler.discard
                or segment_b in handler.output
            ):
                continue

            if segment_a in locations:
                clusters[locations[segment_a]].add(segment_b)
                locations[segment_b] = locations[segment_a]
            elif segment_b in locations:
                clusters[locations[segment_b]].add(segment_a)
                locations[segment_a] = locations[segment_b]
            else:
                clusters.append({segment_a, segment_b})
                locations[segment_a] = len(clusters) - 1

        return clusters
