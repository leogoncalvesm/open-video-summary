from typing import Optional
from open_video_summary.utils import log
from open_video_summary.entities.video import VideoSegment
from open_video_summary.handlers.summary import SummarySegmentHandler
from open_video_summary.core.selection_criteria.base import SelectionCriteria


class ClusterBasedChronology(SelectionCriteria):
    def __init__(
        self,
        cluster_criteria: str,
        write_output: bool = True,
    ) -> None:
        super().__init__(read_from="include")
        self.cluster_criteria = cluster_criteria
        self.write_output = write_output

    def evaluate(self, handler: SummarySegmentHandler) -> SummarySegmentHandler:
        segments = [
            segment
            for segment in self.get_criteria_input(handler)
            if isinstance(segment, VideoSegment)
        ]
        log.info(f"Found {len(segments)} segments to execute {self.name} criteria.")

        clusters = [
            set(cluster) for cluster in handler.agent_logs[self.cluster_criteria].pick
        ]
        log.info(f"Found {len(clusters)} clusters to compare segments chronology.")

        result: list[VideoSegment] = []
        for segment, cluster in zip(segments, clusters):
            index = self.find_insert_position(segment, cluster, result)
            result.insert(index, segment)

        for segment in result:
            func = self.output if self.write_output else self.include
            func(handler, segment)

        return handler

    def find_same_video_in_cluster(
        self, item: VideoSegment, cluster: set[VideoSegment]
    ) -> Optional[VideoSegment]:
        for segment in cluster:
            if item.video_path == segment.video_path:
                return segment
        return None

    def find_insert_position(
        self,
        new_item: VideoSegment,
        cluster: set[VideoSegment],
        result_items: list[VideoSegment],
    ):
        insert_position = 0
        for segment in result_items:
            same_video_item = self.find_same_video_in_cluster(segment, cluster)
            is_later = (
                same_video_item.start > segment.start
                if same_video_item is not None
                else new_item.start > segment.start
            )
            if not is_later:
                return insert_position
            insert_position += 1
        return insert_position
