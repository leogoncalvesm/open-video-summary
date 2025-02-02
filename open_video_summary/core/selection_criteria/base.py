from typing import Optional
from abc import ABC, abstractmethod

from open_video_summary.entities.video import Video, VideoSegment
from open_video_summary.handlers.summary import SummarySegmentHandler


class SelectionCriteria(ABC):
    def __init__(
        self, read_from: str = "source", source_criteria: Optional[str] = None
    ) -> None:
        self.read_from = read_from
        self.source_criteria = source_criteria

    @abstractmethod
    def evaluate(self, handler: SummarySegmentHandler) -> SummarySegmentHandler:
        ...

    def get_criteria_input(
        self, handler: SummarySegmentHandler
    ) -> (
        set[VideoSegment] | list[VideoSegment] | list[list[VideoSegment]] | list[Video]
    ):
        source_obj = (
            handler
            if self.read_from == "source" or self.source_criteria is None
            else handler.agent_logs.get(self.source_criteria)
        )
        return getattr(source_obj, self.read_from)

    def remove_discarded(
        self, handler: SummarySegmentHandler, segments: list[VideoSegment]
    ) -> list[VideoSegment]:
        return list(filter(lambda x: x not in handler.discard, segments))

    def remove_outputted(
        self, handler: SummarySegmentHandler, segments: list[VideoSegment]
    ) -> list[VideoSegment]:
        return list(filter(lambda x: x not in handler.output, segments))

    def include(self, handler: SummarySegmentHandler, segment: VideoSegment) -> None:
        handler.include_segment(segment=segment, agent=self.name)

    def discard(self, handler: SummarySegmentHandler, segment: VideoSegment) -> None:
        handler.discard_segment(segment=segment, agent=self.name)

    def pick(self, handler: SummarySegmentHandler, segments: set[VideoSegment]) -> None:
        handler.add_segments_to_pick(segments=segments, agent=self.name)

    def output(self, handler: SummarySegmentHandler, segment: VideoSegment) -> None:
        handler.add_output_segment(segment=segment, agent=self.name)

    @property
    def name(self) -> str:
        return self.__class__.__name__
