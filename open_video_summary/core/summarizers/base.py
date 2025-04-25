from functools import reduce

from open_video_summary.entities.video import Video
from open_video_summary.core.selection_criteria.base import SelectionCriteria
from open_video_summary.handlers.summary import (
    SummarySegmentHandler,
    SummarySegmentHandlerIO,
)


class Summarizer:
    def __init__(self, selection_criteria: list[SelectionCriteria]) -> None:
        self.selection_criteria = selection_criteria

    def summarize(
        self,
        videos: list[Video],
        title: str = "",
        video_output_path: str = "output.mp4",
        handler_output_path: str = "output_handler.json",
        save_output: bool = True,
    ) -> Video:
        handler = SummarySegmentHandler()
        handler.set_source_videos(videos)

        handler = reduce(lambda h, c: c.evaluate(h), self.selection_criteria, handler)

        if save_output:
            SummarySegmentHandlerIO.save(handler, handler_output_path)

        topics = list(set(segment.video_topic for segment in handler.output))
        return Video(
            name=title,
            segments=handler.output,
            topics=topics,
            path=video_output_path,
        )
