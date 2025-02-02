from functools import reduce

from open_video_summary.entities.video import Video
from open_video_summary.handlers.summary import SummarySegmentHandler
from open_video_summary.core.selection_criteria.base import SelectionCriteria


class Summarizer:
    def __init__(self, selection_criteria: list[SelectionCriteria]) -> None:
        self.selection_criteria = selection_criteria

    def summarize(
        self, videos: list[Video], title: str = "", output_path: str = "output.mp4"
    ) -> Video:
        handler = SummarySegmentHandler()
        handler.set_source_videos(videos)

        handler = reduce(lambda h, c: c.evaluate(h), self.selection_criteria, handler)

        topics = list(set(segment.video_topic for segment in handler.output))
        return Video(
            name=title,
            segments=handler.output,
            topics=topics,
            path=output_path,
        )
