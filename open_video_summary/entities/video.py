from __future__ import annotations

from typing import Optional
from datetime import timedelta
from dataclasses import dataclass, field


@dataclass
class VideoSegment:
    content: str
    start: float
    end: float
    order: Optional[int] = field(default=None)
    video_topic: str = field(default="")
    global_topic: str = field(default="")
    video_path: str = field(default="")

    def __str__(self) -> str:
        return "%s --> %s\n%s" % (
            self.formatted_start,
            self.formatted_end,
            self.content,
        )

    @property
    def formatted_start(self) -> str:
        return str(timedelta(seconds=self.start))

    @property
    def formatted_end(self) -> str:
        return str(timedelta(seconds=self.end))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VideoSegment):
            return False
        return (
            self.content == other.content
            and self.start == other.start
            and self.end == other.end
            and self.order == other.order
            and self.video_topic == other.video_topic
            and self.global_topic == other.global_topic
            and self.video_path == other.video_path
        )

    def __hash__(self):
        return hash(
            (
                self.content,
                self.start,
                self.end,
                self.order,
                self.video_topic,
                self.global_topic,
                self.video_path,
            )
        )


@dataclass
class Video:
    name: str
    path: str
    topics: list[str] = field(default_factory=list)
    segments: list[VideoSegment] = field(default_factory=list)

    def __post_init__(self) -> None:
        for segment in self.segments:
            if not segment.video_path:
                segment.video_path = self.path
