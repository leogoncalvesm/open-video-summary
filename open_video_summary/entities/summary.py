from typing import Collection
from dataclasses import dataclass, field

from open_video_summary.entities.video import VideoSegment


@dataclass
class SummaryLog:
    include: list[VideoSegment] = field(default_factory=list)
    discard: list[VideoSegment] = field(default_factory=list)
    output: list[VideoSegment] = field(default_factory=list)
    pick: list[Collection[VideoSegment]] = field(default_factory=list)
