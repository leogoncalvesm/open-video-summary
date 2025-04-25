from __future__ import annotations

from re import search
from dataclasses import dataclass, field

from open_video_summary.entities.video import VideoSegment


@dataclass
class SegmentsCluster:
    raw_segments: list[VideoSegment] = field(default_factory=list)
    duration: float = field(default=0, init=False)
    full_content: str = field(default="", init=False)

    def __post_init__(self):
        if self.raw_segments:
            self.update_metadata()

    @property
    def segments(self) -> list[VideoSegment]:
        return self.raw_segments

    @property
    def content(self) -> str:
        if not self.full_content:
            self.load_content()

        return self.full_content

    @property
    def first(self) -> VideoSegment:
        return self.raw_segments[0]

    @property
    def last(self) -> VideoSegment:
        return self.raw_segments[-1]

    @property
    def start(self) -> float:
        return self.first.start

    @property
    def end(self) -> float:
        return self.last.end

    def update_metadata(self, load_content: bool = False) -> None:
        self.update_duration()
        if load_content:
            self.load_content()
        else:
            self.full_content = ""

    def update_duration(self) -> None:
        self.duration = self.end - self.start

    def load_content(self) -> None:
        self.full_content = " ".join([s.content for s in self.segments])

    def append(self, segment: VideoSegment) -> None:
        self.raw_segments.append(segment)
        self.update_metadata()

    def insert_at(self, segment: VideoSegment, position: int = 0) -> None:
        self.raw_segments.insert(position, segment)
        self.update_metadata()

    def ends_with_punctuation(self) -> bool:
        return search("[.!?]$", self.last.content) is not None

    def next_overlaping_cluster(self, overlap_seconds) -> SegmentsCluster:
        new_cluster = SegmentsCluster()
        new_cluster.append(self.last)

        # Ignoring first and last segments. First must not be included, last is already inserted
        for seg in self.segments[-2:0:-1]:
            cur_overlap_diff = abs(new_cluster.duration - overlap_seconds)
            new_overlap_diff = abs(new_cluster.end - seg.start - overlap_seconds)

            # Checking if appending new segment gets the cluster further to the overlap in seconds required
            if new_overlap_diff > cur_overlap_diff:
                return new_cluster

            # Appending new cluster in the beginning
            new_cluster.insert_at(segment=seg, position=0)

        return new_cluster