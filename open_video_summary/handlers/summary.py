from json import dump, load
from dacite import from_dict
from dataclasses import dataclass, field, asdict

from open_video_summary.utils import log
from open_video_summary.entities.summary import SummaryLog
from open_video_summary.entities.video import Video, VideoSegment


@dataclass
class SummarySegmentHandler:
    __source_videos: list[Video] = field(default_factory=list)
    __output: list[VideoSegment] = field(default_factory=list)
    __to_discard: set[VideoSegment] = field(default_factory=set)
    __to_include: set[VideoSegment] = field(default_factory=set)
    __to_pick: list[set[VideoSegment]] = field(default_factory=list)
    __agent_log: dict[str, SummaryLog] = field(default_factory=dict)

    @property
    def source(self) -> list[Video]:
        return self.__source_videos

    @property
    def output(self) -> list[VideoSegment]:
        return self.__output

    @property
    def include(self) -> set[VideoSegment]:
        return self.__to_include

    @property
    def discard(self) -> set[VideoSegment]:
        return self.__to_discard

    @property
    def pick(self) -> list[set[VideoSegment]]:
        return self.__to_pick

    @property
    def agent_logs(self) -> dict[str, SummaryLog]:
        return self.__agent_log

    def __log_agent_action(
        self,
        action: str,
        agent: str,
        segment: VideoSegment | list[VideoSegment] | set[VideoSegment],
    ) -> None:
        if agent not in self.__agent_log:
            self.__agent_log[agent] = SummaryLog()
        action_item = getattr(self.__agent_log[agent], action)
        action_item.append(segment)

    def set_source_videos(self, videos: list[Video]) -> None:
        if self.__source_videos:
            error_msg = "Source Videos cannot change once they are set."
            log.error(error_msg)
            raise ValueError(error_msg)

        self.__source_videos = videos
        log.info("Source videos set.")

    def add_output_segment(self, segment: VideoSegment, agent: str) -> None:
        if segment in self.__output:
            log.info("Segment is already in output.")
            return
        self.__output.append(segment)
        self.__to_discard.discard(segment)
        self.__to_include.discard(segment)
        self.__log_agent_action("output", agent, segment)
        log.info("Added video segment to output.")

    def include_segment(self, segment: VideoSegment, agent: str) -> None:
        self.__to_discard.discard(segment)
        self.__to_include.add(segment)
        self.__log_agent_action("include", agent, segment)
        log.info("Added video segment to 'include' set.")

    def discard_segment(self, segment: VideoSegment, agent: str) -> None:
        if segment in self.__output:
            log.info("Can't discard segment already in output.")
            return
        self.__to_include.discard(segment)
        self.__to_discard.add(segment)
        self.__log_agent_action("discard", agent, segment)
        log.info("Added video segment to 'discard' set.")

    def add_segments_to_pick(self, segments: set[VideoSegment], agent: str) -> None:
        self.__to_pick.append(segments)
        self.__log_agent_action("pick", agent, segments)
        log.info("Added video segments to 'pick' set.")


class SummarySegmentHandlerIO:
    @staticmethod
    def save(handler: SummarySegmentHandler, filepath: str) -> None:
        log.info("Saving SummarySegmentHandler to disk file.")
        dump(asdict(handler), open(filepath, "w"))

    @staticmethod
    def load(filepath: str) -> SummarySegmentHandler:
        log.info("Loading SummarySegmentHandler from disk file.")
        return from_dict(SummarySegmentHandler, load(open(filepath, "r")))
