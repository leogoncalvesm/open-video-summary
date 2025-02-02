from open_video_summary.utils import log
from open_video_summary.entities.video import Video
from open_video_summary.classifiers.image import ObjectDetector
from open_video_summary.classifiers.text import BinaryTextClassifier
from open_video_summary.utils.processing.video import VideoProcessor
from open_video_summary.handlers.summary import SummarySegmentHandler
from open_video_summary.core.selection_criteria.base import SelectionCriteria


class ObjectContentSubjectivity(SelectionCriteria):
    def __init__(
        self,
        subjectivity_classifier: BinaryTextClassifier,
        object_detector: ObjectDetector,
        object_search_fps: int = 1,
        object_search_grayscale: bool = True,
        include_subjectivity: bool = False,
    ) -> None:
        super().__init__(read_from="source")
        self.object_detector = object_detector
        self.subjectivity_classifier = subjectivity_classifier
        self.object_search_fps = object_search_fps
        self.object_search_grayscale = object_search_grayscale
        self.include_subjectivity = include_subjectivity

    def evaluate(self, handler: SummarySegmentHandler) -> SummarySegmentHandler:
        videos = [
            video
            for video in self.get_criteria_input(handler)
            if isinstance(video, Video)
        ]
        log.info(f"Found {len(videos)} videos to execute {self.name} criteria.")

        for video in videos:
            segments = self.remove_discarded(handler, video.segments)
            segments = self.remove_outputted(handler, segments)
            for segment in segments:
                if self.segment_contains_object(
                    video_path=video.path, start=segment.start, end=segment.end
                ) and self.segment_is_subjective(content=segment.content):
                    action = self.include if self.include_subjectivity else self.discard
                    action(handler, segment)

        return handler

    def segment_contains_object(
        self, video_path: str, start: float, end: float
    ) -> bool:
        log.info(f"Detecting object in video {video_path} from {start} to {end}.")
        frames = VideoProcessor.retrieve_video_frames(
            video_path,
            target_fps=self.object_search_fps,
            grayscale=self.object_search_grayscale,
            start_second=start,
            end_second=end,
        )

        contains = any(map(self.object_detector.detect, frames))

        log_str = (
            f"Segment {'does not contain' if not contains else 'contains'} object."
        )
        log.info(log_str)

        return contains

    def segment_is_subjective(self, content: str) -> bool:
        log.info(f"Classifying subjectivity in segment content.")
        is_subjective = self.subjectivity_classifier.classify(content)

        log_str = f"Content {'is not' if not is_subjective else 'is'} subjective."
        log.info(log_str)
        return is_subjective
