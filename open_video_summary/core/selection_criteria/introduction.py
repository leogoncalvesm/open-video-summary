from math import ceil

from open_video_summary.utils import log
from open_video_summary.handlers.video import VideoHandler
from open_video_summary.entities.video import Video, VideoSegment
from open_video_summary.utils.processing.image import ImageProcessor
from open_video_summary.utils.processing.video import VideoProcessor
from open_video_summary.handlers.summary import SummarySegmentHandler
from open_video_summary.core.selection_criteria.base import SelectionCriteria


class Introduction(SelectionCriteria):
    def __init__(
        self,
        fps_to_compare: float = 1.0,
        compare_grayscale: bool = True,
        frame_diff_threshold: float = 0.7,  # Semelhança, não diff
        skip_frames: int = 1,
    ) -> None:
        super().__init__(read_from="source")
        self.fps_to_compare = fps_to_compare
        self.compare_grayscale = compare_grayscale
        self.frame_diff_threshold = frame_diff_threshold
        self.skip_frames = skip_frames

    def evaluate(self, handler: SummarySegmentHandler) -> SummarySegmentHandler:
        videos = [
            video
            for video in self.get_criteria_input(handler)
            if isinstance(video, Video)
        ]
        log.info(f"Retrieved {len(videos)} videos to execute {self.name} criteria.")

        shortest_intro = self.find_shortest_introduction(handler, videos)
        if shortest_intro:
            log.info(f"Found shortest introduction, including segments to output.")
            for segment in shortest_intro:
                self.output(handler, segment)

        return handler

    def find_shortest_introduction(
        self, handler: SummarySegmentHandler, source_videos: list[Video]
    ) -> list[VideoSegment]:
        log.info("Looking for shortest introduction.")

        min_end_sec, min_intro_segments = None, []
        for video in source_videos:
            intro_end_sec = self.get_video_introduction_end_second(video)
            intro_segments = VideoHandler.get_segments_until_second(
                video, intro_end_sec
            )

            # Keeping shortest introduction
            if intro_segments and (min_end_sec is None or intro_end_sec < min_end_sec):
                min_end_sec, min_intro_segments = intro_end_sec, intro_segments

            for segment in intro_segments:
                self.discard(handler, segment)

        return min_intro_segments

    def get_video_introduction_end_second(self, video: Video) -> int:
        log.info(f"Retrieving final second for intro in video {video.name}.")

        video_frames = VideoProcessor.retrieve_video_frames(
            video.path,
            target_fps=self.fps_to_compare,
            grayscale=self.compare_grayscale,
            start_second=self.skip_frames,
        )

        curr_frame = video_frames.pop(0)
        curr_histogram = ImageProcessor.get_frame_histogram(curr_frame)

        for frame, next_frame in enumerate(video_frames):
            # Calculating histogram intersection between frames
            next_histogram = ImageProcessor.get_frame_histogram(next_frame)
            histogram_intersec = ImageProcessor.compare_histograms(
                curr_histogram, next_histogram
            )

            # If matches threshold rule, returns the frame second
            if histogram_intersec < self.frame_diff_threshold:
                return ceil(frame / self.fps_to_compare)

            curr_frame, curr_histogram = next_frame, next_histogram

        return ceil(frame / self.fps_to_compare)
