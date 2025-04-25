import whisper_timestamped as whisper
from json import loads
from math import floor
from typing import Optional
from moviepy import VideoFileClip

from open_video_summary.adapters.llm import LLMAdapter, OllamaAdapter
from open_video_summary.core.segmenter.prompts import VideoSegmenterPrompts
from open_video_summary.entities.video import Video, VideoSegment
from open_video_summary.handlers.segment import SegmentsCluster


class VideoSegmenter:
    def __init__(
        self,
        whisper_model: str = "base",
        min_segment_length: int = 10,
        max_segment_length: int = 300,
        segment_overlap_ratio: float = 0.5,
        max_phrase_pause_interval: float = 0.7,
        max_subtopics: Optional[int] = None,
        prompts_template: VideoSegmenterPrompts = VideoSegmenterPrompts(),
        llm_adapter: LLMAdapter = OllamaAdapter(),
    ) -> None:
        self.whisper_model = whisper_model
        self.min_segment_length = min_segment_length
        self.max_segment_length = max_segment_length
        self.segment_overlap_ratio = segment_overlap_ratio
        self.max_phrase_pause_interval = max_phrase_pause_interval
        self.max_subtopics = max_subtopics
        self.prompts_template = prompts_template 
        self.llm_adapter = llm_adapter

    def transcribe_video(self, video_path: str, language: str):
        model = whisper.load_model(self.whisper_model)
        audio = whisper.load_audio(video_path)
        return whisper.transcribe(model, audio, language=language, verbose=True)

    def load_video_topics(self, full_document: str, video: Video) -> dict[str, str]:
        clip = VideoFileClip(video.path)
        video_duration = clip.duration

        max_subtopics = self.max_subtopics or floor(
            video_duration / self.min_segment_length
        )
        topics_prompt = self.prompts_template.generate_subtopics.format(
            full_video_transcript=full_document, max_subtopics=max_subtopics
        )
        topics_str = self.llm_adapter.generate_pattern(
            prompt=topics_prompt,
            pattern="(\{.*?\})",
            options={"format": "json", "temperature": 0.5},
        )
        return loads(topics_str)

    def list_overlapping_clusters(
        self, segmented_document: list[dict]
    ) -> list[SegmentsCluster]:
        cluster_list, cluster = [], SegmentsCluster()
        num_segments = len(segmented_document)

        overlap_seconds = int(self.min_segment_length * self.segment_overlap_ratio)

        # Create overlaping segment cluster_list
        for i, s in enumerate(segmented_document):
            raw_segment = VideoSegment(
                order=i,
                start=float(s["start"]),
                end=float(s["end"]),
                content=s["text"].strip(),
            )
            cluster.append(raw_segment)

            # Checking if current segment is the last one
            if i + 1 == num_segments:
                cluster_list.append(cluster)
                break

            # Checking if next segment is too close
            next_seg = segmented_document[i + 1]
            if (next_seg["start"] - raw_segment.end) < self.max_phrase_pause_interval:
                continue

            # Checking if segment matches ending criteria
            if (
                cluster.duration >= self.min_segment_length
                and cluster.ends_with_punctuation()
            ):
                cluster_list.append(cluster)
                cluster = cluster.next_overlaping_cluster(overlap_seconds)

        return cluster_list

    def classify_segment_topics(
        self, cluster_list: list[SegmentsCluster], topics: dict[str, str]
    ) -> list[VideoSegment]:
        min_segments: dict[int, dict] = {}

        for cluster in cluster_list:
            prompt = self.prompts_template.classify_subtopic.format(
                content=cluster.content, topics=topics
            )
            topics_str = self.llm_adapter.generate_pattern(
                prompt=prompt,
                pattern="(\{.*?\})",
                options={"format": "json", "temperature": 0.2},
            )

            cluster_topic = loads(topics_str)
            topic_id, _ = cluster_topic.popitem()

            for cluster_seg in cluster.segments:
                if (
                    cluster_seg.order is not None
                    and (cluster_seg.order not in min_segments
                    or cluster.duration
                    < min_segments[cluster_seg.order]["cluster_duration"])
                ):
                    cluster_seg.video_topic = topics[topic_id]
                    min_segments[cluster_seg.order] = {
                        "segment": cluster_seg,
                        "cluster_duration": cluster.duration,
                    }

        return [min_seg["segment"] for min_seg in min_segments.values()]

    def fuse_similar_segments(
        self, min_segments: list[VideoSegment]
    ) -> list[VideoSegment]:
        video_segments, video_segment = [], None
        for current_segment in min_segments:
            if video_segment is None:
                video_segment = current_segment
                continue

            segment_duration = video_segment.end - video_segment.start
            current_duration = current_segment.end - current_segment.start
            combined_duration = segment_duration + current_duration
           
            # Extend current segment
            if (video_segment.video_topic == current_segment.video_topic) and (combined_duration < self.max_segment_length):
                video_segment.content += f" {current_segment.content}"
                video_segment.end = current_segment.end
                continue

            # Flush current segment to segments list and start a new one
            video_segments.append(video_segment)
            video_segment = current_segment
            started_new = True

        if video_segment is not None:
            video_segments.append(video_segment)

        return video_segments

    def adjust_segments_order(self, segments: list[VideoSegment]) -> list[VideoSegment]:
        adjusted_segments = []
        for i, seg in enumerate(segments):
            seg.order = i
            adjusted_segments.append(seg)
        return adjusted_segments

    def fix_segments_content(self, segments: list[VideoSegment]) -> list[VideoSegment]:
        adjusted_segments = []
        for seg in segments:
            prompt = self.prompts_template.fix_transcription.format(content=seg.content)
            response = self.llm_adapter.generate_pattern(
                prompt=prompt,
                pattern="([\w \.\?!,:;ºª\-]+)",
                options={"format": "json", "temperature": 0.2},
            )
            seg.content = response.strip()
            adjusted_segments.append(seg)
        return adjusted_segments

    def create_video_segments(self, video: Video, language: str = "pt") -> Video:
        whisper_result = self.transcribe_video(video.path, language)
        full_document = whisper_result.get("text")
        segmented_document = whisper_result.get("segments")

        # Adding topics to video dataclass
        if not video.topics:
            video_topics = self.load_video_topics(full_document, video)
            video.topics = list(video_topics.values())

        segment_clusters = self.list_overlapping_clusters(segmented_document)
        labled_min_segments = self.classify_segment_topics(
            segment_clusters, video_topics
        )
        segments = self.fuse_similar_segments(labled_min_segments)
        segments = self.adjust_segments_order(segments)
        segments = self.fix_segments_content(segments)

        # Adding segments to video dataclass
        video.segments = segments

        return video
