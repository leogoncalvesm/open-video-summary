from __future__ import annotations

from pathlib import Path
from json import load, dump
from dacite import from_dict
from dataclasses import asdict
from moviepy.video.fx import FadeIn, FadeOut
from moviepy import VideoFileClip, concatenate_videoclips

from open_video_summary.entities.video import Video
from open_video_summary.utils.config import PROJECT_DIR


class VideoLoader:

    @staticmethod
    def load_videos_from_directory(directory: str, video_file_format: str = "mp4") -> list[Video]:
        return [
            VideoLoader.video_from_file(path.as_posix())
            for path in Path(directory).rglob(f"*.{video_file_format}")
        ]

    @staticmethod
    def load_videos_from_json(json_file: str) -> list[Video]:
        videos_data = load(open(json_file))

        for item in videos_data:
            item["path"] = (
                item["path"]
                if Path(item["path"]).is_absolute()
                else f"{PROJECT_DIR}/{item['path']}"
            )

        return list(map(VideoLoader.video_from_dict, videos_data))

    @staticmethod
    def video_from_file(video_file: str) -> Video:
        video_path = Path(video_file)

        # Check if the file exists
        if not video_path.exists():
            raise FileNotFoundError(f"Video file {video_file} does not exist.")

        # Check if the file is a video file
        if video_path.suffix not in {".mp4", ".mov", ".avi", ".mkv"}:
            raise ValueError(f"File {video_file} is not a valid video file.")

        return Video(
            name=video_path.stem,
            path=(
                video_path.absolute()
                if not video_path.is_absolute()
                else video_path
            ).as_posix()
        )
    
    @staticmethod
    def video_from_dict(video_data: dict) -> Video:
        return from_dict(data_class=Video, data=video_data)


class VideoDumper:
    @staticmethod
    def dump_videos_to_json(videos: list[Video], json_file: str) -> None:
        with open(json_file, "w") as file:
            dump(
                list(map(VideoDumper.video_to_dict, videos)),
                file,
                indent=4,
                ensure_ascii=False
            )

    @staticmethod
    def video_to_dict(video: Video) -> dict:
        return asdict(video)


class SummaryWriter:
    @staticmethod
    def write_video_summary(
        video: Video, fadein_seconds: float = 0.5, fadeout_seconds: float = 0.5
    ) -> None:
        fadein_fx = FadeIn(duration=fadein_seconds)
        fadeout_fx = FadeOut(duration=fadeout_seconds)

        clips_list = []
        for segment in video.segments:
            clip = VideoFileClip(segment.video_path).subclipped(
                segment.start, segment.end
            )
            clip = fadein_fx.apply(clip)
            clip = fadeout_fx.apply(clip)
            clips_list.append(clip)

        final_clip = concatenate_videoclips(clips_list)
        final_clip.write_videofile(video.path)
