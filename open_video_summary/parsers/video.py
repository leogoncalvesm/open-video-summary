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
    def video_from_dict(video_data: dict) -> Video:
        return from_dict(data_class=Video, data=video_data)

class VideoDumper:
    @staticmethod
    def dump_videos_to_json(videos: list[Video], json_file: str) -> None:
        videos_data = list(map(VideoDumper.video_to_dict, videos))

        with open(json_file, "w") as file:
            dump(videos_data, file, indent=4)

    @staticmethod
    def video_to_dict(video: Video) -> dict:
        return asdict(video)


class SummaryWriter:
    @staticmethod
    def write_video_summary(video: Video, fadein_seconds: float = 0.5, fadeout_seconds: float = 0.5) -> None:
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
