from typing import Optional
from cv2 import (
    cvtColor,
    VideoCapture,
    CAP_PROP_FPS,
    COLOR_BGR2GRAY,
    CAP_PROP_FRAME_COUNT,
)


class VideoProcessor:
    @staticmethod
    def retrieve_video_frames(
        video_path: str,
        target_fps: int | float = 1,
        grayscale: bool = False,
        start_second: int | float = 0,
        end_second: Optional[int | float] = None,
    ) -> list:
        video = VideoCapture(video_path)
        source_fps = int(video.get(CAP_PROP_FPS))
        total_frames = int(video.get(CAP_PROP_FRAME_COUNT))
        end_second = end_second or (total_frames / source_fps)

        frames = []
        frames_interval = int(source_fps / target_fps)
        frame, success = -1, True
        while success:
            success, img = video.read()
            if not success:
                break

            frame += 1
            if (frame / source_fps) < start_second:
                continue

            if (frame / source_fps) > end_second:
                break

            if frame % frames_interval != 0:
                continue

            if grayscale:
                img = cvtColor(img, COLOR_BGR2GRAY)

            frames.append(img)

        video.release()
        return frames
