import numpy as np
from pathlib import Path
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from PIL import Image


def _ken_burns_clip(image_path: Path, duration: float, fps: int, zoom: float = 0.04):
    """
    Slow zoom-in (Ken Burns) effect.
    zoom=0.04 means 4% scale increase over the full clip duration.
    Implemented as a per-frame filter on an ImageClip.
    """
    img_array = np.array(Image.open(str(image_path)).convert("RGB"))
    h, w = img_array.shape[:2]

    def make_frame(t):
        progress = t / duration
        scale = 1.0 + zoom * progress          # grows from 1.0 to 1+zoom
        crop_h = int(h / scale)
        crop_w = int(w / scale)
        y0 = (h - crop_h) // 2
        x0 = (w - crop_w) // 2
        cropped = img_array[y0 : y0 + crop_h, x0 : x0 + crop_w]
        resized = np.array(
            Image.fromarray(cropped).resize((w, h), Image.LANCZOS)
        )
        return resized

    from moviepy.video.VideoClip import VideoClip
    clip = VideoClip(make_frame, duration=duration)
    clip.fps = fps
    return clip


class VideoAssembler:
    def __init__(self, config):
        self.config = config

    def _build_clip(self, image_path: Path, audio_path: Path):
        audio = AudioFileClip(str(audio_path))
        duration = audio.duration

        if self.config.ken_burns:
            video = _ken_burns_clip(image_path, duration, self.config.fps)
        else:
            video = ImageClip(str(image_path)).set_duration(duration)

        return video.set_audio(audio)

    def assemble(self, page_data: list[dict], output_path: Path):
        """
        page_data: list of {'image': Path, 'audio': Path, 'page_num': int}
        """
        print(f"\n[video] Building {len(page_data)} clips...")
        clips = []
        for item in page_data:
            try:
                clip = self._build_clip(item["image"], item["audio"])
                clips.append(clip)
            except Exception as e:
                print(f"  [video] Skipping page {item['page_num']}: {e}")

        if not clips:
            raise RuntimeError("No clips to assemble.")

        print(f"[video] Concatenating and exporting → {output_path}")
        final = concatenate_videoclips(clips, method="compose")
        final.write_videofile(
            str(output_path),
            fps=self.config.fps,
            codec=self.config.video_codec,
            audio_codec=self.config.audio_codec,
            threads=4,
            logger="bar",
        )
        print(f"[video] Done: {output_path}")
