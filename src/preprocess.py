# -*- coding: utf-8 -*-
"""youtube-chat.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19-8kDZAC3J6cf4SoSXhciEChhiAV68i0
"""

import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydantic import BaseModel, ConfigDict
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from yt_dlp import YoutubeDL


@dataclass
class AppConfig:
    save_dir: Path = Path("/content/data")


@dataclass
class RunConfig:
    seconds_per_frame: int = 5


class VideoMetadata(BaseModel):
    id: str
    title: str
    title_format: str
    video_file: str
    audio_file: str
    transcription_file: str
    img_file: str
    text_file: Optional[str] = None

    def __getitem__(self, item):
        return getattr(self, item)


class TranscriptionChunk(BaseModel):
    timestamp: List[str]
    text: str


class Transcription(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    text: str
    chunks: List[TranscriptionChunk]

    def __getitem__(self, item):
        return getattr(self, item)


def get_yt_id_from_url(url: str) -> str:
    # Do not ask me about this regex. t(o.ot)
    pattern = r"^https:\/\/(?:www\.)*(?:youtube\.com\/watch\?v=|youtu\.be\/)([\w-]*)(?:\?\w=.*)*"
    matches = re.findall(pattern, url)
    if len(matches) == 0:
        raise ValueError("Please use a Youtube link.")
    return matches[0]


class VideoIngestionPipeline:
    metadata_save_dir: Path
    video_save_dir: Path
    audio_save_dir: Path
    text_save_dir: Path

    def __init__(self, config: AppConfig):
        self.metadata_save_dir = config.save_dir / "metadata"
        self.video_save_dir = config.save_dir / "video"
        self.img_save_dir = config.save_dir / "img"
        self.audio_save_dir = config.save_dir / "audio"
        self.text_save_dir = config.save_dir / "text"
        self.transcription_save_dir = config.save_dir / "transcription"

        Path(config.save_dir).mkdir(parents=True, exist_ok=True)
        self.metadata_save_dir.mkdir(parents=True, exist_ok=True)
        self.video_save_dir.mkdir(parents=True, exist_ok=True)
        self.img_save_dir.mkdir(parents=True, exist_ok=True)
        self.audio_save_dir.mkdir(parents=True, exist_ok=True)
        self.text_save_dir.mkdir(parents=True, exist_ok=True)
        self.transcription_save_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_name(self, path: str | Path, return_ext: bool = False):
        if isinstance(path, str):
            path_split = path.split("/")[-1].split(".")
            if return_ext:
                return path_split[0], path_split[1]
            return path_split[0]

        if isinstance(path, Path):
            if return_ext:
                return path.stem, path.suffix
            return path.stem

        raise TypeError("path must be instance of str or Path.")

    def _format_ts(self, seconds: int) -> str:
        m, s = divmod(seconds, 60)
        return f"{m:02d}:{s:02d}"

    def _reformat_youtube_trans(self, transcription: list[dict]) -> dict[str, str]:
        chunks = []
        merged_text = ""
        for chunk in transcription:
            start = floor(chunk["start"])
            end = floor(start + chunk["duration"])
            chunks.append(
                {
                    "timestamp": (self._format_ts(start), self._format_ts(end)),
                    "text": chunk["text"],
                }
            )
            merged_text += " " + chunk["text"]
        formatted = {
            "text": merged_text,
            "chunks": chunks,
        }
        return formatted

    def _title_to_filename(self, title: str) -> str:
        # Convert to lowercase
        title = title.lower()

        # Remove special characters
        title = re.sub(r"[^\w\s-]", "", title)

        # Replace spaces with hyphens
        title = re.sub(r"\s", "-", title)
        # Remove double hyphens.
        title = re.sub(r"-+", "-", title)

        return title

    def download_video_from_youtube(self, url: str) -> dict[str, str]:
        # TODO: beautify title.
        params = {
            "paths": {"home": str(self.video_save_dir)},
            "outtmpl": {"default": "%(title)s.%(ext)s"},
            "format": "mp4/bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                }
            ],
            "keepvideo": True,
            "windowsfilenames": True,
        }
        with YoutubeDL(params=params) as ydl:
            info = ydl.extract_info(url, download=True)
            info = ydl.sanitize_info(info)
            info = cast(Dict[str, Any], info)

        title_format = self._title_to_filename(info.get("title", ""))
        video_path = Path("")
        audio_path = ""
        if info["requested_downloads"]:
            video_path = Path(info["requested_downloads"][0]["filename"])
            audio_path = info["requested_downloads"][0]["filepath"]

        metadata = {
            "id": info.get("id"),
            "title": info.get("title"),
            "title_format": title_format,
            "video_file": str(self.video_save_dir / f"{title_format}.mp4"),
            "audio_file": str(self.audio_save_dir / f"{title_format}.wav"),
        }
        # Move audio file to correct folders.
        shutil.move(audio_path, metadata["audio_file"])
        # Rename video file.
        video_path.rename(Path(metadata["video_file"]))

        return metadata

    def video_to_audio(self, video_path: str | Path) -> Path:
        video_name = self._get_file_name(video_path)
        video_name = self._title_to_filename(str(video_name))
        clip = VideoFileClip(video_path)

        print(video_name, video_path)
        audio = cast(AudioFileClip, clip.audio)
        print(audio)
        audio_path = self.audio_save_dir / f"{video_name}.wav"
        audio.write_audiofile(audio_path)

        return audio_path

    def video_to_images(self, video_path: str, seconds_per_frame: int = 5) -> str:
        video_name = self._get_file_name(video_path)
        fps = 1 / seconds_per_frame

        clip = VideoFileClip(video_path)
        img_path = str(self.img_save_dir / f"{video_name}-frame-%04d.png")
        clip.write_images_sequence(img_path, fps=fps)
        return img_path

    def audio_to_text(
        self,
        audio_path: str,
        model: str = "openai/whisper-large-v3",
        device: str = "mps",
        chunk_length_s: int = 30,
        batch_size: int = 10,
    ) -> Any:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,  # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
            torch_dtype=torch.float16,
            device=device,  # or mps for Mac devices
            generate_kwargs={"language": "english"},
        )
        # Output: dict with text and chunks.
        transcription = pipe(
            audio_path,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            return_timestamps=True,
        )
        audio_name = self._get_file_name(audio_path)
        text_path = self.text_save_dir / f"{audio_name}.json"
        with open(text_path, "w") as f:
            json.dump(transcription, f)

        return transcription, text_path

    def get_youtube_transcription(
        self, video_url: str, video_title: str
    ) -> tuple[dict[str, str], Path]:
        video_id = get_yt_id_from_url(video_url)

        trans_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcription = None
        # Prioritize manually created scripts than auto generated.
        for transcript in trans_list:
            if not transcript.is_generated and transcript.language_code == "en":
                transcription = transcript.fetch()
                break
        if not transcription:
            transcription = YouTubeTranscriptApi.get_transcript(video_id)
        transcription = self._reformat_youtube_trans(transcription)
        text_path = self.text_save_dir / f"{video_title}.json"
        with open(text_path, "w") as f:
            json.dump(transcription, f)

        return transcription, text_path

    def run(
        self, input_path_or_url: str | Path, run_config: RunConfig
    ) -> VideoMetadata:
        # If youtube url is provided.
        if re.match("^https?://", str(input_path_or_url)):
            input_path_or_url = cast(str, input_path_or_url)
            # For youtube videos, already got video and audio.
            video_metadata = self.download_video_from_youtube(input_path_or_url)
            # Also, use subtitle api to get transcription.
            transcription, text_path = self.get_youtube_transcription(
                input_path_or_url, video_metadata["title_format"]
            )
        else:
            # Split audio from video
            audio_path = self.video_to_audio(input_path_or_url)
            video_name = self._get_file_name(input_path_or_url)
            video_metadata = {
                "id": str(uuid.uuid4()),
                "title": video_name,
                "title_format": video_name,
                "video_file": str(input_path_or_url),
                "audio_file": str(audio_path),
            }
            # Transcribe video.
            transcription, text_path = self.audio_to_text(video_metadata["audio_file"])

        # Make a text file with each line is timestamp + transcription.
        transcription = cast(Transcription, transcription)
        text = ""
        for chunk in transcription["chunks"]:
            ts = chunk["timestamp"]

            text += str(ts[0]) + ": "
            text += chunk["text"].replace("\n", " ") + "\n"

        transcription_path = self.transcription_save_dir / f"{text_path.stem}.txt"
        with open(
            transcription_path,
            "w",
        ) as f:
            f.write(text)

        video_metadata["text_file"] = str(text_path)
        video_metadata["transcription_file"] = str(transcription_path)
        # Split video in to frames.
        img_path = self.video_to_images(
            video_metadata["video_file"], run_config.seconds_per_frame
        )
        video_metadata["img_file"] = str(img_path)

        with open(
            f"{self.metadata_save_dir}/{video_metadata['title_format']}.json", "w"
        ) as f:
            json.dump(video_metadata, f)

        return cast(VideoMetadata, video_metadata)


# ingest_pipe = VideoIngestionPipeline(AppConfig())

# ingest_pipe.run("https://www.youtube.com/watch?v=9RhWXPcKBI8&t=829s", RunConfig())

# ingest_pipe.run("/content/data/video/sample.mp4", RunConfig())
