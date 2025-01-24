import os
import re
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
import torch
import whisper
from yt_dlp import YoutubeDL

@dataclass
class TranscriptionResult:
    video_id: str
    video_url: str
    audio_file: str
    transcription: str

class YouTubeTranscriber:
    def __init__(self, audio_dir: str = "./data/audios", transcription_dir: str = "./data/transcriptions"):
        """Initialize the transcriber with directories for audio and transcription storage."""
        self.audio_dir = os.path.abspath(audio_dir)
        self.transcription_dir = os.path.abspath(transcription_dir)
        self.model = None
        
        # Create necessary directories
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.transcription_dir, exist_ok=True)

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from a YouTube URL."""
        patterns = [
            r'(?:v=|/)([0-9A-Za-z_-]{11}).*',  # Standard YouTube URL
            r'youtu\.be/([0-9A-Za-z_-]{11})'    # Shortened YouTube URL
        ]
        
        for pattern in patterns:
            if match := re.search(pattern, url):
                return match.group(1)
        return None

    def _get_transcription_path(self, video_id: str) -> str:
        """Get the full path for a transcription file."""
        return os.path.join(self.transcription_dir, f"{video_id}.json")

    def _transcription_exists(self, video_id: str) -> bool:
        """Check if transcription already exists for a video ID."""
        transcript_path = self._get_transcription_path(video_id)
        exists = os.path.exists(transcript_path)
        if exists:
            # Verify the file is valid JSON and has content
            try:
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    return all(key in content for key in ['video_id', 'video_url', 'transcription'])
            except (json.JSONDecodeError, IOError):
                # If file is corrupted or empty, consider it as non-existent
                return False
        return False

    def _download_audio(self, video_url: str, video_id: str) -> Optional[str]:
        """Download audio from YouTube URL."""
        # Double-check transcription doesn't exist before downloading
        if self._transcription_exists(video_id):
            print(f"Transcription found for {video_id} before download, skipping...")
            return None
            
        output_path = os.path.join(self.audio_dir, f"{video_id}.mp3")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(self.audio_dir, f'{video_id}.%(ext)s'),
            'quiet': True,
        }

        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(video_url, download=True)
            return output_path if os.path.exists(output_path) else None
        except Exception as e:
            print(f"Error downloading audio for {video_id}: {str(e)}")
            return None

    def _load_model(self):
        """Load the Whisper model if not already loaded."""
        if self.model is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = whisper.load_model("medium", device=device)

    def _save_transcription(self, result: TranscriptionResult):
        """Save transcription result to JSON file."""
        output_path = self._get_transcription_path(result.video_id)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(vars(result), f, indent=2, ensure_ascii=False)

    async def transcribe_videos(self, video_urls: List[str]) -> Dict[str, TranscriptionResult]:
        """
        Process a list of YouTube URLs, downloading and transcribing each video.
        Skips videos that have already been transcribed.
        """
        results = {}

        for url in video_urls:
            video_id = self._extract_video_id(url)
            if not video_id:
                print(f"Invalid YouTube URL: {url}")
                continue

            # Check if transcription exists before any processing
            if self._transcription_exists(video_id):
                print(f"Valid transcription already exists for video {video_id}, skipping...")
                # Load existing transcription
                try:
                    with open(self._get_transcription_path(video_id), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        results[video_id] = TranscriptionResult(**data)
                    continue
                except Exception as e:
                    print(f"Error loading existing transcription for {video_id}: {str(e)}")

            print(f"Processing video {video_id}...")
            
            # Download audio
            audio_path = self._download_audio(url, video_id)
            if not audio_path:
                continue

            # Double-check transcription hasn't been created while downloading
            if self._transcription_exists(video_id):
                print(f"Transcription appeared for {video_id} after download, skipping...")
                continue

            try:
                # Load model only when needed
                if self.model is None:
                    self._load_model()
                
                # Transcribe
                print(f"Transcribing {video_id}...")
                result = self.model.transcribe(audio_path)
                
                # Final check before saving
                if self._transcription_exists(video_id):
                    print(f"Transcription appeared for {video_id} before saving, skipping...")
                    continue
                
                # Create and save result
                transcription_result = TranscriptionResult(
                    video_id=video_id,
                    video_url=url,
                    audio_file=audio_path,
                    transcription=result["text"]
                )
                self._save_transcription(transcription_result)
                results[video_id] = transcription_result
                print(f"Successfully transcribed video {video_id}")

            except Exception as e:
                print(f"Error transcribing video {video_id}: {str(e)}")
                continue

        return results

def main():
    video_urls = [
        "https://www.youtube.com/watch?v=sNa_uiqSlJo",
    ]
    
    transcriber = YouTubeTranscriber()
    results = transcriber.transcribe_videos(video_urls)
    
    for video_id, result in results.items():
        print(f"\nProcessed: {video_id}")
        print(f"Audio file: {result.audio_file}")

if __name__ == "__main__":
    main()