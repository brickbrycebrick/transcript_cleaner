# YouTube Transcript Analysis Tool

A Python tool that compares YouTube's auto-generated transcripts with Whisper transcriptions and analyzes their quality using semantic similarity.

## Features

- Downloads audio from YouTube videos
- Generates high-quality transcriptions using OpenAI's Whisper model
- Fetches YouTube's auto-generated transcripts
- Cleans and improves transcript quality using AI
- Compares transcripts using semantic similarity analysis
- Supports batch processing of multiple videos

## Requirements

- Python 3.8+
- FFmpeg (for audio processing)
- CUDA-compatible GPU (optional, for faster transcription)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/brickbrycebrick/transcript_cleaner.git
cd youtube_summary
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```
DEEPSEEK_API_KEY=your_api_key_here
```

## Project Structure

```
youtube_summary/
├── data/
│   ├── audios/         # Downloaded audio files
│   └── transcriptions/ # Generated transcriptions
├── ytsummary/
│   ├── analysis.py     # Transcript comparison logic
│   ├── cleaner.py      # Transcript cleaning using AI
│   ├── yt_summary.py   # YouTube transcript processing
│   └── yt_transcriber.py # Whisper transcription logic
├── .env                # API keys and configuration
├── .gitignore         # Git ignore rules
└── requirements.txt    # Python dependencies
```

## Usage

1. Basic transcript analysis:
```python
from ytsummary.analysis import TranscriptAnalyzer
import asyncio

async def main():
    video_urls = [
        "https://www.youtube.com/watch?v=example1",
        "https://www.youtube.com/watch?v=example2"
    ]
    
    analyzer = TranscriptAnalyzer()
    results = await analyzer.analyze_video_transcripts(video_urls)
    
    for result in results:
        print(f"\nAnalysis for {result['url']}:")
        if result['error']:
            print(f"Error: {result['error']}")
        else:
            print(f"Original similarity: {result['original_similarity']:.2%}")
            print(f"Cleaned similarity: {result['cleaned_similarity']:.2%}")
            print(f"Improvement: {result['improvement']:.2%}")

if __name__ == "__main__":
    asyncio.run(main())
```

2. Just transcribe videos:
```python
from ytsummary.yt_transcriber import YouTubeTranscriber
import asyncio

async def main():
    transcriber = YouTubeTranscriber()
    results = await transcriber.transcribe_videos([
        "https://www.youtube.com/watch?v=example"
    ])

if __name__ == "__main__":
    asyncio.run(main())
```

## Notes

- Transcriptions are cached in the `data/transcriptions` directory
- Audio files are saved in the `data/audios` directory
- The tool skips processing if a transcription already exists
- GPU acceleration is used automatically if available

## License

MIT License - feel free to use and modify as needed. 