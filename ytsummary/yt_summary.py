import re
import os
import asyncio
import json
from youtube_transcript_api import YouTubeTranscriptApi
from typing import Optional, Dict, List
from dotenv import load_dotenv
from ytsummary.cleaner import TranscriptCleaner

# Load environment variables
load_dotenv()

class YouTubeProcessor:
    def __init__(self, summary_dir: str = "./data/summary_transcripts"):
        self.video_id_regex = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
        self.cleaner = TranscriptCleaner()
        self.summary_dir = os.path.abspath(summary_dir)
        
        # Create summary directory if it doesn't exist
        os.makedirs(self.summary_dir, exist_ok=True)
        
    def _get_summary_path(self, video_id: str) -> str:
        """Get the full path for a summary file."""
        return os.path.join(self.summary_dir, f"{video_id}.json")
        
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL using regex."""
        match = re.search(self.video_id_regex, url)
        return match.group(1) if match else None

    def get_transcript(self, video_id: str) -> Optional[str]:
        """Extract transcript from YouTube video."""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id)
            text_list = [entry['text'] for entry in transcript]
            return '\n'.join(text_list)
        except Exception as e:
            print(f"Error getting transcript: {str(e)}")
            return None

    def _save_result(self, result: Dict) -> None:
        """Save processing result to a JSON file."""
        if result["success"] and result["video_id"]:
            output_path = self._get_summary_path(result["video_id"])
            try:
                youtube_transcript = result["transcript"].replace("\n", " ").strip()
                cleaned_transcript = result["cleaned_transcript"].replace("...", " ").strip()
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "video_id": result["video_id"],
                        "youtube_transcript": youtube_transcript,
                        "cleaned_transcript": cleaned_transcript,
                        "success": result["success"],
                        "total_tokens": result["total_tokens"]
                    }, f, indent=2, ensure_ascii=False)
                print(f"Saved results to {output_path}")
            except Exception as e:
                print(f"Error saving results: {str(e)}")

    async def process_video(self, url: str) -> Dict:
        """
        Process a YouTube video URL and return video ID, transcript, and cleaned transcript.
        
        Args:
            url (str): YouTube video URL
            
        Returns:
            Dict: Dictionary containing video_id, transcript, and cleaned transcript
        """
        result = {
            "video_id": None,
            "transcript": None,
            "cleaned_transcript": None,
            "success": False,
            "error": None
        }
        
        try:
            # Extract video ID
            video_id = self.extract_video_id(url)
            if not video_id:
                raise ValueError("Invalid YouTube URL")
            result["video_id"] = video_id
            
            # Get transcript
            transcript = self.get_transcript(video_id)
            if not transcript:
                raise ValueError("Could not retrieve transcript")
            result["transcript"] = transcript

            # Clean transcript
            cleaned_transcript, total_tokens = await self.cleaner.clean_transcript(transcript)
            if not cleaned_transcript:
                raise ValueError("Could not clean transcript")
            result["cleaned_transcript"] = cleaned_transcript
            result["total_tokens"] = total_tokens
            result["success"] = True
            
            # Save result to file
            self._save_result(result)
            
        except Exception as e:
            result["error"] = str(e)
            
        return result

    async def process_multiple_videos(self, urls: List[str]) -> List[Dict]:
        """
        Process multiple YouTube video URLs and return their processed transcripts.
        
        Args:
            urls (List[str]): List of YouTube video URLs
            
        Returns:
            List[Dict]: List of dictionaries containing video_id, transcript, and processed transcript for each video
        """
        tasks = []
        for url in urls:
            print(f"\nProcessing video: {url}")
            tasks.append(self.process_video(url))
            
        results = await asyncio.gather(*tasks)
        
        for result in results:
            if result["success"]:
                print(f"Successfully processed video: {result['video_id']}")
            else:
                print(f"Failed to process video. Error: {result['error']}")
                
        return results

async def main():
    processor = YouTubeProcessor()
    
    # List of video URLs to process
    video_urls = [
        "https://www.youtube.com/watch?v=sNa_uiqSlJo"
    ]
    
    # Process multiple videos
    results = await processor.process_multiple_videos(video_urls)
    
    # Print results
    for i, result in enumerate(results, 1):
        print(f"\n--- Video {i} ---")
        if result["success"]:
            print("Video ID:", result["video_id"])
            print(f"Results saved to: {processor._get_summary_path(result['video_id'])}")
        else:
            print("Error:", result["error"])

if __name__ == "__main__":
    asyncio.run(main())