import re
import asyncio
from youtube_transcript_api import YouTubeTranscriptApi
from typing import Optional, Dict, List
from dotenv import load_dotenv
from cleaner import TranscriptCleaner

# Load environment variables
load_dotenv()

class YouTubeProcessor:
    def __init__(self):
        self.video_id_regex = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
        self.cleaner = TranscriptCleaner()
        
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
            cleaned_transcript = await self.cleaner.clean_transcript(transcript)
            if not cleaned_transcript:
                raise ValueError("Could not clean transcript")
            result["cleaned_transcript"] = cleaned_transcript
            
            result["success"] = True
            
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
            # print("\nTranscript excerpt (first 300 chars):", result["transcript"][:300], "...")
            # print("\nCleaned Transcript excerpt:", result["cleaned_transcript"][:300], "...")
        else:
            print("Error:", result["error"])

if __name__ == "__main__":
    asyncio.run(main())