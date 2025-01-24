from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from yt_transcriber import YouTubeTranscriber
from yt_summary import YouTubeProcessor
import asyncio
import json

nltk.download('punkt')

class TranscriptAnalyzer:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.summarizer = YouTubeProcessor()
        self.transcriber = YouTubeTranscriber()
        
    def compare_transcripts(self, text1: str, text2: str) -> float:
        """Compare two texts using sentence embeddings and cosine similarity."""
        # Split texts into sentences
        sentences1 = sent_tokenize(text1)
        sentences2 = sent_tokenize(text2)
        
        # Get embeddings for all sentences
        embeddings1 = self.model.encode(sentences1)
        embeddings2 = self.model.encode(sentences2)
        
        # Calculate similarity between each pair of sentences
        similarities = []
        for emb1 in embeddings1:
            # Reshape embeddings for cosine_similarity
            emb1_reshaped = emb1.reshape(1, -1)
            
            # Calculate similarities with all sentences in text2
            sentence_similarities = [
                cosine_similarity(emb1_reshaped, emb2.reshape(1, -1))[0][0]
                for emb2 in embeddings2
            ]
            
            # Take the maximum similarity (best matching sentence)
            similarities.append(max(sentence_similarities))
        
        # Return average similarity
        return np.mean(similarities)

    async def analyze_video_transcripts(self, video_urls: List[str]) -> List[Dict]:
        """
        Analyze transcripts from multiple sources for a list of videos.
        Returns similarity scores comparing whisper transcripts to both original and cleaned YouTube transcripts.
        """
        results = []
        
        try:
            # Get whisper transcriptions
            print("\nGetting Whisper transcriptions...")
            whisper_results = await self.transcriber.transcribe_videos(video_urls)
            
            # Get YouTube transcripts (original and cleaned)
            print("\nGetting YouTube transcripts...")
            youtube_results = await self.summarizer.process_multiple_videos(video_urls)
            
            # Compare transcripts
            for url in video_urls:
                result = {
                    "url": url,
                    "original_similarity": None,
                    "cleaned_similarity": None,
                    "improvement": None,
                    "error": None
                }
                
                try:
                    video_id = self.transcriber._extract_video_id(url)
                    if not video_id:
                        result["error"] = "Could not extract video ID"
                        results.append(result)
                        continue

                    # Find matching transcripts
                    whisper_transcript = None
                    youtube_result = None
                    
                    # Find whisper transcript
                    if video_id in whisper_results:
                        whisper_transcript = whisper_results[video_id].transcription
                    else:
                        # Try to load from existing transcription file
                        try:
                            transcript_path = self.transcriber._get_transcription_path(video_id)
                            with open(transcript_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                whisper_transcript = data.get('transcription')
                        except Exception as e:
                            print(f"Could not load existing transcription for {video_id}: {str(e)}")
                    
                    # Find YouTube transcript
                    for res in youtube_results:
                        if res['video_id'] == video_id:
                            youtube_result = res
                            break
                    
                    if not whisper_transcript:
                        result["error"] = f"No Whisper transcription found for video {video_id}"
                    elif not youtube_result:
                        result["error"] = f"No YouTube transcription found for video {video_id}"
                    elif not youtube_result['success']:
                        result["error"] = f"YouTube transcription failed: {youtube_result.get('error', 'Unknown error')}"
                    else:
                        # Calculate similarities
                        original_similarity = self.compare_transcripts(
                            whisper_transcript,
                            youtube_result['transcript']
                        )
                        
                        cleaned_similarity = self.compare_transcripts(
                            whisper_transcript,
                            youtube_result['cleaned_transcript']
                        )
                        
                        result.update({
                            "original_similarity": original_similarity,
                            "cleaned_similarity": cleaned_similarity,
                            "improvement": cleaned_similarity - original_similarity
                        })
                        
                except Exception as e:
                    result["error"] = f"Error processing video: {str(e)}"
                
                results.append(result)
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            for url in video_urls:
                results.append({
                    "url": url,
                    "error": f"Analysis failed: {str(e)}"
                })
        
        return results

async def main():
    # Example video URLs
    video_urls = [
        "https://www.youtube.com/watch?v=sNa_uiqSlJo",
        "https://www.youtube.com/watch?v=x9Ekl9Izd38",
        "https://www.youtube.com/watch?v=sGUjmyfof4Q"
    ]
    
    analyzer = TranscriptAnalyzer()
    results = await analyzer.analyze_video_transcripts(video_urls)
    
    # Print results
    for result in results:
        print(f"\nAnalysis for {result['url']}:")
        if result['error']:
            print(f"Error: {result['error']}")
        else:
            print(f"Original transcript similarity: {result['original_similarity']:.2%}")
            print(f"Cleaned transcript similarity: {result['cleaned_similarity']:.2%}")
            print(f"Improvement: {result['improvement']:.2%}")

if __name__ == "__main__":
    asyncio.run(main()) 