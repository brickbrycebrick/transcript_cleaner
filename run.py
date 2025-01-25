import os
import asyncio
import json
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from ytsummary.yt_summary import YouTubeProcessor
from ytsummary.yt_transcriber import YouTubeTranscriber
from ytsummary.analysis import TranscriptAnalyzer

class Pipeline:
    def __init__(self):
        self.processor = YouTubeProcessor()
        self.transcriber = YouTubeTranscriber()
        self.analyzer = TranscriptAnalyzer()
        
    def _check_summary_exists(self, video_id: str) -> bool:
        """Check if summary transcript exists."""
        path = os.path.join("./data/summary_transcripts", f"{video_id}.json")
        return os.path.exists(path)
        
    def _check_transcription_exists(self, video_id: str) -> bool:
        """Check if whisper transcription exists."""
        path = os.path.join("./data/transcriptions", f"{video_id}.json")
        return os.path.exists(path)

    def plot_aggregate_frequencies(self, results: List[Dict]):
        """Create aggregate similarity distribution plot with improved visualization."""
        # Collect all similarities
        orig_similarities = []
        cleaned_similarities = []
        
        # Gather similarities from all videos
        for result in results:
            if not result.get("error"):
                video_id = result["video_id"]
                try:
                    # Use utf-8 encoding for reading JSON files
                    with open(f"./data/transcriptions/{video_id}.json", 'r', encoding='utf-8') as f:
                        whisper_data = json.load(f)
                    with open(f"./data/summary_transcripts/{video_id}.json", 'r', encoding='utf-8') as f:
                        summary_data = json.load(f)
                        
                    # Get similarities
                    _, orig_sims = self.analyzer.compare_transcripts(
                        whisper_data['transcription'],
                        summary_data['youtube_transcript']
                    )
                    _, cleaned_sims = self.analyzer.compare_transcripts(
                        whisper_data['transcription'],
                        summary_data['cleaned_transcript']
                    )
                    
                    orig_similarities.extend(orig_sims)
                    cleaned_similarities.extend(cleaned_sims)
                    
                except Exception as e:
                    print(f"Error processing {video_id} for aggregate plot: {str(e)}")
                    print(f"Skipping video {video_id} in aggregate analysis")
                    continue
        
        if not orig_similarities or not cleaned_similarities:
            print("No valid similarities found for aggregate plot")
            return
            
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Create more granular bins (50 bins between 0 and 1)
        bins = np.linspace(0, 1, 50)
        
        # Plot histograms
        plt.hist(orig_similarities, bins=bins, alpha=0.5, label='Original Transcripts', 
                density=True, color='blue')
        plt.hist(cleaned_similarities, bins=bins, alpha=0.5, label='Cleaned Transcripts',
                density=True, color='orange')
        
        # Add kernel density estimation for smoother curves
        from scipy.stats import gaussian_kde
        
        def plot_kde(data, color):
            kde = gaussian_kde(data)
            x_range = np.linspace(0, 1, 200)
            plt.plot(x_range, kde(x_range), color=color, linewidth=2)
        
        if orig_similarities:  # Only plot if we have data
            plot_kde(orig_similarities, 'blue')
        if cleaned_similarities:  # Only plot if we have data
            plot_kde(cleaned_similarities, 'orange')
        
        plt.title('Aggregate Sentence Similarity Distribution')
        plt.xlabel('Similarity Score')
        plt.ylabel('Density')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add summary statistics to plot
        if orig_similarities and cleaned_similarities:
            stats_text = (
                f'Original Mean: {np.mean(orig_similarities):.3f}\n'
                f'Cleaned Mean: {np.mean(cleaned_similarities):.3f}\n'
                f'Original Std: {np.std(orig_similarities):.3f}\n'
                f'Cleaned Std: {np.std(cleaned_similarities):.3f}\n'
                f'Total Sentences: {len(orig_similarities)}'
            )
            plt.text(0.02, 0.98, stats_text, 
                    transform=plt.gca().transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot with higher resolution
        os.makedirs('plots', exist_ok=True)
        plt.savefig('plots/aggregate_similarity_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nAggregate plot saved as: plots/aggregate_similarity_distribution.png")

    async def process_videos(self, video_urls: List[str]):
        """Process a list of videos through the entire pipeline."""
        print("\n=== Checking and Processing Videos ===")
        
        # Extract video IDs
        video_ids = []
        for url in video_urls:
            video_id = self.transcriber._extract_video_id(url)
            if not video_id:
                print(f"Invalid URL: {url}")
                continue
            video_ids.append(video_id)
            
        # Process missing files
        for video_id in video_ids:
            url = next(url for url in video_urls if video_id in url)
            
            # Check and create summary transcripts
            if not self._check_summary_exists(video_id):
                print(f"\nGenerating summary transcript for {video_id}...")
                await self.processor.process_video(url)
            
            # Check and create whisper transcriptions
            if not self._check_transcription_exists(video_id):
                print(f"\nGenerating whisper transcription for {video_id}...")
                await self.transcriber.transcribe_videos([url])
        
        # Run analysis
        print("\n=== Running Analysis ===")
        results = self.analyzer.analyze_transcripts()
        
        # Create aggregate plot
        print("\n=== Creating Aggregate Analysis ===")
        self.plot_aggregate_frequencies(results)
        
        return results

async def main():
    # List of video URLs to process
    video_urls = [
        "https://www.youtube.com/watch?v=sNa_uiqSlJo",
        "https://www.youtube.com/watch?v=x9Ekl9Izd38",
        "https://www.youtube.com/watch?v=sGUjmyfof4Q",
        "https://www.youtube.com/watch?v=CSE77wAdDLg",
        "https://www.youtube.com/watch?v=w9WE1aOPjHc"
    ]
    
    pipeline = Pipeline()
    results = await pipeline.process_videos(video_urls)
    
    # Print final summary
    print("\n=== Final Summary ===")
    successful_results = [r for r in results if r["improvement"] is not None]
    if successful_results:
        avg_improvement = np.mean([r["improvement"] for r in successful_results])
        total_tokens = sum(r["total_tokens"] for r in successful_results if r["total_tokens"])
        print(f"Average improvement: {avg_improvement:.2%}")
        print(f"Total tokens used: {total_tokens}")
        print(f"Successfully analyzed: {len(successful_results)} out of {len(results)} videos")
    else:
        print("No successful comparisons to analyze")

if __name__ == "__main__":
    asyncio.run(main()) 