from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import os
import json
import glob
import matplotlib.pyplot as plt
from collections import defaultdict

nltk.download('punkt')

class TranscriptAnalyzer:
    def __init__(self, 
                 summary_dir: str = "./data/summary_transcripts",
                 whisper_dir: str = "./data/transcriptions"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.summary_dir = os.path.abspath(summary_dir)
        self.whisper_dir = os.path.abspath(whisper_dir)
        
    def compare_transcripts(self, text1: str, text2: str) -> Tuple[float, List[float]]:
        """
        Compare two texts using sentence embeddings and cosine similarity.
        Returns both the average similarity and list of individual sentence similarities.
        """
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
        
        # Return both average and individual similarities
        return np.mean(similarities), similarities

    def plot_similarity_frequencies(self, original_similarities: List[float], cleaned_similarities: List[float], video_id: str):
        """Create a line plot comparing similarity frequencies with improved binning."""
        plt.figure(figsize=(12, 7))
        
        # Create more granular bins (100 bins between 0 and 1)
        bins = np.linspace(0, 1, 50)
        
        # Plot histograms
        plt.hist(original_similarities, bins=bins, alpha=0.5, label='Original Transcript', 
                density=True, color='blue')
        plt.hist(cleaned_similarities, bins=bins, alpha=0.5, label='Cleaned Transcript',
                density=True, color='orange')
        
        # Add kernel density estimation for smoother curves
        from scipy.stats import gaussian_kde
        
        def plot_kde(data, color):
            kde = gaussian_kde(data)
            x_range = np.linspace(0, 1, 200)
            plt.plot(x_range, kde(x_range), color=color, linewidth=2)
        
        plot_kde(original_similarities, 'blue')
        plot_kde(cleaned_similarities, 'orange')
        
        plt.title(f'Sentence Similarity Distribution - Video {video_id}')
        plt.xlabel('Similarity Score')
        plt.ylabel('Density')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Add summary statistics to plot
        stats_text = (
            f'Original Mean: {np.mean(original_similarities):.3f}\n'
            f'Cleaned Mean: {np.mean(cleaned_similarities):.3f}\n'
            f'Original Std: {np.std(original_similarities):.3f}\n'
            f'Cleaned Std: {np.std(cleaned_similarities):.3f}'
        )
        plt.text(0.02, 0.98, stats_text, 
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot
        os.makedirs('plots', exist_ok=True)
        plt.savefig(f'plots/similarity_distribution_{video_id}.png', dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_transcripts(self) -> List[Dict]:
        """
        Analyze all transcripts in the summary_transcripts directory by comparing them
        with their corresponding whisper transcriptions.
        """
        results = []
        
        # Get all summary transcript files
        summary_files = glob.glob(os.path.join(self.summary_dir, "*.json"))
        
        for summary_file in summary_files:
            video_id = os.path.splitext(os.path.basename(summary_file))[0]
            whisper_file = os.path.join(self.whisper_dir, f"{video_id}.json")
            
            result = {
                "video_id": video_id,
                "original_similarity": None,
                "cleaned_similarity": None,
                "improvement": None,
                "error": None,
                "total_tokens": None
            }
            
            try:
                # Load summary transcripts
                with open(summary_file, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                    
                # Load whisper transcription
                with open(whisper_file, 'r', encoding='utf-8') as f:
                    whisper_data = json.load(f)
                
                whisper_transcript = whisper_data.get('transcription', '').replace('\n', ' ').strip()
                youtube_transcript = summary_data.get('youtube_transcript', '')
                cleaned_transcript = summary_data.get('cleaned_transcript', '')
                
                if whisper_transcript and youtube_transcript and cleaned_transcript:
                    # Calculate similarities
                    original_similarity, original_similarities = self.compare_transcripts(
                        whisper_transcript,
                        youtube_transcript
                    )
                    
                    cleaned_similarity, cleaned_similarities = self.compare_transcripts(
                        whisper_transcript,
                        cleaned_transcript
                    )
                    
                    # Create similarity distribution plot
                    self.plot_similarity_frequencies(original_similarities, cleaned_similarities, video_id)
                    
                    result.update({
                        "original_similarity": original_similarity,
                        "cleaned_similarity": cleaned_similarity,
                        "improvement": cleaned_similarity - original_similarity,
                        "total_tokens": summary_data.get('total_tokens', 0)
                    })
                else:
                    result["error"] = "Missing transcript data"
                    
            except FileNotFoundError:
                result["error"] = f"Missing corresponding file in {'whisper' if not os.path.exists(whisper_file) else 'summary'} directory"
            except Exception as e:
                result["error"] = str(e)
            
            results.append(result)
            
            # Print results for this video
            print(f"\nAnalysis for video {video_id}:")
            if result["error"]:
                print(f"Error: {result['error']}")
            else:
                print(f"Original similarity: {result['original_similarity']:.2%}")
                print(f"Cleaned similarity: {result['cleaned_similarity']:.2%}")
                print(f"Improvement: {result['improvement']:.2%}")
                print(f"Total tokens used: {result['total_tokens']}")
                print(f"Plot saved as: plots/similarity_distribution_{video_id}.png")
        
        return results

def main():
    # The video URLs are just for reference - the analysis will process all files in the directories
    print("Analyzing transcripts in data/summary_transcripts/ and data/transcriptions/...")
    
    analyzer = TranscriptAnalyzer()
    results = analyzer.analyze_transcripts()  # No arguments needed
    
    # Print summary statistics
    if results:
        successful_results = [r for r in results if r["improvement"] is not None]
        if successful_results:
            avg_improvement = np.mean([r["improvement"] for r in successful_results])
            total_tokens = sum(r["total_tokens"] for r in successful_results if r["total_tokens"])
            print("\n=== Overall Statistics ===")
            print(f"Average improvement: {avg_improvement:.2%}")
            print(f"Total tokens used: {total_tokens}")
            print(f"Successfully analyzed: {len(successful_results)} out of {len(results)} videos")
        else:
            print("\nNo successful comparisons to analyze")

if __name__ == "__main__":
    main() 
