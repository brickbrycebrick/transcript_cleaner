from typing import Optional, Dict, List, Tuple
import asyncio
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI API
client = AsyncOpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
MODEL = 'deepseek-chat'
MODEL_2 = 'deepseek-reasoner'

class PromptTemplates:
    @staticmethod
    def get_summary_prompt(transcript: str) -> Dict[str, str]:
        """Get prompt for generating transcript summary."""
        system_prompt = """You are an expert at understanding and summarizing transcripts.
        Your task is to provide a concise and information-dense summary that captures the main topics and conceptual flow of the transcript."""
        
        prompt = f"""
        Please provide a concise summary of this transcript that captures:
        1. The main topics or themes
        2. The logical flow of ideas
        3. Any key context that would help understand individual segments
        
        Transcript:
        {transcript}
        """
        
        return {
            "system_prompt": system_prompt,
            "prompt": prompt
        }
    
    @staticmethod
    def get_cleaning_prompt(chunk: str, summary: str, prev_chunks: Optional[List[str]] = None) -> Dict[str, str]:
        """Get prompt for cleaning a chunk with summary and previous context."""
        system_prompt = """You are an expert at improving transcript quality.
        Your task is to clean and improve transcript segments while maintaining consistency with the overall context."""
        
        context = ""
        if prev_chunks:
            context = "\nPrevious context:\n" + "\n".join(prev_chunks)
            
        prompt = f"""
        **Overall transcript summary for context:**
        {summary}
        
        ** Previous transcript segments for context: **
        {context}
        
        ** Task description: **
        1. Clean up the following chunk of text to be grammatically correct and semantically consistent with the previous context.
        2. Ensure it flows naturally from the previous context (if any), if no context, then just clean the chunk.
        3. Maintain consistency with the overall summary while maintaining the original meaning and speaker's intent.
        4. Replace words that do not match the context of the transcript, but only make changes if you are confident they improve accuracy.
        5. Return ONLY the cleaned text, no explanations.
        
        ** Text to clean: **
        {chunk}
        """
        
        return {
            "system_prompt": system_prompt,
            "prompt": prompt
        }

class TranscriptCleaner:
    def __init__(self, chunk_size: int = 1000):
        """
        Initialize the transcript cleaner.
        
        Args:
            chunk_size: Approximate number of characters per chunk
        """
        self.chunk_size = chunk_size
        
    def _create_chunks(self, transcript: str) -> List[str]:
        """Split transcript into chunks of approximately equal size."""
        words = transcript.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    async def _get_summary(self, transcript: str) -> str:
        """Generate a summary of the entire transcript."""
        try:
            prompt_config = PromptTemplates.get_summary_prompt(transcript)
            
            response = await client.chat.completions.create(
                model=MODEL_2,
                messages=[
                    {"role": "system", "content": prompt_config["system_prompt"]},
                    {"role": "user", "content": prompt_config["prompt"]}
                ],
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            total_tokens = response.usage.total_tokens
            print(f"\n========== Transcript Summary ==========\n{summary}\n")
            return summary, total_tokens
            
        except Exception as e:
            print(f"Error generating summary: {str(e)}")
            return ""

    async def _clean_chunk(self, chunk: str, summary: str, prev_chunks: Optional[List[str]] = None) -> str:
        """Clean a single chunk using summary and previous chunks as context."""
        try:
            prompt_config = PromptTemplates.get_cleaning_prompt(
                chunk=chunk,
                summary=summary,
                prev_chunks=prev_chunks
            )
            
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": prompt_config["system_prompt"]},
                    {"role": "user", "content": prompt_config["prompt"]}
                ],
                temperature=0.7
            )
            
            cleaned_text = response.choices[0].message.content.strip()
            total_tokens_cleaning = response.usage.total_tokens
            print(f"\n========== Original Chunk ==========\n{chunk}")
            print(f"\n========== Cleaned Chunk ==========\n{cleaned_text}\n")
            return cleaned_text, total_tokens_cleaning
            
        except Exception as e:
            print(f"Error cleaning chunk: {str(e)}")
            return chunk

    async def clean_transcript(self, transcript: str) -> str:
        """
        Clean transcript using a hierarchical approach with summary and sequential context.
        
        Args:
            transcript: The transcript to clean
            
        Returns:
            str: The cleaned transcript
        """
        if not transcript:
            return ""
            
        try:
            # Get overall summary first
            summary, total_tokens_summary = await self._get_summary(transcript)
            total_tokens = total_tokens_summary
            if not summary:
                print("Warning: Failed to generate summary, proceeding with limited context")
                
            # Split into chunks
            chunks = self._create_chunks(transcript)
            if not chunks:
                return transcript
                
            # Clean chunks sequentially, maintaining context
            cleaned_chunks = []
            for i, chunk in enumerate(chunks):
                # Get previous chunks for context (up to 2)
                prev_chunks = cleaned_chunks[-2:] if cleaned_chunks else None
                
                # Clean current chunk
                cleaned_chunk, total_tokens_cleaning = await self._clean_chunk(chunk, summary, prev_chunks)
                cleaned_chunks.append(cleaned_chunk)
                total_tokens += total_tokens_cleaning

            # Reassemble transcript
            return ' '.join(cleaned_chunks), total_tokens
            
        except Exception as e:
            print(f"Error in transcript cleaning: {str(e)}")
            return transcript