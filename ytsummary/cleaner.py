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
        
        ** Transcript: **
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
    def __init__(self, chunk_size: int = 1000, batch_size: int = 3):
        """
        Initialize the transcript cleaner.
        
        Args:
            chunk_size: Approximate number of characters per chunk
            batch_size: Number of chunks to process in parallel
        """
        self.chunk_size = chunk_size
        self.batch_size = batch_size

    def _create_chunks(self, transcript: str) -> List[str]:
        """Split transcript into chunks of approximately equal size."""
        # Normalize line breaks and whitespace first
        transcript = ' '.join(transcript.split())
        
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

    async def _get_summary(self, transcript: str) -> Tuple[str, int]:
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
            return "", 0

    async def _clean_chunk(self, chunk: str, summary: str, prev_chunks: Optional[List[str]] = None) -> Tuple[str, int]:
        """Clean a single chunk using summary and previous chunks as context."""
        try:
            # Normalize chunk text
            chunk = ' '.join(chunk.split())
            
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
            cleaned_text = ' '.join(cleaned_text.split())  # Normalize whitespace
            total_tokens = response.usage.total_tokens
            
            print(f"\n========== Original Chunk ==========\n{chunk}")
            print(f"\n========== Cleaned Chunk ==========\n{cleaned_text}\n")
            return cleaned_text, total_tokens
            
        except Exception as e:
            print(f"Error cleaning chunk: {str(e)}")
            return chunk, 0

    async def _process_chunk_batch(self, chunks: List[str], summary: str, 
                                start_idx: int, prev_cleaned: List[str]) -> List[Tuple[str, int]]:
        """Process a batch of chunks in parallel while maintaining context."""
        async def process_chunk(chunk: str, position: int, completed_results: List[Tuple[str, int]]):
            # Get context from previous batch and completed chunks
            context_chunks = []
            
            # Add context from previous batch
            if prev_cleaned:
                context_chunks.extend(prev_cleaned[-2:])
                
            # Add context from completed chunks in current batch
            if position > 0:
                context_chunks.extend([result[0] for result in completed_results[:position][-2:]])
                
            context_chunks = context_chunks[-2:] if context_chunks else None
            return await self._clean_chunk(chunk, summary, context_chunks)
        
        # Process chunks with semaphore to control concurrency
        semaphore = asyncio.Semaphore(self.batch_size)
        async def bounded_process(chunk: str, position: int, completed_results: List[Tuple[str, int]]):
            async with semaphore:
                return await process_chunk(chunk, position, completed_results)
        
        # Track results while maintaining order
        results = []
        tasks = []
        
        for i, chunk in enumerate(chunks):
            task = bounded_process(chunk, i, results)
            tasks.append(task)
        
        # Wait for all chunks in batch to complete
        completed = await asyncio.gather(*tasks)
        return completed

    async def clean_transcript(self, transcript: str) -> Tuple[str, int]:
        if not transcript:
            return "", 0
            
        try:
            transcript = ' '.join(transcript.split())
            summary, total_tokens = await self._get_summary(transcript)
            chunks = self._create_chunks(transcript)
            
            if not chunks:
                return transcript, 0
                
            cleaned_chunks = []
            batch_number = 1
            total_batches = (len(chunks) + self.batch_size - 1) // self.batch_size
            
            for i in range(0, len(chunks), self.batch_size):
                print(f"\nProcessing batch {batch_number}/{total_batches}")
                batch = chunks[i:i + self.batch_size]
                prev_cleaned = cleaned_chunks[-2:] if cleaned_chunks else None
                
                try:
                    results = await self._process_chunk_batch(batch, summary, i, prev_cleaned)
                    
                    # Extract results while checking for failures
                    for j, (cleaned_text, batch_tokens) in enumerate(results):
                        if cleaned_text and batch_tokens > 0:
                            cleaned_chunks.append(cleaned_text.strip())
                            total_tokens += batch_tokens
                        else:
                            print(f"Warning: Chunk {i + j} failed to process, using original")
                            cleaned_chunks.append(batch[j].strip())
                            
                except Exception as e:
                    print(f"Error processing batch {batch_number}: {str(e)}")
                    # On batch failure, add original chunks to maintain sequence
                    cleaned_chunks.extend([chunk.strip() for chunk in batch])
                    
                batch_number += 1
                
            cleaned_transcript = ' '.join(cleaned_chunks).strip()
            return cleaned_transcript, total_tokens
                
        except Exception as e:
            print(f"Error in transcript cleaning: {str(e)}")
            return transcript, 0
                
