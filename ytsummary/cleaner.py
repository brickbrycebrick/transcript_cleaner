from typing import Optional, Dict, List
import asyncio
import nltk
import os
from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# Load environment variables
load_dotenv()

# Configure OpenAI API
client = AsyncOpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
MODEL = 'deepseek-chat'

class PromptTemplates:
    @staticmethod
    def get_cleaning_prompt(context: str, target: str) -> Dict[str, str]:
        """Get the prompt configuration for transcript cleaning."""
        system_prompt = """You are an expert at improving transcript quality. 
        Analyze this transcript segment carefully and use the context to infer semantic meaning of the sentence."""
        
        prompt = f"""
        # Your task is to:
        1. Identify any grammatical errors or unclear phrasing
        2. Use context clues to infer correct words if there are obvious transcription errors
        3. Maintain the original meaning and speaker's intent
        4. Only make changes if you are confident they improve accuracy
        5. Return ONLY the corrected sentence, with no explanation

        # Use the context below to infer the purpose of the sentence:
        {context}

        # Target sentence to clean: {target}

        If the sentence appears correct, return it unchanged.
        """
        
        return {
            "system_prompt": system_prompt,
            "prompt": prompt
        }

class TranscriptCleaner:
    def __init__(self, window_size: int = 3, overlap: int = 1):
        """Initialize the transcript cleaner with configurable window size and overlap."""
        self.window_size = window_size
        self.overlap = overlap

    def _create_chunks(self, transcript: str) -> List[Dict]:
        """Create overlapping chunks from the transcript."""
        sentences = sent_tokenize(transcript)
        chunks = []
        
        # Handle case where transcript is too short for window size
        if len(sentences) <= self.window_size:
            chunks.append({
                'target': sentences[0] if sentences else "",
                'context': ' '.join(sentences),
                'position': 0
            })
            return chunks
            
        for i in range(0, len(sentences) - self.window_size + 1):
            chunk = {
                'target': sentences[i + self.overlap],  # The sentence we want to clean
                'context': ' '.join(sentences[i:i + self.window_size]),  # Full context window
                'position': i + self.overlap  # Keep track of position for reassembly
            }
            chunks.append(chunk)
        
        return chunks

    async def _clean_chunk(self, chunk: Dict) -> Dict:
        """Clean a single chunk of the transcript."""
        try:
            prompt_config = PromptTemplates.get_cleaning_prompt(
                context=chunk['context'],
                target=chunk['target']
            )
            
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": prompt_config["system_prompt"]},
                    {"role": "user", "content": prompt_config["prompt"]}
                ],
                temperature=0.7,
                max_tokens=5000
            )
            
            cleaned_text = response.choices[0].message.content.strip()
            return {
                'position': chunk['position'],
                'cleaned_text': cleaned_text
            }
        except Exception as e:
            print(f"Error cleaning chunk: {str(e)}")
            return {
                'position': chunk['position'],
                'cleaned_text': chunk['target']  # Return original text if cleaning fails
            }

    async def clean_transcript(self, transcript: str) -> str:
        """
        Clean a transcript by processing it in chunks with context.
        
        Args:
            transcript (str): The transcript to clean
            
        Returns:
            str: The cleaned transcript
        """
        if not transcript:
            return ""
            
        try:
            # Create chunks with context
            chunks = self._create_chunks(transcript)
            if not chunks:
                return transcript
            
            # Process chunks in parallel
            tasks = [self._clean_chunk(chunk) for chunk in chunks]
            cleaned_chunks = await asyncio.gather(*tasks)
            
            # Sort by position and reassemble
            cleaned_chunks.sort(key=lambda x: x['position'])
            cleaned_transcript = ' '.join(chunk['cleaned_text'] for chunk in cleaned_chunks)
            
            return cleaned_transcript
            
        except Exception as e:
            print(f"Error in transcript cleaning: {str(e)}")
            return transcript  # Return original transcript if cleaning fails
    
