import requests
import json
from typing import Generator, Optional

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize Ollama client with base URL."""
        self.base_url = base_url.rstrip('/')
        
    def generate_stream(
        self,
        payload: dict,
    ) -> Generator[str, None, None]:
        """
        Generate streaming response from Ollama.
        
        Args:
            payload (dict): The request payload to send to Ollama.
            
        Yields:
            Stream of tokens from the response
        """
        url = f"{self.base_url}/api/generate"
        
        # Prepare the request payload
        
        try:
            # Make streaming request
            with requests.post(url, json=payload, stream=True) as response:
                response.raise_for_status()
                
                # Process the stream
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if 'error' in chunk:
                            raise Exception(chunk['error'])
                        
                        if 'response' in chunk:
                            yield chunk['response']
                            
                        # Check if done
                        if chunk.get('done', False):
                            break
                            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Failed to communicate with Ollama: {str(e)}")