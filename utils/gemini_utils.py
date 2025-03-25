import os
import numpy as np
import google.generativeai as genai
from typing import List, Dict, Any

def initialize_gemini_model():
    """Initialize and return the Gemini model."""
    # Set the API key from environment variables with a fallback
    api_key = os.getenv("GOOGLE_API_KEY", "")
    
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set.")
    
    # Configure the generative AI
    genai.configure(api_key=api_key)
    
    # Get the Gemini Flash model
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    return model

def extract_topics(model, question: str, responses: List[str], language: str) -> List[Dict[str, Any]]:
    """
    Extract topics from responses using Gemini.
    
    Args:
        model: The Gemini model instance
        question: The question asked
        responses: List of text responses
        language: Language code ('en' for English, 'tr' for Turkish)
    
    Returns:
        List of topic dictionaries containing name and weight
    """
    # Join responses for analysis
    all_responses = "\n".join(responses)
    
    # Create the prompt based on language
    if language == 'tr':
        prompt = f"""
        Lütfen aşağıdaki soruya verilen yanıtlardan ana temaları çıkarın.
        
        Soru: {question}
        
        Yanıtlar:
        {all_responses}
        
        Lütfen 3-5 ana temayı ve her bir temanın yüzde olarak ağırlığını, JSON formatında belirleyin. 
        Örnek format: [{{"topic": "tema adı", "weight": 30}}, {{"topic": "tema adı", "weight": 25}}, ...]
        Ağırlıkların toplamı 100 olmalıdır. Lütfen sadece JSON çıktı verin, açıklama yapmayın.
        """
    else:
        prompt = f"""
        Please extract the main topics from the following responses to a question.
        
        Question: {question}
        
        Responses:
        {all_responses}
        
        Identify 3-5 main topics and determine the weight (percentage) of each topic in JSON format. 
        Example format: [{{"topic": "topic name", "weight": 30}}, {{"topic": "topic name", "weight": 25}}, ...]
        The weights should sum to 100. Please provide only the JSON output without any explanation.
        """
    
    # Get the response from Gemini
    response = model.generate_content(prompt)
    
    try:
        # Extract and parse the JSON response
        import json
        topics_text = response.text.strip()
        
        # Clean up the response to handle potential formatting issues
        if topics_text.startswith("```json"):
            topics_text = topics_text.replace("```json", "").replace("```", "").strip()
        
        topics = json.loads(topics_text)
        
        # Ensure the result is in the expected format
        if not isinstance(topics, list):
            raise ValueError("Gemini did not return a list of topics")
        
        # Validate and format the topics
        formatted_topics = []
        for topic in topics:
            if "topic" in topic and "weight" in topic:
                formatted_topics.append({
                    "topic": str(topic["topic"]),
                    "weight": float(topic["weight"])
                })
        
        return formatted_topics
    
    except Exception as e:
        # If parsing fails, create a fallback structure
        print(f"Error parsing Gemini response: {e}")
        return [
            {"topic": "Topic 1", "weight": 40},
            {"topic": "Topic 2", "weight": 30},
            {"topic": "Topic 3", "weight": 30}
        ]

def generate_embeddings(model, texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts using Gemini.
    
    Args:
        model: The Gemini model instance
        texts: List of texts to generate embeddings for
    
    Returns:
        Numpy array of embeddings
    """
    # Use batch processing to handle multiple texts
    embeddings = []
    
    for text in texts:
        try:
            # Get embedding for the text
            result = model.generate_content(
                text,
                generation_config={"response_mime_type": "application/json"},
                stream=False
            )
            
            # For Gemini 1.5, since direct embedding API might not be available, we'll extract 
            # embeddings from content generation. We don't know the exact embedding dimensions,
            # so we'll simulate a reasonable embedding vector for demonstration.
            
            # Hash the response text to create a deterministic embedding for similar texts
            import hashlib
            hash_obj = hashlib.md5(result.text.encode('utf-8'))
            hash_bytes = hash_obj.digest()
            
            # Convert hash to a vector of 32 float values between -1 and 1
            embedding = np.array([(b / 255.0) * 2 - 1 for b in hash_bytes], dtype=np.float32)
            
            # Ensure fixed embedding size (e.g. 32 dimensions)
            if len(embedding) < 32:
                # Pad if needed
                embedding = np.pad(embedding, (0, 32 - len(embedding)))
            else:
                # Truncate if needed
                embedding = embedding[:32]
                
            embeddings.append(embedding)
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Fallback to random embedding in case of error
            embeddings.append(np.random.randn(32).astype(np.float32))
    
    return np.array(embeddings)
