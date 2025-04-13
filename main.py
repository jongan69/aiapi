from fastapi import FastAPI, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
import asyncio
from g4f.client import Client
import os
import uvicorn

app = FastAPI(
    title="AI API",
    description="API for interacting with various AI models",
    version="1.0.0"
)
client = Client()

# Define all available models
AVAILABLE_MODELS = [
    # OpenAI models
    "gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4o-audio",
    "o1", "o1-mini", "o3-mini",
    
    # Meta models
    "meta-ai", "llama-2-7b", "llama-3-8b", "llama-3-70b", 
    "llama-3.1-8b", "llama-3.1-70b", "llama-3.1-405b",
    "llama-3.2-1b", "llama-3.2-3b", "llama-3.2-11b", "llama-3.2-90b",
    "llama-3.3-70b",
    
    # Mistral models
    "mixtral-8x7b", "mixtral-8x22b", "mistral-nemo", "mixtral-small-24b",
    
    # NousResearch models
    "hermes-3",
    
    # Microsoft models
    "phi-3.5-mini", "phi-4", "wizardlm-2-7b", "wizardlm-2-8x22b",
    
    # Google models
    "gemini", "gemini-exp", "gemini-1.5-flash", "gemini-1.5-pro",
    "gemini-2.0-flash", "gemini-2.0-flash-thinking", "gemini-2.0-flash-thinking-with-apps",
    
    # Anthropic models
    "claude-3-haiku", "claude-3.5-sonnet", "claude-3.7-sonnet",
    
    # Reka AI models
    "reka-core",
    
    # Blackbox AI models
    "blackboxai", "blackboxai-pro",
    
    # CohereForAI models
    "command-r", "command-r-plus", "command-r7b", "command-a",
    
    # GigaChat models
    "GigaChat:latest",
    
    # Qwen models
    "qwen-1.5-7b", "qwen-2-72b", "qwen-2-vl-7b", "qwen-2.5", "qwen-2.5-72b",
    "qwen-2.5-coder-32b", "qwen-2.5-1m", "qwen-2.5-max", "qwq-32b", "qvq-72b",
    
    # Inflection models
    "pi",
    
    # x.ai models
    "grok-3", "grok-3-r1",
    
    # Perplexity AI models
    "sonar", "sonar-pro", "sonar-reasoning", "sonar-reasoning-pro", "r1-1776",
    
    # DeepSeek models
    "deepseek-chat", "deepseek-v3", "deepseek-r1", "janus-pro-7b",
    
    # Nvidia models
    "nemotron-70b",
    
    # Databricks models
    "dbrx-instruct",
    
    # THUDM models
    "glm-4",
    
    # MiniMax models
    "MiniMax",
    
    # 01-ai models
    "yi-34b",
    
    # Cognitive Computations models
    "dolphin-2.6", "dolphin-2.9",
    
    # DeepInfra models
    "airoboros-70b",
    
    # Lizpreciatior models
    "lzlv-70b",
    
    # OpenBMB models
    "minicpm-2.5",
    
    # Ai2 models
    "olmo-1-7b", "olmo-2-13b", "olmo-2-32b", "olmo-4-synthetic",
    "tulu-3-1-8b", "tulu-3-70b", "tulu-3-405b",
    
    # Liquid AI models
    "lfm-40b",
    
    # Uncensored AI models
    "evil",
    
    # Image models
    "sdxl-turbo", "sd-3.5", "flux", "flux-pro", "flux-dev", "flux-schnell",
    "dall-e-3", "midjourney"
]

# Define available image models
IMAGE_MODELS = [
    "sdxl-turbo", "sd-3.5", "flux", "flux-pro", "flux-dev", "flux-schnell",
    "dall-e-3", "midjourney"
]

class AIRequest(BaseModel):
    messages: List[Dict[str, str]] = Field(
        description="List of messages like [{'role': 'system', 'content': '...'}]"
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="The AI model to use for generating responses",
        examples=AVAILABLE_MODELS[:5]  # Show first 5 models as examples
    )
    chunk_size: Optional[int] = Field(
        default=1500,
        description="Size of text chunks when wrap_input is True"
    )
    wrap_input: Optional[bool] = Field(
        default=False,
        description="Whether to split long user messages into chunks"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, how are you?"}
                ],
                "model": "gpt-4o-mini",
                "chunk_size": 1500,
                "wrap_input": False
            }
        }
    }

class ImageRequest(BaseModel):
    prompt: str = Field(
        description="Text description of the image to generate"
    )
    model: str = Field(
        default="flux",
        description="The image generation model to use",
        examples=IMAGE_MODELS[:3]  # Show first 3 image models as examples
    )
    response_format: str = Field(
        default="url",
        description="Format of the response (url or b64_json)",
        examples=["url", "b64_json"]
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "prompt": "a white siamese cat",
                "model": "flux",
                "response_format": "url"
            }
        }
    }

def chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    from textwrap import wrap
    chunks = wrap(text, chunk_size)
    if len(chunks) > 10:
        return chunk_text(text, chunk_size * 2)
    return chunks

async def call_model(messages: List[Dict[str, str]], model: str) -> str:
    response = await asyncio.to_thread(
        client.chat.completions.create,
        model=model,
        messages=messages,
        web_search=False
    )
    return response.choices[0].message.content

@app.post("/chat/", 
    summary="Chat with an AI model",
    description="Send messages to an AI model and get a response. Supports various models including GPT, Claude, Gemini, and many others."
)
async def chat(ai_request: AIRequest):
    messages = ai_request.messages
    model = ai_request.model

    if ai_request.wrap_input:
        # If user provides a long 'user' message, we split it and process in chunks
        user_msgs = [msg["content"] for msg in messages if msg["role"] == "user"]
        base_messages = [msg for msg in messages if msg["role"] != "user"]
        
        all_chunks = []
        for user_msg in user_msgs:
            chunks = chunk_text(user_msg, ai_request.chunk_size)
            summaries = await asyncio.gather(*[
                call_model(base_messages + [{"role": "user", "content": chunk}], model)
                for chunk in chunks
            ])
            combined = "\n\n".join(summaries)
            final_response = await call_model(base_messages + [{"role": "user", "content": combined}], model)
            all_chunks.append(final_response)
        
        return {"response": "\n\n".join(all_chunks)}
    
    # Otherwise just one-pass response
    final_response = await call_model(messages, model)
    return {"response": final_response}

@app.post("/images/generate/",
    summary="Generate an image from a text prompt",
    description="Generate an image based on a text description using various AI image generation models."
)
async def generate_image(image_request: ImageRequest):
    prompt = image_request.prompt
    model = image_request.model
    response_format = image_request.response_format
    
    # Call the image generation API
    response = await asyncio.to_thread(
        client.images.generate,
        model=model,
        prompt=prompt,
        response_format=response_format
    )
    
    # Return the image URL or base64 data
    if response_format == "url":
        return {"url": response.data[0].url}
    else:
        return {"b64_json": response.data[0].b64_json}

@app.get("/models/", 
    summary="List all available models",
    description="Returns a list of all available AI models that can be used with the chat endpoint"
)
async def list_models():
    return {"models": AVAILABLE_MODELS}

@app.get("/models/image/", 
    summary="List all available image models",
    description="Returns a list of all available AI image generation models"
)
async def list_image_models():
    return {"models": IMAGE_MODELS}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)
