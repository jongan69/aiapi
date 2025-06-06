from fastapi import FastAPI, Body, UploadFile, File, Form, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
import asyncio
from g4f.client import Client
from g4f.Provider import OpenaiChat
import os
import uvicorn
import json
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi import Request
import base64
import io
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AI API",
    description="""
    A powerful API for interacting with various AI models.
    
    Features:
    - Chat with AI models using streaming responses
    - Generate images with various models
    - Create image variations
    - Support for JSON mode responses
    
    All endpoints are fully documented with examples and detailed descriptions.
    """,
    version="1.0.0",
    docs_url=None,  # Disable default docs
    redoc_url=None  # Disable redoc
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Client()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Define all available models
AVAILABLE_MODELS = [
    # OpenAI models
    "gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4o-audio", "o1", "o1-mini", "o3-mini",
    
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
    "gemini-2.0", "gemini-exp", "gemini-1.5-pro", "gemini-1.5-flash",
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
    "qwen-2.5-coder-32b", "qwen-2.5-1m", "qwen-2-5-max", "qwq-32b", "qvq-72b",
    
    # Inflection models
    "pi",
    
    # x.ai models
    "grok-3",
    
    # Perplexity AI models
    "sonar", "sonar-pro", "sonar-reasoning", "sonar-reasoning-pro", "r1-1776",
    
    # DeepSeek models
    "deepseek-chat", "deepseek-v3", "deepseek-r1",
    
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
    "evil"
]

# Define available image models
IMAGE_MODELS = [
    "sdxl-turbo", "sd-3.5", "flux", "flux-pro", "flux-dev", "flux-schnell",
    "dall-e-3", "midjourney"
]

class AIRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: str = Field(default="gpt-4o-mini")
    chunk_size: Optional[int] = Field(default=1500)
    wrap_input: Optional[bool] = Field(default=False)
    json_mode: Optional[bool] = Field(
        default=False,
        description="Force model to respond strictly in JSON format."
    )
    stream: Optional[bool] = Field(
        default=False,
        description="Stream the response chunks."
    )

class ImageRequest(BaseModel):
    prompt: str
    model: str = Field(default="sdxl-turbo")
    response_format: str = Field(default="url")

class ImageVariationRequest(BaseModel):
    model: str = Field(default="dall-e-3")
    response_format: str = Field(default="url")

def chunk_text(text: str, chunk_size: int = 1500) -> List[str]:
    from textwrap import wrap
    chunks = wrap(text, chunk_size)
    if len(chunks) > 10:
        return chunk_text(text, chunk_size * 2)
    return chunks

async def call_model(messages: List[Dict[str, str]], model: str, json_mode: bool = False, max_retries: int = 3) -> Any:
    if json_mode:
        messages = [{"role": "system", "content": "Respond only with valid minified JSON. No extra text or explanations."}] + messages

    for attempt in range(max_retries):
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=model,
                messages=messages,
                web_search=False
            )

            content = response.choices[0].message.content

            if json_mode:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    if attempt < max_retries - 1:
                        continue  # Retry
                    else:
                        return {
                            "error": f"Model did not return valid JSON after {max_retries} attempts.",
                            "raw_response": content
                        }
            else:
                return content  # If no json_mode, just return plain response
        except Exception as e:
            if attempt < max_retries - 1:
                # Log the error but continue to retry
                print(f"Attempt {attempt+1} failed: {str(e)}")
                continue
            else:
                # On the last attempt, raise the exception
                raise e

@app.post("/chat/")
async def chat(ai_request: AIRequest):
    messages = ai_request.messages
    model = ai_request.model
    json_mode = ai_request.json_mode
    stream = ai_request.stream

    try:
        if stream:
            async def generate_stream():
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                    web_search=False
                )
                
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        # For streaming, we don't validate JSON - just pass through the content
                        yield f"data: {json.dumps({'content': content})}\n\n"
                
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream"
            )

        if ai_request.wrap_input:
            user_msgs = [msg["content"] for msg in messages if msg["role"] == "user"]
            base_messages = [msg for msg in messages if msg["role"] != "user"]
            
            all_chunks = []
            for user_msg in user_msgs:
                chunks = chunk_text(user_msg, ai_request.chunk_size)
                summaries = await asyncio.gather(*[
                    call_model(base_messages + [{"role": "user", "content": chunk}], model, json_mode)
                    for chunk in chunks
                ])
                combined = "\n\n".join([summary if isinstance(summary, str) else json.dumps(summary) for summary in summaries])
                final_response = await call_model(base_messages + [{"role": "user", "content": combined}], model, json_mode)
                all_chunks.append(final_response)
            
            return {"response": all_chunks}
        
        # For non-streaming requests, we use call_model which handles JSON validation
        final_response = await call_model(messages, model, json_mode)
        return {"response": final_response}
    except Exception as e:
        error_message = str(e)
        if "ProviderNotFoundError" in error_message:
            return {"error": f"Model '{model}' is not supported. Please try a different model."}
        else:
            return {"error": f"Error processing request: {error_message}"}

@app.post("/images/generate/")
async def generate_image(image_request: ImageRequest):
    prompt = image_request.prompt
    model = image_request.model
    response_format = image_request.response_format
    
    try:
        response = await asyncio.to_thread(
            client.images.generate,
            model=model,
            prompt=prompt,
            response_format=response_format
        )
        
        if response_format == "url":
            return {"url": response.data[0].url}
        else:
            return {"b64_json": response.data[0].b64_json}
    except Exception as e:
        error_message = str(e)
        if "MissingAuthError" in error_message:
            return {"error": f"Authentication required for model '{model}'. Please try a different model or configure authentication."}
        elif "ProviderNotFoundError" in error_message:
            return {"error": f"Model '{model}' is not supported. Please try a different model."}
        else:
            return {"error": f"Error generating image: {error_message}"}

@app.post("/images/variations/")
async def create_image_variation(
    file: UploadFile = File(...),
    model: str = Form("sdxl-turbo"),
    response_format: str = Form("url")
):
    try:
        # Read the uploaded file
        image_data = await file.read()
        
        # Create a client
        image_client = Client()
        
        # Create a BytesIO object from the image data
        image_io = io.BytesIO(image_data)
        
        # For sdxl-turbo, we'll use the generate endpoint with the image as a prompt
        # Convert the image to base64 for the prompt
        base64_image = base64.b64encode(image_data).decode('utf-8')
        prompt = f"Create a variation of this image: data:image/jpeg;base64,{base64_image}"
        
        # Generate image variations
        response = await asyncio.to_thread(
            image_client.images.generate,
            prompt=prompt,
            model=model,
            response_format=response_format
        )
        
        if response_format == "url":
            return {"url": response.data[0].url}
        else:
            return {"b64_json": response.data[0].b64_json}
    except Exception as e:
        error_message = str(e)
        if "MissingAuthError" in error_message:
            return {"error": f"Authentication required for model '{model}'. Please try a different model or configure authentication."}
        elif "ProviderNotFoundError" in error_message:
            return {"error": f"Model '{model}' is not supported. Please try a different model."}
        else:
            return {"error": f"Error generating image variation: {error_message}"}

@app.get("/models/")
async def list_models():
    return {"models": AVAILABLE_MODELS}

@app.get("/models/image/")
async def list_image_models():
    return {"models": IMAGE_MODELS}

@app.get("/openapi.json")
async def get_openapi_endpoint():
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

# Custom Swagger UI route
@app.get("/", response_class=HTMLResponse)
async def custom_swagger_ui(request: Request):
    return templates.TemplateResponse("swagger-ui.html", {"request": request})

@app.get("/healthz")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
