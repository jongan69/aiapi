from fastapi import FastAPI, UploadFile, File, Form, Request, Body
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
from g4f.client import Client, AsyncClient
from g4f.Provider import RetryProvider, OpenaiChat, Free2GPT, FreeGpt, LambdaChat, PollinationsAI, PollinationsImage, ImageLabs, HuggingFaceMedia
import os
import uvicorn
import json
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi import Request
import base64
import io
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
import g4f.debug

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

g4f.debug.logging = True
g4f.debug.version_check = False

text_providers = [OpenaiChat, Free2GPT, FreeGpt, LambdaChat]
image_providers = [PollinationsAI, PollinationsImage, ImageLabs]

client = Client(
    provider=RetryProvider(text_providers, shuffle=True),
    image_provider=RetryProvider(image_providers, shuffle=True)
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

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
    model: str = Field(default="midjourney")
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

def get_available_models(providers):
    models = []
    for provider in providers:
        try:
            models += provider.get_models()
        except Exception as e:
            print(f"{provider.__name__} error: {e}")
    return list(set(models))

def get_available_image_models(providers):
    image_models = []
    for provider in providers:
        try:
            image_models += provider.get_models()
        except Exception as e:
            print(f"{provider.__name__} error: {e}")
    return list(set(image_models))

@app.post("/chat/")
async def chat(ai_request: AIRequest):
    messages = ai_request.messages
    model = ai_request.model
    json_mode = ai_request.json_mode
    stream = ai_request.stream

    # Validate model using all text providers
    available_models = get_available_models(text_providers)
    if model not in available_models:
        return {"error": f"Model '{model}' is not available. Please choose from: {available_models}"}

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

    # Validate image model
    available_image_models = get_available_image_models(image_providers)
    if model not in available_image_models:
        return {"error": f"Image model '{model}' is not available. Please choose from: {available_image_models}"}

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
    # Validate image model
    available_image_models = get_available_image_models(image_providers)
    if model not in available_image_models:
        return {"error": f"Image model '{model}' is not available. Please choose from: {available_image_models}"}
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

@app.post("/images/variations/async/")
async def create_image_variation_async(
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(None),
    model: str = Form("dall-e-3"),
    response_format: str = Form("url"),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    n: Optional[int] = Form(1)
):
    available_image_models = get_available_image_models(image_providers)
    if model not in available_image_models:
        return {"error": f"Image model '{model}' is not available. Please choose from: {available_image_models}"}
    try:
        image_data = await file.read()
        image_file = io.BytesIO(image_data)
        image_file.name = file.filename
        image_client = AsyncClient(image_provider=client.image_provider)
        kwargs = {"prompt": prompt, "image": image_file, "model": model, "response_format": response_format}
        if width: kwargs["width"] = width
        if height: kwargs["height"] = height
        if n: kwargs["n"] = n
        response = await image_client.images.create_variation(**kwargs)
        if response_format == "url":
            return {"urls": [img.url for img in response.data]}
        else:
            return {"b64_jsons": [img.b64_json for img in response.data]}
    except Exception as e:
        return {"error": str(e)}

@app.get("/models/audio/")
async def list_audio_models():
    audio_models = []
    # Add more providers if you have them
    for provider in [PollinationsAI]:
        try:
            if hasattr(provider, "audio_models") and provider.audio_models:
                audio_models += provider.audio_models
            elif hasattr(provider, "get_models"):
                audio_models += [m for m in provider.get_models() if "audio" in m]
        except Exception as e:
            print(f"{provider.__name__} error: {e}")
    audio_models = list(set(audio_models))
    return {"models": audio_models}

@app.post("/audio/generate/")
async def generate_audio(
    text: str = Body(...),
    model: str = Body(...),
    voice: str = Body("alloy"),
    format: str = Body("mp3")
):
    available_audio_models = (await list_audio_models())["models"]
    if model not in available_audio_models:
        return {"error": f"Audio model '{model}' is not available. Please choose from: {available_audio_models}"}
    try:
        audio_client = AsyncClient(provider=client.provider)
        response = await audio_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": text}],
            audio={"voice": voice, "format": format},
        )
        audio_bytes = response.choices[0].message.content
        return {"audio": audio_bytes}
    except Exception as e:
        return {"error": str(e)}

@app.post("/audio/transcribe/")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form(...)
):
    available_audio_models = (await list_audio_models())["models"]
    if model not in available_audio_models:
        return {"error": f"Audio model '{model}' is not available. Please choose from: {available_audio_models}"}
    try:
        audio_data = await file.read()
        audio_file = io.BytesIO(audio_data)
        audio_file.name = file.filename
        audio_client = AsyncClient(provider=client.provider)
        response = await audio_client.chat.completions.create(
            messages=[{"role": "user", "content": "Transcribe this audio"}],
            media=[[audio_file, file.filename]],
            modalities=["text"],
            model=model
        )
        return {"transcription": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}

@app.post("/video/generate/")
async def generate_video(
    prompt: str = Body(...),
    model: str = Body(...),
    resolution: str = Body("720p"),
    aspect_ratio: str = Body("16:9"),
    n: int = Body(1),
    response_format: str = Body("url")
):
    available_video_models = (await list_video_models())["models"]
    if model not in available_video_models:
        return {"error": f"Video model '{model}' is not available. Please choose from: {available_video_models}"}
    try:
        video_client = AsyncClient(
            provider=HuggingFaceMedia,
            api_key=os.getenv("HF_TOKEN")
        )
        result = await video_client.media.generate(
            model=model,
            prompt=prompt,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            n=n,
            response_format=response_format,
        )
        return {"urls": [video.url for video in result.data]}
    except Exception as e:
        return {"error": str(e)}

@app.get("/models/")
async def list_models():
    models = get_available_models(text_providers)
    return {"models": models}

@app.get("/models/image/")
async def list_image_models():
    image_models = get_available_image_models(image_providers)
    return {"models": image_models}

@app.get("/models/video/")
async def list_video_models():
    try:
        video_client = AsyncClient(
            provider=HuggingFaceMedia,
            api_key=os.getenv("HF_TOKEN")
        )
        video_models = video_client.models.get_video()
        return {"models": video_models}
    except Exception as e:
        return {"error": str(e)}

@app.get("/models/image/variation/")
async def list_image_variation_models():
    """
    Returns all available image models that can be used for image variation.
    Note: Not all models may actually support variation, but this is the full list from your providers.
    """
    image_models = get_available_image_models(image_providers)
    return {"models": image_models}

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
