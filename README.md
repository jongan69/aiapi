# AI API

A powerful API for interacting with various AI models, built with FastAPI.

## Overview

This API provides endpoints for:
- Chatting with AI models using streaming responses
- Generating images with various models
- Creating image variations
- Generating audio (including elevator pitches)
- Transcribing audio
- Generating videos

All endpoints are fully documented with examples and detailed descriptions.

## Features

- **Chat**: Interact with AI models using streaming responses
- **Image Generation**: Generate images with various models
- **Image Variations**: Create variations of existing images
- **Audio Generation**: Generate audio from text, including elevator pitches
- **Audio Transcription**: Transcribe audio files
- **Video Generation**: Generate videos based on prompts
- **CORS Support**: Configured to allow cross-origin requests

## Setup

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jongan69/aiapi.git
   cd aiapi
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory with the following variables:
   ```
   PROXY_URL=your_proxy_url
   PORT=8000
   HF_TOKEN=your_huggingface_token
   ```

4. Run the server:
   ```bash
   python main.py
   ```

The server will start at `http://localhost:8000`.

## API Documentation

### Chat

**Endpoint:** `POST /chat/`

**Request Body:**
```json
{
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "model": "auto",
  "chunk_size": 1500,
  "wrap_input": false,
  "json_mode": false,
  "stream": false
}
```

**Response:**
```json
{
  "response": "I'm doing well, thank you for asking!"
}
```

### Image Generation

**Endpoint:** `POST /images/generate/`

**Request Body:**
```json
{
  "prompt": "A beautiful sunset over the ocean",
  "model": "midjourney",
  "response_format": "url"
}
```

**Response:**
```json
{
  "url": "https://example.com/image.jpg"
}
```

### Image Variations

**Endpoint:** `POST /images/variations/`

**Request Body:**
```
file: [binary file]
model: sdxl-turbo
response_format: url
```

**Response:**
```json
{
  "url": "https://example.com/variation.jpg"
}
```

### Audio Generation

**Endpoint:** `POST /audio/generate/`

**Request Body:**
```json
{
  "text": "Hello, this is a test.",
  "model": "gpt-4o-mini-tts",
  "voice": "alloy",
  "format": "mp3",
  "provider": "OpenAIFM"
}
```

**Response:**
```
[Binary audio file]
```

### Audio Elevator Pitch

**Endpoint:** `POST /audio/elevator_pitch/`

**Request Body:**
```json
{
  "prompt": "My new AI startup",
  "model": "gpt-4o-mini",
  "audio_model": "openai-audio",
  "voice": "alloy",
  "format": "mp3",
  "provider": "PollinationsAI"
}
```

**Response:**
```
[Binary audio file]
```

### Audio Transcription

**Endpoint:** `POST /audio/transcribe/`

**Request Body:**
```
file: [binary file]
model: whisper
```

**Response:**
```json
{
  "transcription": "This is the transcribed text."
}
```

### Video Generation

**Endpoint:** `POST /video/generate/`

**Request Body:**
```json
{
  "prompt": "A beautiful sunset over the ocean",
  "model": "stable-diffusion",
  "resolution": "720p",
  "aspect_ratio": "16:9",
  "n": 1,
  "response_format": "url"
}
```

**Response:**
```json
{
  "urls": ["https://example.com/video.mp4"]
}
```

## Available Models

### Text Models

**Endpoint:** `GET /models/`

**Response:**
```json
{
  "models": ["gpt-4", "gpt-3.5-turbo", "claude-2", "palm-2"]
}
```

### Image Models

**Endpoint:** `GET /models/image/`

**Response:**
```json
{
  "models": ["midjourney", "dall-e-3", "stable-diffusion"]
}
```

### Video Models

**Endpoint:** `GET /models/video/`

**Response:**
```json
{
  "models": ["stable-diffusion", "runway"]
}
```

### Audio Models

**Endpoint:** `GET /models/audio/`

**Response:**
```json
{
  "models": ["gpt-4o-mini-tts", "openai-audio", "hypnosis-tracy"]
}
```

### Audio Voices

**Endpoint:** `GET /models/audio/voices/`

**Response:**
```json
{
  "voices": ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"]
}
```

## CORS Support

The API is configured to allow cross-origin requests. CORS headers are set on all responses, including file and streaming responses.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
