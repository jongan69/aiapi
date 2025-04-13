from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
from g4f.client import Client
import os
import uvicorn

app = FastAPI()
client = Client()

class AIRequest(BaseModel):
    messages: List[Dict[str, str]]  # List of messages like [{"role": "system", "content": "..."}]
    model: Optional[str] = "gpt-4o-mini"
    chunk_size: Optional[int] = 1500
    wrap_input: Optional[bool] = False

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

@app.post("/chat/")
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

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Default to 8000 if PORT is not set
    uvicorn.run(app, host="0.0.0.0", port=port)
