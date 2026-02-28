from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import re
import json
import time
import subprocess
import tempfile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    video_url: str
    topic: str


def seconds_to_hhmmss(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def normalize_timestamp(ts: str) -> str:
    parts = ts.strip().split(":")
    if len(parts) == 2:
        return f"00:{parts[0].zfill(2)}:{parts[1].zfill(2)}"
    elif len(parts) == 3:
        return f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)}"
    return "00:00:00"


def try_transcript(video_url: str, topic: str):
    """Try YouTube transcript API first (fast path)."""
    from youtube_transcript_api import YouTubeTranscriptApi
    match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})', video_url)
    if not match:
        raise ValueError("Bad URL")
    video_id = match.group(1)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    transcript_text = "\n".join([
        f"[{seconds_to_hhmmss(e['start'])}] {e['text']}"
        for e in transcript
    ])
    return transcript_text


def find_with_gemini_text(transcript_text: str, topic: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    prompt = f"""Find the FIRST moment where this topic is spoken: "{topic}"

TRANSCRIPT:
{transcript_text[:50000]}

Return the timestamp in HH:MM:SS format."""

    response = client.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={"timestamp": types.Schema(type=types.Type.STRING)},
                required=["timestamp"]
            )
        )
    )
    result = json.loads(response.text)
    return normalize_timestamp(result.get("timestamp", "00:00:00"))


def find_with_gemini_audio(audio_path: str, topic: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # Upload audio file
    with open(audio_path, "rb") as f:
        uploaded = client.files.upload(
            file=f,
            config={"mime_type": "audio/mpeg", "display_name": "audio"}
        )

    # Poll until ACTIVE
    for _ in range(20):
        file_info = client.files.get(name=uploaded.name)
        if file_info.state.name == "ACTIVE":
            break
        time.sleep(3)

    prompt = f"""Listen to this audio and find the FIRST moment where this topic is spoken or discussed: "{topic}"
Return the timestamp in HH:MM:SS format when this topic first appears."""

    response = client.models.generate_content(
        model='gemini-2.0-flash-exp',
        contents=[
            types.Part.from_uri(file_uri=uploaded.uri, mime_type="audio/mpeg"),
            prompt
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={"timestamp": types.Schema(type=types.Type.STRING)},
                required=["timestamp"]
            )
        )
    )

    # Cleanup
    try:
        client.files.delete(name=uploaded.name)
    except Exception:
        pass

    result = json.loads(response.text)
    return normalize_timestamp(result.get("timestamp", "00:00:00"))


@app.post("/ask")
async def ask(request: AskRequest):
    if not request.video_url or not request.topic:
        raise HTTPException(status_code=422, detail="video_url and topic are required")

    # Try transcript first (fast)
    try:
        transcript_text = try_transcript(request.video_url, request.topic)
        timestamp = find_with_gemini_text(transcript_text, request.topic)
        return JSONResponse(content={
            "timestamp": timestamp,
            "video_url": request.video_url,
            "topic": request.topic
        })
    except Exception:
        pass

    # Fallback: download audio with yt-dlp
    audio_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            audio_path = tmp.name

        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "mp3",
            "--audio-quality", "3",
            "--no-playlist",
            "-o", audio_path,
            request.video_url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise HTTPException(status_code=400, detail=f"yt-dlp failed: {result.stderr}")

        timestamp = find_with_gemini_audio(audio_path, request.topic)
        return JSONResponse(content={
            "timestamp": timestamp,
            "video_url": request.video_url,
            "topic": request.topic
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception:
                pass


@app.get("/health")
async def health():
    return {"status": "ok"}
