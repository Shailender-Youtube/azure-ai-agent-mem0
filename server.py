import os
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import CookingAssistantWithMemory, project


app = FastAPI(title="Cooking Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend under /static and route / to index.html
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.isdir(static_dir):
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def root():
    index_path = os.path.join(static_dir, "index.html")
    return FileResponse(index_path)


assistant = CookingAssistantWithMemory()

# In-memory mapping: user_id -> thread
user_threads: Dict[str, object] = {}


class StartSessionRequest(BaseModel):
    user_id: str


class ChatRequest(BaseModel):
    user_id: str
    message: str


@app.post("/api/start_session")
def start_session(payload: StartSessionRequest):
    user_id = payload.user_id.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")

    # Create or reuse a thread per user for this server lifetime
    thread = user_threads.get(user_id)
    if thread is None:
        thread = project.agents.threads.create()
        user_threads[user_id] = thread

    # Build greeting using merged profile completeness
    structured = assistant._get_profile_status(user_id=user_id)
    inferred, inferred_conf = assistant._infer_profile_from_plain_text(user_id=user_id)
    merged_profile = assistant._merge_structured_and_inferred(structured, inferred, inferred_conf)
    merged_complete = all(f in merged_profile and merged_profile[f] for f in assistant.required_profile_fields)

    # Persist inferred critical fields if missing from structured profile (self-healing)
    # This guarantees future sessions don't re-ask despite prior free-text answers
    try:
        for key in ("skill_level", "dietary_preferences"):
            if key not in structured and key in inferred and inferred_conf.get(key, 0.0) >= 0.7:
                assistant.memory.add(f"PROFILE.{key}: {inferred[key]}", user_id=user_id)
                structured[key] = inferred[key]
        # Recompute merged after self-heal
        merged_profile = assistant._merge_structured_and_inferred(structured, inferred, inferred_conf)
        merged_complete = all(f in merged_profile and merged_profile[f] for f in assistant.required_profile_fields)
    except Exception:
        pass

    # If minimal fields exist, skip onboarding and go straight to recipe flow
    minimal_ready = all(f in merged_profile and merged_profile[f] for f in assistant.min_profile_fields)

    if merged_complete or minimal_ready:
        profile_bits = []
        for key in assistant.required_profile_fields:
            if key in merged_profile:
                profile_bits.append(f"{key.replace('_', ' ')}: {merged_profile[key]}")
        profile_summary = "; ".join(profile_bits)
        if minimal_ready and not merged_complete:
            greeting = (
                f"Welcome back, {user_id}! I have enough info ({profile_summary}) to suggest recipes. "
                f"Share ingredients if you like, or I can suggest something now."
            )
        else:
            greeting = (
                f"Welcome back, {user_id}! I have your profile ({profile_summary}). "
                f"Are you looking for a recipe now? Share ingredients or I'll suggest one."
            )
    else:
        next_field = assistant._first_missing_with_priority(merged_profile)
        have_bits = []
        for key in assistant.required_profile_fields:
            if key in merged_profile:
                have_bits.append(f"{key.replace('_', ' ')}: {merged_profile[key]}")
        have_summary = "; ".join(have_bits)
        if not have_summary:
            # Fall back to count of general memories so we don't show "none yet" when we do know things
            mems = assistant.get_all_memories(user_id=user_id)
            have_summary = f"I remember {len(mems)} things about your preferences" if mems else "none yet"
        greeting = (
            f"Welcome back, {user_id}! We'll complete your profile quickly. So far -> {have_summary}. "
            f"Please share your {next_field.replace('_',' ')}."
        )

    return {"thread_id": thread.id, "message": greeting}


@app.post("/api/chat")
def chat(payload: ChatRequest):
    user_id = payload.user_id.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    if payload.message is None:
        raise HTTPException(status_code=400, detail="message is required")

    thread = user_threads.get(user_id)
    if thread is None:
        thread = project.agents.threads.create()
        user_threads[user_id] = thread

    response = assistant.chat_with_memory(payload.message, user_id=user_id, thread=thread)
    if response is None:
        raise HTTPException(status_code=500, detail="Agent returned no response")
    return {"response": response}


@app.get("/api/memories")
def list_memories(user_id: str):
    items = assistant.get_all_memories(user_id=user_id)
    return {"count": len(items), "items": items}


# To run: uvicorn server:app --reload --port 8000

