from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import logging
import uuid
import os
import shutil # For file operations, if needed, though not for just passing bytes.

from chat_logic import send_message_to_model, start_new_chat, active_chats, logger, last_questions_context, last_answers

# Initialize FastAPI app
app = FastAPI(
    title="Science Helper (Up to Class 10)",
    description="Ask questions about Physics, Chemistry, or Biology (text or image)",
    version="0.1.0"
)

# Mount static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
    logger.info(f"Created static directory at {static_dir}")

try:
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    logger.info(f"Mounted static directory from {static_dir}")
except RuntimeError as e:
    logger.warning(f"Could not mount static directory: {e}. Ensure 'static' directory exists at {static_dir}.")


# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root(request: Request):
    index_html_path = os.path.join(static_dir, "index.html")
    try:
        with open(index_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        logger.error(f"{index_html_path} not found.")
        raise HTTPException(status_code=404, detail="Frontend interface not found.")
    except Exception as e:
        logger.error(f"Error reading {index_html_path}: {e}")
        raise HTTPException(status_code=500, detail="Server configuration error.")

class AnswerResponse(BaseModel):
    session_id: str
    answer: str

@app.post("/ask", response_model=AnswerResponse)
async def ask_question_endpoint(
    session_id: str | None = Form(None),
    question: str | None = Form(None),
    action: str = Form("ask"), # "ask", "regenerate", "simplify"
    image: UploadFile | None = File(None) # Image file is optional
):
    current_session_id = session_id

    # Basic validation
    if action == "ask" and not question and not image:
        raise HTTPException(status_code=400, detail="Please provide a question or an image.")

    # Session handling
    if not current_session_id:
        if action != "ask":
            raise HTTPException(status_code=400, detail="Session ID is required for regenerate/simplify actions.")
        current_session_id = str(uuid.uuid4())
        try:
            start_new_chat(current_session_id)
            logger.info(f"Started new session: {current_session_id}")
        except Exception as e:
            logger.error(f"Failed to start new chat session {current_session_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Could not initialize chat session: {e}")
    elif current_session_id not in active_chats:
        logger.warning(f"Session ID {current_session_id} provided but not found. Starting new chat for this ID.")
        try:
            start_new_chat(current_session_id)
        except Exception as e:
            logger.error(f"Failed to re-initialize chat session {current_session_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Could not initialize chat session: {e}")

    # Image processing
    image_data_bytes: bytes | None = None
    image_mime_type: str | None = None
    if image:
        # Ensure it's a valid image type if necessary, or let Gemini handle it.
        # Common web image types: image/jpeg, image/png, image/webp, image/gif
        allowed_mime_types = ["image/jpeg", "image/png", "image/webp", "image/gif", "image/heic", "image/heif"]
        if image.content_type not in allowed_mime_types:
            raise HTTPException(status_code=400, detail=f"Unsupported image type: {image.content_type}. Please use JPEG, PNG, WEBP, GIF, HEIC or HEIF.")
        
        image_data_bytes = await image.read()
        image_mime_type = image.content_type
        logger.info(f"Received image: {image.filename}, type: {image_mime_type}, size: {len(image_data_bytes)} bytes")


    # Prepare for chat_logic
    # The `send_message_to_model` will use its internal context for regenerate/simplify
    # The `question` text for regenerate/simplify is more of a trigger than the primary content
    message_for_logic = question
    if action == "regenerate":
        if not last_questions_context.get(current_session_id, {}).get('text') and not last_questions_context.get(current_session_id, {}).get('image_parts'):
             raise HTTPException(status_code=404, detail="No previous question found in this session to regenerate.")
        # `message_for_logic` can be minimal, chat_logic handles reconstruction.
    elif action == "simplify":
        if not last_answers.get(current_session_id):
            raise HTTPException(status_code=404, detail="No previous answer found in this session to simplify.")
        # `message_for_logic` can be minimal.


    try:
        bot_response = send_message_to_model(
            session_id=current_session_id,
            text_message=message_for_logic,
            image_data=image_data_bytes,
            image_mime_type=image_mime_type,
            action=action
        )
        return AnswerResponse(session_id=current_session_id, answer=bot_response)

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unhandled exception in /ask endpoint for session {current_session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")


@app.get("/health", status_code=200)
async def health_check():
    return {"status": "ok", "message": "Science Helper is running!"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server locally for Science Helper...")
    if not os.getenv("GOOGLE_API_KEY"):
        logger.warning("GOOGLE_API_KEY not set in environment. Please create a .env file.")

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    ) 