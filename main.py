from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware # For potential frontend dev on diff port
from pydantic import BaseModel
import logging
import uuid
import os
import shutil # For saving/handling uploaded files if needed temporarily

from chat_logic import generate_answer, logger

# Define the response body structure
class AnswerResponse(BaseModel):
    session_id: str # Or question_id if preferred for stateless
    answer: str

# Initialize FastAPI app
app = FastAPI(
    title="Science Helper App",
    description="Ask science questions (Physics, Chemistry, Biology) for Class 10 and below. Supports image uploads.",
    version="0.1.0"
)

# CORS Middleware (optional, useful for local development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Or specify your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    """Serves the HTML interface."""
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


@app.post("/ask-science", response_model=AnswerResponse)
async def ask_science_question(
    request: Request, # Keep request for potential future use (e.g. headers, client info)
    session_id: str = Form(None), # Session ID can be optional, generate if not present
    text_question: str = Form(None), # Text part of the question
    image_question: UploadFile = File(None) # Image part of the question
):
    """
    Receives a science question (text and/or image) and returns an answer.
    """
    current_session_id = session_id if session_id else str(uuid.uuid4())
    logger.info(f"Received request for session: {current_session_id}. Text: '{text_question}'. Image: {image_question.filename if image_question else 'No'}")

    if not text_question and not image_question:
        raise HTTPException(status_code=400, detail="Please provide a question (text or image).")

    image_data_bytes: bytes | None = None
    image_mime_type: str | None = None

    if image_question:
        if image_question.content_type not in ["image/jpeg", "image/png", "image/webp", "image/heic", "image/heif"]:
            raise HTTPException(status_code=400, detail="Invalid image type. Please upload JPEG, PNG, WEBP, HEIC, or HEIF.")
        try:
            image_data_bytes = await image_question.read() # Read image data into bytes
            image_mime_type = image_question.content_type
            logger.info(f"Image received: {image_question.filename}, type: {image_mime_type}, size: {len(image_data_bytes)} bytes")
        except Exception as e:
            logger.error(f"Error reading uploaded image: {e}")
            raise HTTPException(status_code=500, detail="Could not process uploaded image.")
        finally:
            await image_question.close() # Important to close the file handle

    try:
        # Call the logic to get answer from Gemini
        bot_response = generate_answer(
            session_id=current_session_id,
            text_prompt=text_question,
            image_data=image_data_bytes,
            image_mime_type=image_mime_type
        )
        return AnswerResponse(session_id=current_session_id, answer=bot_response)

    except HTTPException as e:
        # Re-raise HTTP exceptions from chat_logic or this handler
        raise e
    except Exception as e:
        logger.error(f"Unhandled exception in /ask-science endpoint for session {current_session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")


@app.get("/health", status_code=200)
async def health_check():
    """Simple health check endpoint."""
    return {"status": "Science Helper is A-OK!"}

# --- Main execution block (for running locally with uvicorn) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn server locally for Science Helper App...")
    if not os.getenv("GOOGLE_API_KEY"):
        logger.warning("GOOGLE_API_KEY not set in environment. Please create a .env file.")

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    ) 