from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import uuid
import os
import shutil
from pathlib import Path
import sys
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from chat_logic import generate_answer
except Exception as e:
    logger.error(f"Failed to import chat_logic: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# Define the response body structure
class AnswerResponse(BaseModel):
    session_id: str
    answer: str

# Initialize FastAPI app
app = FastAPI(
    title="Science Helper App",
    description="Ask science questions (Physics, Chemistry, Biology) for Class 10 and below. Supports image uploads.",
    version="0.1.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the current directory
static_dir = current_dir / "static"
if not static_dir.exists():
    static_dir.mkdir(parents=True)
    logger.info(f"Created static directory at {static_dir}")

try:
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info(f"Mounted static directory from {static_dir}")
except Exception as e:
    logger.error(f"Failed to mount static directory: {str(e)}")
    logger.error(traceback.format_exc())

# --- API Endpoints ---

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def read_root(request: Request):
    """Serves the HTML interface."""
    try:
        index_html_path = static_dir / "index.html"
        if not index_html_path.exists():
            logger.error(f"index.html not found at {index_html_path}")
            raise HTTPException(status_code=404, detail="Frontend interface not found.")
        
        with open(index_html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error serving frontend: {str(e)}")

@app.post("/ask-science", response_model=AnswerResponse)
async def ask_science_question(
    request: Request,
    session_id: str = Form(None),
    text_question: str = Form(None),
    image_question: UploadFile = File(None)
):
    """
    Receives a science question (text and/or image) and returns an answer.
    """
    try:
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
                image_data_bytes = await image_question.read()
                image_mime_type = image_question.content_type
                logger.info(f"Image received: {image_question.filename}, type: {image_mime_type}, size: {len(image_data_bytes)} bytes")
            except Exception as e:
                logger.error(f"Error reading uploaded image: {str(e)}")
                logger.error(traceback.format_exc())
                raise HTTPException(status_code=500, detail="Could not process uploaded image.")
            finally:
                await image_question.close()

        try:
            bot_response = generate_answer(
                session_id=current_session_id,
                text_prompt=text_question,
                image_data=image_data_bytes,
                image_mime_type=image_mime_type
            )
            return AnswerResponse(session_id=current_session_id, answer=bot_response)
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unhandled exception in /ask-science endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

@app.get("/health", status_code=200)
async def health_check():
    """Simple health check endpoint."""
    try:
        return {"status": "Science Helper is A-OK!"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# For local development
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