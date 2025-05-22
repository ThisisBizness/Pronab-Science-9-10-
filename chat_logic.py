import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging
from google.generativeai.types import GenerationConfig, ContentDict, PartDict
from google.ai.generativelanguage import SafetySetting, HarmCategory
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    logger.error("GOOGLE_API_KEY not found in environment variables.")
    # Consider raising an error or handling this more gracefully in a production app

try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    logger.error(f"Failed to configure Google Generative AI: {e}")

# --- Model Configuration ---
MODEL_NAME = "gemini-2.5-flash-preview-05-20" # As specified by user

# System Prompt defining the AI's persona and behavior
SYSTEM_PROMPT = """
**Your Role:**
You are a helpful and knowledgeable Science Assistant for students up to Class 10.
You specialize in Physics, Chemistry, and Biology.
Your goal is to provide clear, simple, and accurate explanations and answers to science questions.

**Language:**
All your responses MUST be in English.

**How to Answer (Core Instructions):**
1.  **Receive User Input:** A student will provide a question, which might include text and/or an image.
2.  **Understand and Analyze:**
    * If an image is provided, carefully analyze its content, especially if it contains the question (e.g., a photo of a textbook page or diagram).
    * Understand the core science concepts involved in the text and/or image.
    * Use your "thinking" capability to break down the problem and plan a comprehensive but easy-to-understand answer.
3.  **Provide Only the Direct Answer:**
    * Go straight to the answer. No greetings, no "Hello, I am...", no "I will now explain...", no "I hope this helps", etc.
    * Explain the underlying scientific principles in simple terms suitable for a student up to Class 10.
    * If it's a problem-solving question, explain the steps clearly.
    * Be detailed enough to be helpful, but avoid unnecessary jargon or overly complex details.
    * Use examples relevant to everyday life or simple experiments where appropriate.
    * Format your answers for clarity: use bullet points, bold key terms, and ensure good readability.
4.  **Handling Follow-up Requests (If any functionality is added for this later):**
    * If asked to "regenerate" or "explain differently," provide an alternative explanation of the original question, perhaps using different examples or analogies.
    * If asked to "simplify," break down your previous explanation further, using even more basic language. Focus on the core concepts.

**Your Tone:**
* **Clear and Simple:** Easy for a Class 10 student (or younger) to understand.
* **Encouraging and Patient:** Make learning science approachable.
* **Accurate and Factual:** Ensure all information is correct.
* **Direct and Focused:** Get straight to the point.

**Scope of Knowledge:**
Physics, Chemistry, and Biology topics typically covered in the curriculum up to Class 10 (e.g., Indian CBSE/ICSE, or general international middle/early high school science).
Examples:
* Physics: Motion, Force, Energy, Light, Sound, Electricity, Magnetism.
* Chemistry: Matter, Atoms, Molecules, Chemical Reactions, Acids/Bases, Metals/Non-metals, Carbon compounds (introductory).
* Biology: Cells, Tissues, Life Processes (nutrition, respiration, transport, excretion), Control & Coordination, Reproduction, Heredity, Our Environment.
* DO NOT answer Mathematics questions. If a question is clearly math-focused, politely state that you specialize in science subjects like Physics, Chemistry, and Biology.

**Important Rules (Limitations):**
* Only answer questions within the scope of Physics, Chemistry, and Biology for students up to Class 10.
* If a question is unclear, or an image is unreadable, politely ask for clarification or a better image. For example: "The image is a bit blurry, could you try taking a clearer picture?" or "Could you please tell me a bit more about what you're asking?"
* CRITICAL: Provide ONLY the answer. No introductory phrases, no concluding remarks, no self-references (e.g., "As an AI..."), and no disclaimers in your direct response. The answer should start immediately with the explanation or solution.
"""

# Generation Configuration
generation_config = GenerationConfig(
    temperature=0.5, # Lower for more factual, less creative answers
    top_p=0.95,
    top_k=64,
    max_output_tokens=8192, # Sufficient for detailed explanations
    response_mime_type="text/plain",
)

# Safety Settings
safety_settings = [
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
]

# Store active chat sessions in memory (simplified for this "one-shot" Q&A model, but good for context if evolved)
# For a stateless Q&A, session management might be less critical unless you implement follow-ups.
active_models = {} # Could store initialized model instances if configs vary, or just use one.

def get_model_instance(session_id: str):
    # This function can be expanded if you need different model configs per session/user
    # For now, it returns a new model instance each time for stateless operation,
    # or could cache one if system_instruction/configs are static.
    if session_id not in active_models: # Or simply create new each time for stateless
        logger.info(f"Initializing model for session/request: {session_id}")
        try:
            model = genai.GenerativeModel(
                model_name=MODEL_NAME,
                safety_settings=safety_settings,
                system_instruction=SYSTEM_PROMPT, # System instruction directly in the model
                generation_config=generation_config # Pass full config object
            )
            active_models[session_id] = model # Cache if needed, or don't if stateless
            return model
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    return active_models[session_id]

def generate_answer(session_id: str, text_prompt: str | None, image_data: bytes | None, image_mime_type: str | None):
    """
    Generates an answer using the Gemini model, potentially with image input.
    """
    logger.info(f"Generating answer for session_id: {session_id}")

    contents_list = []

    if image_data and image_mime_type:
        try:
            # Ensure image_data is bytes, not UploadFile object directly if coming from FastAPI
            if hasattr(image_data, 'read'): # If it's a file-like object
                img_bytes = image_data.read()
            else:
                img_bytes = image_data

            pil_image = Image.open(io.BytesIO(img_bytes)) # Validate/process with PIL
            logger.info(f"Image MIME type: {image_mime_type}, size: {len(img_bytes)} bytes")
            contents_list.append(PartDict(inline_data=PartDict.Blob(mime_type=image_mime_type, data=img_bytes)))
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            # Fallback or raise error
            return "I had trouble understanding the image. Please try uploading a clear image in PNG or JPEG format."

    if text_prompt:
        contents_list.append(PartDict(text=text_prompt))
    elif not image_data: # Must have at least text or image
        logger.error("No text prompt or image data provided.")
        return "Please provide a question or an image."

    if not contents_list:
        logger.error("Content list is empty. Cannot generate answer.")
        return "I didn't receive any input to process."

    try:
        logger.info(f"Sending to Gemini: {len(contents_list)} parts. Text prompt (first 50 chars): '{text_prompt[:50] if text_prompt else 'N/A'}'")

        # Using the direct `generate_content` method for multimodal input as per docs
        model_instance = genai.GenerativeModel(
            MODEL_NAME,
            system_instruction=SYSTEM_PROMPT, # Pass system instruction at model level
            safety_settings=safety_settings
        )

        response = model_instance.generate_content(
            contents=contents_list, # List of Parts or strings/Images
            generation_config=generation_config, # Full config object
        )

        logger.info(f"Received response. Usage metadata: {response.usage_metadata if hasattr(response, 'usage_metadata') else 'N/A'}")

        if not response.parts:
            block_reason_text = "Unknown reason"
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                 block_reason_text = response.prompt_feedback.block_reason.name
            logger.warning(f"Response potentially blocked. Finish reason: {block_reason_text}")
            return f"I couldn't generate a response for that query due to safety guidelines ({block_reason_text}). Could you try rephrasing or asking something else?"

        # Extract text from response
        # response.text directly gives the combined text of all parts
        response_text = response.text
        return response_text

    except Exception as e:
        logger.error(f"Error during Gemini API call: {e}", exc_info=True)
        return f"Sorry, I encountered an error trying to process your request: {str(e)}" 