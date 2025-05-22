import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging
from google.generativeai.types import GenerationConfig, ContentDict, PartDict
from google.ai.generativelanguage import SafetySetting, HarmCategory
from PIL import Image
import io
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    logger.error("GOOGLE_API_KEY not found in environment variables.")
    # In a real app, you might want to prevent startup or raise a more critical error.

try:
    genai.configure(api_key=API_KEY)
except Exception as e:
    logger.error(f"Failed to configure Google Generative AI: {e}")

# --- Model Configuration ---
MODEL_NAME = "gemini-1.5-flash-latest" # Using the latest flash model available generally.
                                        # The specific preview 'gemini-1.5-flash-preview-05-20' might phase out.
                                        # If you specifically need the preview, ensure it's available.

# System Prompt for Science Helper (up to Class 10, Physics, Chem, Bio, image input)
SYSTEM_PROMPT = """
**Your Role and Persona:**
You are a friendly and knowledgeable "Science Helper" for students up to Class 10. You specialize in Physics, Chemistry, and Biology. Your goal is to make science easy and fun to understand.

**Language:**
All your responses MUST be in clear, simple **English**.

**Core Interaction Protocol:**
1.  **Receive User Input:** A student will ask a question. This might be text-only, an image of a question (e.g., from a textbook or worksheet), or a combination of text and an image.
2.  **Understand and Analyze:**
    * Carefully look at any images provided. Identify text, diagrams, or specific elements in the image that are part of the question.
    * Read any accompanying text.
    * Use your "thinking" capability to understand the core scientific concept being asked about (Physics, Chemistry, or Biology).
    * Break down the problem into smaller, manageable parts.
3.  **Provide Clear, Step-by-Step Answers:**
    * Provide ONLY the direct answer to the question. No extra conversational phrases, greetings, or sign-offs.
    * Explain the underlying scientific principles in a simple, step-by-step manner.
    * Use age-appropriate language (for students up to Class 10). Avoid complex jargon unless you explain it immediately.
    * If it's a problem-solving question (e.g., a physics calculation or balancing a chemical equation), show the steps clearly.
    * Use examples relevant to everyday life or simple experiments where possible.
    * Format your answers for easy reading: use bullet points, **bold text** for key terms, and line breaks.
4.  **Handling Follow-up Requests (Regenerate/Simplify):**
    * If the user asks to **"Explain again"**, **"Regenerate"**, or similar, provide an alternative explanation or solution for the *original question*. Try a different approach or example, maintaining clarity and accuracy. Assume they are referring to the last question you answered.
    * If the user asks to **"Make it simpler"**, **"Explain more easily"**, or similar, break down your *previous explanation* into even simpler steps or use more basic language. Focus on clarity for a younger learner. Assume they are referring to the last answer you provided.
5.  **If an Image is Unclear:**
    * If an uploaded image is blurry, unreadable, or doesn't seem to contain a clear question, politely ask the student to upload a clearer image or type the question. For example: "The image is a bit unclear. Could you please try uploading a clearer picture or typing out the question?"

**Tone:**
* **Encouraging and Patient:** Make students feel comfortable asking questions.
* **Clear and Simple:** Avoid talking down to them, but ensure explanations are easy to grasp.
* **Accurate and Factual:** Provide correct scientific information.
* **Engaging:** Try to make science interesting!

**Scope of Knowledge:**
General science topics in Physics, Chemistry, and Biology typically covered in the curriculum for students up to Class 10 in India (e.g., CBSE, ICSE, State Boards). This includes:
* **Physics:** Motion, Force, Energy, Light, Sound, Electricity, Magnetism, basic concepts of the universe.
* **Chemistry:** Matter, Atoms and Molecules, Chemical Reactions, Acids/Bases/Salts, Metals/Non-metals, Carbon compounds (introductory).
* **Biology:** Cells, Tissues, Life Processes (nutrition, respiration, transport, excretion), Control and Coordination, Reproduction, Heredity and Evolution, Our Environment.

**Important Rules (Limitations):**
* **Strictly no Math questions.** If a question is primarily a math problem (even if science-themed), politely state that you can help with the science concepts but not the mathematical calculations themselves. For example: "I can help explain the science behind this, but I'm not designed to solve math problems."
* Only answer questions within the scope of Physics, Chemistry, and Biology for up to Class 10.
* Do not engage in conversations unrelated to these science subjects.
* If a question is ambiguous (even with text), ask for clarification. Example: "To help you better, could you tell me a bit more about [specific part of the question]?"
* **CRITICAL: Provide ONLY the answer. Do NOT add any introductory phrases (like "Hello!"), concluding phrases (like "Hope this helps!"), or any form of self-identification or disclaimers in your response. Just the direct scientific explanation or solution.**
"""

# Generation Configuration
generation_config = GenerationConfig(
    temperature=0.5, # Slightly lower for factual accuracy
    top_p=0.95,
    top_k=64,
    max_output_tokens=8192
)

# Safety Settings
safety_settings = [
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
    SafetySetting(category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=SafetySetting.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE),
]

# --- Chat Session Management ---
active_chats = {} # Key: session_id, Value: genai.ChatSession
last_questions_context = {} # Key: session_id, Value: {'text': str, 'image_parts': list[PartDict] or None}
last_answers = {} # Key: session_id, Value: str

def start_new_chat(session_id: str):
    logger.info(f"Starting new chat session: {session_id}")
    try:
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            safety_settings=safety_settings,
            generation_config=generation_config
        )
        chat_session = model.start_chat(history=[])
        active_chats[session_id] = chat_session
        last_questions_context[session_id] = {'text': None, 'image_parts': None}
        last_answers[session_id] = None
        return chat_session
    except Exception as e:
        logger.error(f"Error initializing model or starting chat for session {session_id}: {e}")
        raise

def send_message_to_model(session_id: str, text_message: str | None = None, image_data: bytes | None = None, image_mime_type: str | None = None, action: str = "ask"):
    if session_id not in active_chats:
        logger.warning(f"Session ID {session_id} not found. Starting new chat.")
        start_new_chat(session_id)

    chat_session = active_chats[session_id]
    
    content_parts = []
    current_question_text = text_message
    current_image_parts = None

    if image_data and image_mime_type:
        current_image_parts = [PartDict(inline_data=PartDict(data=image_data, mime_type=image_mime_type))]

    # Construct the prompt for the model based on action and available context
    if action == "ask":
        if text_message:
            content_parts.append(PartDict(text=text_message))
        if current_image_parts:
            content_parts.extend(current_image_parts)
        
        # Store context for potential follow-ups
        last_questions_context[session_id] = {'text': text_message, 'image_parts': current_image_parts}

    elif action == "regenerate":
        prev_context = last_questions_context.get(session_id, {})
        original_text = prev_context.get('text')
        original_image_parts = prev_context.get('image_parts')

        prompt_text = "Explain again based on the previous question"
        if original_text:
            prompt_text += f" (which was about: '{original_text[:50]}...')."
        else:
            prompt_text += "."
        
        content_parts.append(PartDict(text=prompt_text))
        if original_image_parts: # Re-send the original image if there was one
            content_parts.extend(original_image_parts)
        current_question_text = prompt_text # For logging

    elif action == "simplify":
        prev_answer = last_answers.get(session_id)
        if prev_answer:
            prompt_text = f"Make the following explanation simpler: \"{prev_answer[:100]}...\""
        else:
            prompt_text = "Make the previous explanation simpler." # Fallback
        content_parts.append(PartDict(text=prompt_text))
        current_question_text = prompt_text # For logging
        # Simplification typically doesn't need the original image again unless specified.

    if not content_parts:
        logger.warning(f"No content to send for session {session_id} with action {action}.")
        return "I'm sorry, I didn't receive a question or enough context to respond."

    logger.info(f"Sending to session {session_id} (action: {action}): Text='{current_question_text[:70] if current_question_text else 'N/A'}' Image parts present: {bool(current_image_parts or (action == 'regenerate' and last_questions_context.get(session_id, {}).get('image_parts')))}")

    try:
        # Add system prompt as the first content
        content_parts.insert(0, PartDict(text=SYSTEM_PROMPT))
        
        response = chat_session.send_message(content_parts)
        
        if not response.parts:
            block_reason = response.prompt_feedback.block_reason.name if response.prompt_feedback and response.prompt_feedback.block_reason else "UNKNOWN_REASON"
            logger.warning(f"Response potentially blocked for session {session_id}. Reason: {block_reason}")
            return f"My apologies, but I cannot respond due to safety guidelines ({block_reason}). Could you please rephrase or ask something else?"

        response_text = "".join(part.text for part in response.parts if hasattr(part, 'text'))
        last_answers[session_id] = response_text # Store the latest answer

        logger.debug(f"Session {session_id} history length: {len(chat_session.history)}")
        return response_text

    except Exception as e:
        logger.error(f"Error during send_message for session {session_id}: {e}", exc_info=True)
        return f"Sorry, I encountered an error trying to process your request: {e}" 