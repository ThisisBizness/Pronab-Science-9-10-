import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging
from google.generativeai.types import GenerationConfig, ContentDict, PartDict
from google.ai.generativelanguage import SafetySetting, HarmCategory
# PIL Image and io are not directly used in this version of the file,
# but might be if you pre-process images before sending bytes.
# For now, assuming bytes are handled correctly by the calling main.py.
# from PIL import Image
# import io
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
MODEL_NAME = "gemini-2.5-flash-preview-05-20"

# System Prompt for Science Helper (up to Class 10, Physics, Chem, Bio, image input, with science-math)
SYSTEM_PROMPT = """
**Your Role and Persona:**
You are a friendly and knowledgeable "Science Helper" for students up to Class 10. You specialize in Physics, Chemistry, and Biology. Your goal is to make science easy and fun to understand, including the mathematical parts of science problems.

**Language:**
All your responses MUST be in clear, simple **English**.

**Core Interaction Protocol:**
1.  **Receive User Input:** A student will ask a question. This might be text-only, an image of a question (e.g., from a textbook or worksheet), or a combination of text and an image.
2.  **Understand and Analyze:**
    * Carefully look at any images provided. Identify text, diagrams, formulas, or specific elements in the image that are part of the question.
    * Read any accompanying text.
    * Use your "thinking" capability to understand the core scientific concept being asked about (Physics, Chemistry, or Biology).
    * Identify if the question involves mathematical calculations as part of the science problem.
    * Break down the problem into smaller, manageable parts.
3.  **Provide Clear, Step-by-Step Answers:**
    * Provide ONLY the direct answer to the question. No extra conversational phrases, greetings, or sign-offs.
    * Explain the underlying scientific principles in a simple, step-by-step manner.
    * Use age-appropriate language (for students up to Class 10). Avoid complex jargon unless you explain it immediately.
    * If it's a problem-solving question that involves calculations (e.g., physics formulas like F=ma, V=IR; chemical stoichiometry, molarity calculations, gas laws) or balancing equations, show all the steps clearly, including the mathematical calculations. Explain the formula used, why it's used, and how the values are substituted and computed.
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
* **Accurate and Factual:** Provide correct scientific information and calculations.
* **Engaging:** Try to make science interesting!

**Scope of Knowledge:**
General science topics in Physics, Chemistry, and Biology typically covered in the curriculum for students up to Class 10 in India (e.g., CBSE, ICSE, State Boards). This includes related mathematical applications within these subjects.
* **Physics:** Motion (equations of motion), Force (F=ma, momentum), Work, Energy & Power (calculations), Light (mirror/lens formulas - basic numericals), Sound, Electricity (Ohm's law, resistance, power calculations), Magnetism, basic concepts of the universe.
* **Chemistry:** Matter, Atoms and Molecules (mole concept, molar mass), Chemical Reactions (balancing, stoichiometry), Acids/Bases/Salts (pH - basic understanding), Metals/Non-metals, Carbon compounds (introductory), Solutions (concentration terms like molarity - basic calculations).
* **Biology:** Cells, Tissues, Life Processes (e.g., calculating BMI, basic data interpretation from graphs/charts related to biological processes), Control and Coordination, Reproduction, Heredity and Evolution, Our Environment.

**Important Rules (Limitations):**
* **Focus on Science, including necessary Math:** You should solve mathematical calculations that are part of Physics, Chemistry, or Biology problems (e.g., physics formulas, chemical stoichiometry, dilutions, concentrations, calculating magnifications in biology, etc.). Show the calculation steps clearly. However, do not solve purely mathematical problems that are unrelated to these science subjects (e.g., algebra problems without a science context, pure geometry proofs, advanced calculus). If a question is *only* a math problem without any clear science context, you can politely state: "I can help with math that's part of a science problem. Is this calculation related to a specific topic in Physics, Chemistry, or Biology that you're studying?"
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
    max_output_tokens=8192,
    response_mime_type="text/plain" # Added for clarity
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
            generation_config=generation_config,
            system_instruction=SYSTEM_PROMPT # System prompt is passed here
        )
        chat_session = model.start_chat(history=[]) # Start with an empty history
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
    current_question_text_for_log = text_message # Use a distinct variable for logging
    current_image_parts_for_context = None # Use a distinct variable for context

    if image_data and image_mime_type:
        current_image_parts_for_context = [PartDict(inline_data=PartDict(data=image_data, mime_type=image_mime_type))]

    # Construct the prompt for the model based on action and available context
    if action == "ask":
        if text_message:
            content_parts.append(PartDict(text=text_message))
        if current_image_parts_for_context:
            content_parts.extend(current_image_parts_for_context)
        
        # Store context for potential follow-ups
        last_questions_context[session_id] = {'text': text_message, 'image_parts': current_image_parts_for_context}

    elif action == "regenerate":
        prev_context = last_questions_context.get(session_id, {})
        original_text = prev_context.get('text')
        original_image_parts = prev_context.get('image_parts')

        # System prompt guides the model on "Explain again". This text is for model's explicit context.
        prompt_text_for_model = "Based on the previous question and image (if any), please explain again or provide an alternative solution."
        if original_text:
            # We don't want to put the entire original text into the prompt again if it's very long
            # The model should refer to its history. A short cue can be okay.
            prompt_text_for_model += f" The previous question was about: '{original_text[:70]}...'."
        
        content_parts.append(PartDict(text=prompt_text_for_model))
        if original_image_parts: # Re-send the original image if there was one
            content_parts.extend(original_image_parts)
        current_question_text_for_log = prompt_text_for_model # For logging

    elif action == "simplify":
        prev_answer = last_answers.get(session_id)
        prompt_text_for_model = "Please make the previous explanation simpler." # System prompt will guide "Make it simpler"
        if prev_answer:
            # Similar to regenerate, providing the entire previous answer can be too much.
            # A cue is better. The model should use its conversation history.
            prompt_text_for_model += f" The previous answer started with: \"{prev_answer[:100]}...\""
        content_parts.append(PartDict(text=prompt_text_for_model))
        current_question_text_for_log = prompt_text_for_model # For logging

    if not content_parts:
        logger.warning(f"No content to send for session {session_id} with action {action}.")
        return "I'm sorry, I didn't receive a question or enough context to respond."

    logger.info(f"Sending to session {session_id} (action: {action}): Text='{current_question_text_for_log[:70] if current_question_text_for_log else 'N/A'}' Image parts present: {bool(current_image_parts_for_context or (action == 'regenerate' and last_questions_context.get(session_id, {}).get('image_parts')))}")
    
    try:
        # The SYSTEM_PROMPT is set during model initialization (system_instruction)
        # Do NOT insert it into content_parts here as that's not the standard way for system instructions.
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