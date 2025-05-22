# Science Helper (Class 9-10)

A web-based AI assistant that helps students understand science concepts in Physics, Chemistry, and Biology up to Class 10 level.

## Features

- Interactive web interface for asking science questions
- Support for both text and image-based questions
- Powered by Google's Gemini 2.5 Flash model
- Covers Physics, Chemistry, and Biology topics
- Clear, concise, and student-friendly explanations

## Setup

1. Clone the repository:
```bash
git clone https://github.com/ThisisBizness/Pronab-Science-9-10-.git
cd Pronab-Science-9-10-
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

4. Run the application:
```bash
python main.py
```

The application will be available at `http://127.0.0.1:8000`

## Project Structure

- `main.py`: FastAPI application setup and endpoints
- `chat_logic.py`: Core logic for interacting with the Gemini model
- `static/`: Frontend files
  - `index.html`: Main interface
  - `style.css`: Styling
  - `script.js`: Frontend logic

## Usage

1. Open the web interface in your browser
2. Type your science question or upload an image of a question
3. Click "Ask Question" to get an explanation
4. The AI will provide a clear, concise answer suitable for Class 10 students

## Note

This application is for educational purposes only. Always cross-verify information with your textbooks and teachers. 