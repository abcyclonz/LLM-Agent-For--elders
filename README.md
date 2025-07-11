# LLM-Agent-For-Elders


## The Project Goal:


The primary goal of this project was to create more than just a chatbot. We set out to build a sophisticated, stateful AI Companion specifically designed for the needs of elderly users. This means the agent needed to be more than just conversational; it had to be a reliable assistant with three core pillars:

--Memory: The ability to remember details from past conversations, both specific facts and the general context.

--Action: The ability to perform real-world tasks on the user's behalf, such as scheduling events.
--Awareness: The ability to be proactively aware of the user's well-being through real-time data, like from a smartwatch.
-The Final Architecture: A Decoupled, Multi-Modal System
To achieve this, we evolved the system from a simple script into a professional client-server architecture. This architecture is composed of two main parts:
--The Backend (The "Brain"): A high-performance Python application built with FastAPI. This is the core of the system. It handles all the complex AI logic, database interactions, and communication with external services.
--The Frontend (The "Face"): A modern web application (built with Node.js/Next.js) that provides the user interface. Its job is to capture user input (text or voice) and display the agent's responses in a friendly way.
This decoupled design is a standard industry practice that allows for scalability, independent development of the UI and the AI logic, and a much more robust and maintainable final product.
The Full User Interaction Loop (Speech-to-Speech):

The complete system now supports a full voice-in, voice-out conversation:
A user speaks into the frontend application.
The frontend sends the raw audio to the backend's /transcribe_audio endpoint.
The backend uses a local Whisper model to convert the speech into text.
The frontend receives the transcribed text and sends it to the backend's main /chat endpoint.
The backend's core AI agent processes this text, thinks, and generates a text response.
The frontend receives the AI's text response and displays it in the chat window.
Simultaneously, the frontend sends this AI text response to the backend's /speak_response endpoint.
The backend uses the Kokoro TTS (Text-to-Speech) engine to convert the text into audio.
The frontend receives the audio and plays it back to the user, completing the speech-to-speech loop.
This architecture provides a seamless, natural, and multi-modal conversational experience.

A project with a FastAPI backend and a Next.js (React) frontend to assist seniors with conversational AI, memory, and health monitoring.

---

## Prerequisites

- **Python 3.9+** (for backend)
- **Node.js 18+** and **npm** (for frontend)
- (Optional) **ffmpeg** (for audio features in backend)
- (Optional) **CUDA** (for GPU acceleration in backend, if available)

---

## Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv elderly-env
   source elderly-env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Install ffmpeg for audio processing:**
   ```bash
   sudo apt-get install ffmpeg
   ```

5. **Run the backend server:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   The backend will be available at [http://localhost:8000](http://localhost:8000).

---

## Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```
   or, if you use pnpm:
   ```bash
   pnpm install
   ```

3. **Run the frontend development server:**
   ```bash
   npm run dev
   ```
   or, with pnpm:
   ```bash
   pnpm dev
   ```

   The frontend will be available at [http://localhost:3000](http://localhost:3000).

---

## Usage

- Make sure both backend and frontend servers are running.
- The frontend will communicate with the backend API at port 8000.
- For production, configure CORS and environment variables as needed.

---

## Notes

- The backend uses FastAPI and expects all dependencies in `backend/requirements.txt`.
- The frontend uses Next.js (React) and expects all dependencies in `frontend/package.json`.
- For voice features, ensure `ffmpeg` is installed and the file `xtts_speaker_reference.wav` exists in the backend directory. 
