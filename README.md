# LLM-Agent-For-Elders

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