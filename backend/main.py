from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Response
from pydantic import BaseModel
import uvicorn
import os
import uuid
from typing import TypedDict, List, Annotated, Union
import re
from datetime import datetime
import asyncio
import io
import json
from fastapi.responses import JSONResponse

# --- Local Whisper Import ---
import whisper
import numpy as np
import torchaudio
import torch

# --- Kokoro TTS Imports ---
try:
    from kokoro import KPipeline
except ImportError:
    print("CRITICAL ERROR: Kokoro TTS library (kokoro) not found. Please install it with `pip install kokoro`.")
    KPipeline = None # Set to None to handle gracefully in startup_event


# --- Agent Imports ---
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import chromadb

# --- Import the tools ---
from calendar_tool import schedule_event
from health_monitor_tool import check_health_data, get_health_from_file

# --- CORS Middleware Import ---
from fastapi.middleware.cors import CORSMiddleware

# --- Configuration (Constants) ---
AGENT_NAME = "Senior Assistance Agent"
ROUTER_LLM_MODEL = "mistral:7b-instruct-q4_K_M"
MAIN_LLM_MODEL = "mistral:7b-instruct-q4_K_M"
SUMMARY_INTERVAL = 3
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_PERSIST_PATH = "./chroma_db_store_senior_agent"
FACTS_COLLECTION_NAME = "episodic_facts_senior_agent"
SUMMARIES_COLLECTION_NAME = "ltm_summaries_senior_agent"
RETRIEVAL_K = 3
RETRIEVAL_QUERY_HISTORY_TURNS = 2
NO_FACT_TOKEN = "NO_FACT"

# --- Actions ---
RETRIEVE_ACTION = "RETRIEVE_MEMORY"
GENERATE_ACTION = "GENERATE_ONLY"
CALENDAR_ACTION = "USE_CALENDAR_TOOL"

# Initialize FastAPI app
app = FastAPI(
    title="Senior Assistance Agent API",
    description="API for the Senior Assistance Agent, providing conversational, memory, voice input (Whisper), and voice output (Kokoro TTS).",
    version="1.2.0",
)

# --- CORS Configuration ---
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5500",
    "*" # WARNING: USE THIS ONLY FOR DEVELOPMENT. For production, specify explicit origins.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Agent State Storage (for simplicity in this example) ---
session_states = {}

# --- Import authentication modules ---
from models import User, get_db
from auth_utils import verify_password, get_password_hash, create_access_token, verify_token, generate_user_id
from sqlalchemy.orm import Session
from fastapi import Depends

# --- Pydantic Models for Request/Response ---
class ChatRequest(BaseModel):
    session_id: str
    user_input: str
    user_token: str = None  # Optional token to get user persona

class ChatResponse(BaseModel):
    session_id: str
    ai_response: str
    episodic_memory_log: List[str]
    long_term_memory_log: List[str]
    current_router_decision: str
    retrieved_context_for_turn: str
    health_alerts_for_turn: List[str]
    transcribed_text: Union[str, None] = None

class ResetRequest(BaseModel):
    session_id: str

class MemoryQueryResponse(BaseModel):
    facts: List[dict]
    summaries: List[dict]

class VoiceOutputRequest(BaseModel):
    session_id: str
    text_to_speak: str
    # Kokoro uses predefined voices, not necessarily speaker_wav_path for cloning via KPipeline
    # If using specific voice cloning, that might be handled differently or in a specific API.
    # For now, stick to predefined voices or direct voice tensor loading.
    kokoro_voice_name: str = "af_heart" # Default voice, check Kokoro docs for options (e.g., 'af_bella', 'am_adam')

# --- Authentication Models ---
class UserSignupRequest(BaseModel):
    email: str
    password: str
    full_name: str
    age: str
    preferred_language: str
    background: str
    interests: List[str]
    conversation_preferences: List[str]
    technology_usage: str
    conversation_goals: List[str]
    additional_info: str = ""

class UserLoginRequest(BaseModel):
    email: str
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    age: str
    preferred_language: str
    background: str
    interests: List[str]
    conversation_preferences: List[str]
    technology_usage: str
    conversation_goals: List[str]
    additional_info: str
    created_at: str
    updated_at: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse


# --- Agent State Definition ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    long_term_memory_session_log: List[str]
    episodic_memory_session_log: List[str]
    user_persona: Union[dict, str, None]
    user_input: str
    router_decision: str
    retrieved_context: str
    turn_count: int
    tool_result: Union[str, None]
    health_alerts: Union[List[str], None]
    snoozed_alerts: List[str]
    user_id: Union[str, None]

# --- Cached Resource Initializations ---
router_llm: ChatOllama = None
main_llm: ChatOllama = None
summarizer_llm: ChatOllama = None
fact_extractor_llm: ChatOllama = None
embedding_model: HuggingFaceEmbeddings = None
chroma_client: chromadb.PersistentClient = None
facts_collection: chromadb.Collection = None
summaries_collection: chromadb.Collection = None
app_graph: StateGraph = None
user_persona_data: dict = None
whisper_model: whisper.Whisper = None

# --- Kokoro TTS Globals ---
kokoro_pipeline = None # The KPipeline instance
KOKORO_LANG_CODE = 'a' # 'a' for American English, 'b' for British English


@app.on_event("startup")
async def startup_event():
    global router_llm, main_llm, summarizer_llm, fact_extractor_llm
    global embedding_model, chroma_client, facts_collection, summaries_collection
    global app_graph, user_persona_data, whisper_model
    global kokoro_pipeline

    print("Initializing LLMs...")
    router_llm = ChatOllama(model=ROUTER_LLM_MODEL, temperature=0.0)
    main_llm = ChatOllama(model=MAIN_LLM_MODEL, temperature=0.7)
    summarizer_llm = ChatOllama(model=MAIN_LLM_MODEL, temperature=0.2)
    fact_extractor_llm = ChatOllama(model=ROUTER_LLM_MODEL, temperature=0.1)
    print("LLMs initialized.")

    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    print("Embedding model initialized.")

    print(f"Initializing ChromaDB client at: {CHROMA_PERSIST_PATH}")
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
        facts_collection = chroma_client.get_or_create_collection(
            name=FACTS_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        if facts_collection is None:
            raise ValueError(f"Failed to get or create ChromaDB collection: {FACTS_COLLECTION_NAME}")
        print(f"Fact collection '{FACTS_COLLECTION_NAME}' loaded/created. Initial count: {facts_collection.count()}")

        summaries_collection = chroma_client.get_or_create_collection(
            name=SUMMARIES_COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        if summaries_collection is None:
            raise ValueError(f"Failed to get or create ChromaDB collection: {SUMMARIES_COLLECTION_NAME}")
        print(f"Summaries collection '{SUMMARIES_COLLECTION_NAME}' loaded/created. Initial count: {summaries_collection.count()}")
    except Exception as e:
        print(f"CRITICAL Error initializing ChromaDB collections: {e}")
        raise RuntimeError(f"Failed to initialize ChromaDB: {e}")

    print("Loading local Whisper model ('small' model)... This may take a moment.")
    try:
        whisper_model = whisper.load_model("small")
        print("Whisper model loaded successfully.")
    except Exception as e:
        print(f"CRITICAL Error loading Whisper model: {e}")
        raise RuntimeError(f"Failed to load Whisper model: {e}")

    print("Loading Kokoro TTS model...")
    if KPipeline is None:
        print("Kokoro TTS (KPipeline) class not found. TTS functionality will be unavailable.")
    else:
        try:
            kokoro_pipeline = KPipeline(lang_code=KOKORO_LANG_CODE, device="cpu")
            print("Kokoro TTS model loaded successfully (on CPU).")
        except Exception as e:
            kokoro_pipeline = None
            print(f"CRITICAL Error loading Kokoro TTS model: {e}")
            print("Ensure `kokoro` Python package is installed and `espeak-ng` system dependency is met.")
            print(f"Detailed Kokoro load error: {e}")
            raise RuntimeError(f"Failed to load Kokoro TTS model: {e}")

    print("Compiling LangGraph app...")
    app_graph = get_compiled_app()
    print("LangGraph app compiled.")

    # Minimal default persona - only used as fallback when no user token is provided
    user_persona_data = {
        "name": "Guest User",
        "age_group": "Unknown",
        "preferred_language": "English",
        "background": "Guest user without personalized profile.",
        "interests": [],
        "communication_style_preference": "general conversation",
        "technology_use": "Unknown",
        "goals_with_agent": "general assistance and conversation"
    }
    print("Minimal default persona loaded (fallback only).")

# --- Helper Functions ---
def format_messages_for_llm(messages: List[BaseMessage], max_history=10) -> str:
    formatted = []
    start_index = max(0, len(messages) - max_history)
    for msg in messages[start_index:]:
        if isinstance(msg, HumanMessage):
            formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"AI: {msg.content}")
    return "\n".join(formatted)

def format_persona_for_prompt(persona_data: Union[dict, str, None]) -> str:
    if not persona_data:
        return ""
    if isinstance(persona_data, str):
        return f"User Persona Information:\n{persona_data.strip()}\n"
    if isinstance(persona_data, dict):
        formatted_persona = "User Persona Information:\n"
        for key, value in persona_data.items():
            formatted_persona += f"- {key.replace('_', ' ').capitalize()}: {value}\n"
        return formatted_persona
    return ""

def get_user_persona_from_db(user_id: str, db: Session) -> dict:
    """Get user persona data from database"""
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return None
        
        # Convert user data to persona format
        persona = {
            "name": user.full_name,
            "age_group": f"Age {user.age}",
            "preferred_language": user.preferred_language,
            "background": user.background,
            "interests": json.loads(user.interests) if user.interests else [],
            "communication_style_preference": ", ".join(json.loads(user.conversation_preferences)) if user.conversation_preferences else "",
            "technology_use": user.technology_usage,
            "goals_with_agent": ", ".join(json.loads(user.conversation_goals)) if user.conversation_goals else "",
            "additional_info": user.additional_info
        }
        return persona
    except Exception as e:
        print(f"Error getting user persona from database: {e}")
        return None


import asyncio
# --- Speech-to-Text with Local Whisper ---
async def transcribe_audio_with_whisper(audio_file: UploadFile) -> str:
    global whisper_model
    if whisper_model is None:
        raise HTTPException(status_code=500, detail="Whisper model not loaded.")

    temp_file_path = f"temp_{uuid.uuid4()}_{audio_file.filename}"
    try:
        await asyncio.to_thread(lambda: audio_file.file.seek(0))
        audio_bytes = await audio_file.read()

        with open(temp_file_path, "wb") as f:
            f.write(audio_bytes)
        print(f"  Audio saved to {temp_file_path} for transcription.")

        audio_tensor, sample_rate = torchaudio.load(temp_file_path)
        if sample_rate != 16000:
            print(f"  Resampling audio from {sample_rate}Hz to 16000Hz.")
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_tensor = resampler(audio_tensor)
            sample_rate = 16000
        if audio_tensor.shape[0] > 1:
            print("  Converting stereo audio to mono.")
            audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
        audio_np = audio_tensor.squeeze().numpy()

        print("  Starting Whisper transcription...")
        result = whisper_model.transcribe(audio_np, fp16=torch.cuda.is_available())
        transcribed_text = result["text"].strip()
        print(f"  Whisper transcription complete.")
        return transcribed_text

    except Exception as e:
        print(f"Error during local Whisper transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Audio transcription failed: {e}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"  Cleaned up temporary file: {temp_file_path}")

# --- Text-to-Speech with Kokoro TTS ---
async def generate_speech_with_kokoro(text: str, voice_name: str) -> bytes:
    """
    Generates speech audio from text using the loaded Kokoro TTS KPipeline.
    Returns audio as bytes in WAV format.
    Uses a predefined voice name.
    """
    global kokoro_pipeline
    if kokoro_pipeline is None:
        raise HTTPException(status_code=500, detail="Kokoro TTS model not loaded or failed to initialize during startup. Please check server logs for details.")

    print(f"  Generating speech for text (first 50 chars) with Kokoro voice '{voice_name}': '{text[:50]}'")
    try:
        # Kokoro's pipeline can return a generator for streaming or a full audio tensor.
        # For simplicity, we'll get the full audio here.
        # It takes `text` and `voice`. `speed` can also be an option.
        # Check Kokoro's documentation for available voices (e.g., 'af_heart', 'am_adam', etc.).
        
        # The KPipeline call is synchronous, so wrap in asyncio.to_thread
        audio_chunks_generator = await asyncio.to_thread(
            kokoro_pipeline, # The KPipeline instance is callable
            text=text,
            voice=voice_name,
            speed=1.0, # Default speed, can be made configurable
            # split_pattern=r'\n+' # Optional: how to split text, defaults usually fine
        )
        
        # The generator yields (graphemes, phonemes, audio_array). We only need audio_array.
        # It seems to be designed for streaming, so we concatenate the chunks.
        audio_arrays = []
        for i, (gs, ps, audio_array) in enumerate(audio_chunks_generator):
            # audio_array is typically a numpy array
            audio_arrays.append(audio_array)

        if not audio_arrays:
            raise HTTPException(status_code=500, detail="Kokoro TTS generated no audio chunks.")

        # Concatenate numpy arrays and convert to PyTorch tensor for torchaudio.save
        final_audio_np = np.concatenate(audio_arrays)
        final_audio_tensor = torch.from_numpy(final_audio_np).float().unsqueeze(0) # unsqueeze(0) for (1, samples)

        # Kokoro TTS generates at 24000 Hz by default
        KOKORO_SAMPLING_RATE = 24000 # Define or get from pipeline config if available

        audio_buffer = io.BytesIO()
        torchaudio.save(audio_buffer, final_audio_tensor, KOKORO_SAMPLING_RATE, format='wav')
        audio_buffer.seek(0)
        print("  Kokoro TTS speech generation complete.")
        return audio_buffer.getvalue()

    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error during Kokoro TTS generation: {e}")
        # Add more specific error handling based on Kokoro common issues
        if "espeak-ng" in str(e):
            print("  Hint: Kokoro TTS relies on `espeak-ng`. Ensure it's installed as a system dependency (e.g., `sudo apt install espeak-ng`).")
        if "voice" in str(e):
            print("  Hint: The provided `voice_name` might be invalid. Check Kokoro documentation for available voices.")
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {e}. Check Kokoro TTS setup, voice name, and `espeak-ng` installation.")

# --- Agent Nodes ---

def entry_node(state: AgentState) -> dict:
    print("\n--- Entry Node ---")
    new_turn_count = state.get('turn_count', 0) + 1
    print(f"  Turn count: {new_turn_count}")
    # Copy the state and update only what you want to change
    new_state = dict(state)
    new_state["turn_count"] = new_turn_count
    new_state["retrieved_context"] = ""
    new_state["tool_result"] = None
    new_state["user_id"] = state.get("user_id")
    return new_state

def fact_extraction_node(state: AgentState) -> dict:
    global fact_extractor_llm, embedding_model, facts_collection
    print("--- Fact Extraction Node ---")
    if not state["user_input"]:
        print("  No new user input to analyze for facts.")
        new_state = dict(state)
        new_state["user_id"] = state.get("user_id")
        return new_state
    user_statement = state["user_input"]
    recent_history_str = format_messages_for_llm(state["messages"], max_history=6)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         f"You are an AI assistant. Analyze the user's statement to extract specific, important facts about the user, "
         f"their preferences, important entities, or key information that should be remembered for future conversations. "
         f"Do not extract generic statements or questions. Only extract declarative facts about the user or world state. "
         f"If no such specific fact is found, output ONLY '{NO_FACT_TOKEN}'. "
         f"Example User Statement: 'My favorite color is blue and I live in Paris.' Extracted Fact: 'User's favorite color is blue. User lives in Paris.' "
         f"Example User Statement: 'What's the weather like?' Extracted Fact: '{NO_FACT_TOKEN}' "
         f"Example User Statement: 'My cat's name is Whiskers.' Extracted Fact: 'User's cat is named Whiskers.'"),
        ("human", f"Recent conversation context (if any):\n{recent_history_str}\n\nUser statement to analyze: '{user_statement}'\n\nExtracted Fact: ")
    ])
    try:
        response = fact_extractor_llm.invoke(prompt_template.format_messages(
            recent_history_str=recent_history_str,
            user_statement=user_statement
        ))
        extracted_fact = response.content.strip()
    except Exception as e:
        print(f"  Error during fact extraction LLM call: {e}")
        extracted_fact = NO_FACT_TOKEN
    new_state = dict(state)
    new_state["user_id"] = state.get("user_id")
    if extracted_fact != NO_FACT_TOKEN and extracted_fact:
        print(f"  Extracted fact: {extracted_fact}")
        user_id = state.get("user_id")
        if not user_id or user_id == "None":
            print("ERROR: user_id is missing or None, cannot store fact!")
            new_state["user_id"] = user_id
            return new_state
        if facts_collection:
            try:
                fact_id = str(uuid.uuid4())
                fact_embedding = embedding_model.embed_documents([extracted_fact])[0]
                facts_collection.add(
                    ids=[fact_id],
                    embeddings=[fact_embedding],
                    documents=[extracted_fact],
                    metadatas=[{"source": "user_statement", "turn": state.get("turn_count", 0), "user_id": user_id }]
                )
                print(f"  Fact added to ChromaDB (Collection: {FACTS_COLLECTION_NAME}) with ID: {fact_id}")
                current_episodic_log = state.get("episodic_memory_session_log", [])
                new_state["episodic_memory_session_log"] = current_episodic_log + [extracted_fact]
            except Exception as e:
                print(f"  Error adding fact to ChromaDB: {e}")
        else:
            print("  ChromaDB facts_collection not available. Fact not persisted.")
    else:
        print("  No specific fact extracted.")
    new_state["user_id"] = state.get("user_id")
    return new_state

def assimilate_health_data_node(state: AgentState) -> dict:
    global facts_collection, embedding_model
    new_state = dict(state)
    new_state["user_id"] = state.get("user_id")
    health_alerts = state.get("health_alerts")
    if not health_alerts:
        return new_state
    print("--- Assimilating Health Data into Memory ---")
    today_str = datetime.now().strftime("%Y-%m-%d")
    facts_to_add = []
    for alert in health_alerts:
        fact = f"Health fact recorded on {today_str}: {alert}"
        facts_to_add.append(fact)
        print(f"  Saving fact: {fact}")
    if facts_to_add and facts_collection:
        try:
            fact_ids = [str(uuid.uuid4()) for _ in facts_to_add]
            fact_embeddings = embedding_model.embed_documents(facts_to_add)
            user_id = state.get("user_id")
            facts_collection.add(
                ids=fact_ids,
                embeddings=fact_embeddings,
                documents=facts_to_add,
                metadatas=[{"source": "health_monitor", "date": today_str, "user_id": user_id}] * len(facts_to_add)
            )
            print(f"  Successfully added {len(facts_to_add)} health fact(s) to ChromaDB.")
            current_episodic_log = state.get("episodic_memory_session_log", [])
            new_state["episodic_memory_session_log"] = current_episodic_log + facts_to_add
        except Exception as e:
            print(f"  Error adding health facts to ChromaDB: {e}")
    new_state["user_id"] = state.get("user_id")
    return new_state

def router_node(state: AgentState) -> dict:
    global router_llm
    print("--- Router Node ---")
    user_input = state["user_input"]
    

    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         f"You are an expert router. Your job is to choose the best action to address the user's latest message. You have three choices:\n\n"
         f"1. `{CALENDAR_ACTION}`: Select this to **create, add, or schedule a new event, reminder, or appointment**. Use for any request that implies a future action. Keywords: 'schedule', 'add to calendar', 'book', **'remind me'**. Example: 'remind me about the football match'.**Do NOT use this to list existing events.**\n\n"
         f"2. `{RETRIEVE_ACTION}`: Select this to **answer questions from memory**. This is for questions like 'when was my meeting?', 'what did I say?','Do you remember...' or **for requests to list or summarize known information like 'what are my upcoming schedules?'.**\n\n"
         f"3. `{GENERATE_ACTION}`: Select this for general conversation and greetings.\n\n"
         f"IMPORTANT: Your response MUST be ONLY the name of the action (e.g., `{CALENDAR_ACTION}`)."),
        ("human", f"User query: '{user_input}'\n\nAction: ")
    ])

    try:
        response = router_llm.invoke(prompt_template.format_messages(user_input=user_input))
        raw_decision = response.content.strip().upper()
        cleaned_decision = re.sub(r'[^A-Z_]', '', raw_decision)

        if cleaned_decision in [CALENDAR_ACTION, RETRIEVE_ACTION, GENERATE_ACTION]:
            decision = cleaned_decision
        else:
            print(f"  Router made an invalid decision. Cleaned output '{cleaned_decision}' from raw '{raw_decision}'. Defaulting to {GENERATE_ACTION}.")
            decision = GENERATE_ACTION

    except Exception as e:
        print(f"  Error during router LLM call: {e}. Defaulting to {GENERATE_ACTION}.")
        decision = GENERATE_ACTION

    print(f"  Router decision: {decision}")
    new_state = dict(state)
    new_state["user_id"] = state.get("user_id")
    new_state["router_decision"] = decision
    return new_state

def retrieve_memory_node(state: AgentState) -> dict:
    global embedding_model, facts_collection, summaries_collection, router_llm
    print("--- Retrieve Memory Node (RAG) ---")
    new_state = dict(state)
    new_state["user_id"] = state.get("user_id")
    if state.get("router_decision") != RETRIEVE_ACTION:
        new_state["retrieved_context"] = ""
        return new_state

    current_user_query = state["user_input"]
    if not current_user_query:
        new_state["retrieved_context"] = ""
        return new_state

    print("  >> Rewriting user query for better retrieval...")
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at rewriting a user's question into a concise, keyword-focused search query for a vector database. "
                    "Focus on the core nouns and topics. Do not answer the question, just provide the ideal search query. "
                    "For example, if the user asks 'when was my meeting about electronics', the best query is 'user's meeting about electronics'."),
        ("human", "Rewrite the following user question into a search query: '{question}'")
    ])

    query_rewriter_chain = rewrite_prompt | router_llm
    try:
        rewritten_query_response = query_rewriter_chain.invoke({"question": current_user_query})
        rewritten_query = rewritten_query_response.content.strip()
        print(f"  Rewritten search query: '{rewritten_query}'")
    except Exception as e:
        print(f"  Error during query rewriting, falling back to original query. Error: {e}")
        rewritten_query = current_user_query

    print(f"  Attempting to retrieve relevant memories for query: '{rewritten_query}'")
    try:
        query_embedding = embedding_model.embed_query(rewritten_query)

        retrieved_context_parts = []
        if facts_collection and facts_collection.count() > 0:
            fact_results = facts_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(RETRIEVAL_K, facts_collection.count()),
                include=["documents", "distances"],
                where={"user_id": state.get("user_id")}
            )
            if fact_results and fact_results.get("documents") and fact_results["documents"][0]:
                docs = fact_results["documents"][0]
                dists = fact_results.get("distances", [[]])[0]
                retrieved_facts_str = "\n".join(f"- Fact: {doc} (Similarity Score: {1 - dist:.4f})" for doc, dist in zip(docs, dists))
                retrieved_context_parts.append(f"Potentially relevant specific facts known:\n{retrieved_facts_str}")
                print(f"  Retrieved {len(docs)} fact(s).")

        if summaries_collection and summaries_collection.count() > 0:
            print("  >> Querying SUMMARY collection...")
            summary_results = summaries_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(RETRIEVAL_K, summaries_collection.count()),
                include=["documents", "distances"],
                where={"user_id": state.get("user_id")}
            )
            if summary_results and summary_results.get("documents") and summary_results["documents"][0]:
                docs = summary_results["documents"][0]
                dists = summary_results.get("distances", [[]])[0]
                retrieved_summaries_str = "\n".join(f"- Summary: {doc} (Similarity Score: {1 - dist:.4f})" for doc, dist in zip(docs, dists))
                retrieved_context_parts.append(f"Potentially relevant past conversation summaries:\n{retrieved_summaries_str}")
                print(f"  Retrieved {len(docs)} summary(ies).")
            

    except Exception as e:
        new_state["retrieved_context"] = f"Error retrieving memories: {e}"
        return new_state

    context_str = "\n\n".join(retrieved_context_parts).strip()
    if not context_str:
        print("  No relevant dynamic memories retrieved from ChromaDB via RAG.")
        new_state["retrieved_context"] = ""
        return new_state

    print(f"  Retrieved RAG context (first 300 chars):\n{context_str[:300]}...")
    new_state["retrieved_context"] = context_str
    new_state["user_id"] = state.get("user_id")
    return new_state

def calendar_tool_node(state: AgentState) -> dict:
    """Executes the calendar tool and puts the result in the state."""
    print("--- Calendar Tool Node ---")
    user_input = state["user_input"]
    conversation_history = state["messages"]
    result_string = schedule_event(user_input, conversation_history)
    print(f"  Calendar tool result: {result_string}")
    return {"tool_result": result_string}


def _has_alert_been_mentioned(alert_text: str, messages: List[BaseMessage], turns_to_check=1) -> bool:
    """Checks if a similar alert has been mentioned in the last few AI responses."""
    # We check the last `turns_to_check * 2` messages (user + AI)
    start_index = max(0, len(messages) - (turns_to_check * 2))
    # Look for a keyword from the alert in recent AI messages
    # e.g., 'blood pressure', 'fall', 'oxygen'
    try:
        keyword = alert_text.split(':')[1].split(' ')[1].lower()
    except IndexError:
        keyword = alert_text.lower() # Fallback for simple alerts like "fall detected"

    for msg in messages[start_index:]:
        if isinstance(msg, AIMessage):
            if keyword in msg.content.lower():
                print(f"  Alert keyword '{keyword}' found in recent AI message. Suppressing alert.")
                return True
    return False

def generate_response_node(state: AgentState) -> dict:
    """
    Generates a response using a session-aware snooze list to prevent repeating alerts.
    This is the final and most robust version of the agent's brain.
    """
    global main_llm, AGENT_NAME
    print("--- Generate Response Node ---")
    user_input = state["user_input"]

    # Get all necessary data from the current state
    tool_result = state.get("tool_result")
    retrieved_context_str = state.get("retrieved_context")
    health_alerts = state.get("health_alerts")
    snoozed_alerts = state.get("snoozed_alerts", []) # Get the snooze list

    user_persona_data = state.get("user_persona", {})
    user_name = user_persona_data.get("name", "the user")
    
    # This will hold the final list of prompt parts
    prompt_parts = []
    
    # This will hold any new alerts that we discuss in this turn
    newly_snoozed_keywords = []

    # --- START: FINAL, ROBUST LOGIC BLOCK ---

    # Determine which alerts are new and need to be mentioned.
    new_alerts_to_mention = []
    if health_alerts:
        for alert in health_alerts:
            # Create a simple, stable keyword for the alert type (e.g., 'pressure', 'fall')
            try:
                keyword = alert.split(':')[0].split(' ')[-1].lower()
            except IndexError:
                keyword = alert.split(' ')[0].lower()
            
            # If this alert type is NOT already in our session's snooze list, it's new.
            if keyword not in snoozed_alerts:
                new_alerts_to_mention.append(alert)
                newly_snoozed_keywords.append(keyword) # Mark it to be snoozed

    if new_alerts_to_mention:
        # --- PRIORITY 1: Forced Health Alert Mode for NEW alerts ---
        print(f"  >> Entering FORCED Health Alert Mode for new alerts: {new_alerts_to_mention}")
        alerts_str = "\n- ".join(new_alerts_to_mention)
        forced_instruction = (
            f"(SYSTEM NOTE: You have detected new, critical health alerts that have not been discussed yet. Your only goal for this turn is to address them. "
            f"Start your response by gently informing {user_name} about the following, then ask if they are okay. "
            f"New Alerts to address: {alerts_str})"
        )
        system_prompt = (
            f"You are the '{AGENT_NAME}', a kind, patient, and empathetic AI companion. "
            "Your highest priority is the user's safety and well-being."
        )
        prompt_parts = [
            SystemMessage(content=system_prompt),
            *state["messages"],
            HumanMessage(content=forced_instruction)
        ]

    elif retrieved_context_str:
        # --- PRIORITY 2: Question Answering Mode ---
        print("  >> Entering Persona-Aware Question-Answering Mode.")
        qa_prompt = (
            f"You are the '{AGENT_NAME}', a helpful and kind AI companion. You are speaking directly to your friend, {user_name}.\n\n"
            "**Your Task:**\n"
            f"You need to answer {user_name}'s question. The 'Context' below is **your own memory** - it contains facts you have learned about them in the past. Read their question, find the answer in your memory, and respond to them naturally in the first person ('I').\n\n"
            "**Rules:**\n"
            "1.  **Speak as 'I'.** For example, if the context says 'User is a history teacher', and the user asks 'what was my job?', you should answer 'I remember you telling me you were a history teacher.'\n"
            "2.  **Address them as 'you'.**\n"
            "3.  **Do not say 'based on the context' or refer to 'the user' in your response.** Treat the context as your own knowledge.\n"
            "4.  If the answer isn't in your memory, say something natural like 'I don't seem to recall that, I'm sorry.'\n\n"
            "--- CONTEXT (YOUR MEMORY) ---\n"
            f"{retrieved_context_str}\n"
            "---------------------------\n\n"
            f"--- {user_name.upper()}'S QUESTION ---\n"
            f"{user_input}\n"
            "------------------------\n\n"
            f"**YOUR RESPONSE TO {user_name.upper()}:**"
        )
        prompt_parts = [
            SystemMessage(content=qa_prompt)
        ]

    else:
        # --- PRIORITY 3: Standard Conversational Flow ---
        print("  >> Entering Standard Conversational Mode (Tool or General).")
        system_prompt_content = (
            f"You are the '{AGENT_NAME}', a kind, patient, and empathetic AI companion. "
            "Your primary role is to be a supportive and engaging conversational partner."
        )
        if user_persona_data:
             system_prompt_content += f"\n\n--- User Information ---\n{format_persona_for_prompt(user_persona_data)}"
        
        turn_specific_task = ""
        if tool_result:
            if "error" in tool_result.lower() or "failed" in tool_result.lower():
                turn_specific_task = (
                    "Your mission is to apologize and explain that a technical problem occurred. "
                    "Tell the user that you failed to complete the action due to a technical error with one of your tools (like the calendar tool). "
                    "Do not show them the raw error message. Just say something went wrong and you can't do it right now."
                )
            else:
                turn_specific_task = (
                    "Your mission is to confirm to the user that you have completed their request. "
                    "Speak naturally, as if you did it yourself. Do NOT mention a 'tool'.\n"
                    "For example, instead of 'The tool succeeded', say 'Okay, I've scheduled that for you.'\n\n"
                    f"Information to convey: '{tool_result}'"
                )
        else:
            turn_specific_task = (
                "Your mission is to be a good listener and conversational partner. "
                "Your response should be empathetic and directly related to the user's last message.\n"
                "**RULES FOR THIS MISSION:**\n"
                "1. **Prioritize Listening:** If the user shares something personal or emotional, your first step is to validate their feelings (e.g., 'I'm sorry to hear that,' 'That sounds difficult.').\n"
                "2. **Ask Clarifying Questions:** Instead of jumping to solutions, ask open-ended questions to encourage them to share more (e.g., 'Can you tell me more about that?', 'How did that make you feel?').\n"
                "3. **Do NOT give unsolicited advice** (like suggesting therapy or specific actions) unless the user explicitly asks for help or suggestions."
            )

        system_prompt_content += f"\n\n--- YOUR MISSION FOR THIS TURN ---\n{turn_specific_task}"
        prompt_parts = [SystemMessage(content=system_prompt_content.strip())]
        prompt_parts.extend(state["messages"])
        prompt_parts.append(HumanMessage(content=user_input))

    # --- LLM Invocation (no changes) ---
    try:
        response = main_llm.invoke(prompt_parts)
        ai_response_content = response.content
    except Exception as e:
        ai_response_content = f"I'm sorry, I encountered an error: {e}"

    print(f"  AI Response: {ai_response_content}")
    updated_messages = add_messages(state["messages"], [HumanMessage(content=user_input), AIMessage(content=ai_response_content)])
    
    # --- FINAL STEP: Update the state with the new snooze list for the next turn ---
    updated_snooze_list = snoozed_alerts + newly_snoozed_keywords
    print(f"  Updating session snooze list to: {updated_snooze_list}")
    
    new_state = dict(state)
    new_state["messages"] = updated_messages
    new_state["user_input"] = ""
    new_state["snoozed_alerts"] = updated_snooze_list
    new_state["user_id"] = state.get("user_id")
    return new_state

def check_and_summarize_node(state: AgentState) -> dict:
    global summarizer_llm, summaries_collection, embedding_model
    print("--- Check and Summarize Node ---")
    turn_count = state["turn_count"]
    new_state = dict(state)
    new_state["user_id"] = state.get("user_id")
    
    if turn_count > 0 and turn_count % SUMMARY_INTERVAL == 0:
        print(f"  Turn {turn_count}, triggering summarization.")
        num_messages_to_summarize = SUMMARY_INTERVAL * 2
        messages_to_summarize_candidates = state["messages"]
        start_index_for_summary = max(0, len(messages_to_summarize_candidates) - num_messages_to_summarize)
        messages_to_summarize = messages_to_summarize_candidates[start_index_for_summary:]
        if not messages_to_summarize or len(messages_to_summarize) < 2:
            return new_state
        conversation_str = format_messages_for_llm(messages_to_summarize, max_history=len(messages_to_summarize))
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a summarization expert. Summarize the key topics, decisions, and important information "
                        "exchanged in the following conversation segment. Focus on information that would be useful "
                        "to recall in future, distinct conversations. Be concise and factual. Do not include conversational fluff. "
                        "The summary should be self-contained and understandable without the original conversation."),
            ("human", f"Conversation segment to summarize:\n{conversation_str}\n\nConcise Summary: ")
        ])
        try:
            response = summarizer_llm.invoke(prompt_template.format_messages(conversation_str=conversation_str))
            summary = response.content.strip()
        except Exception as e:
            print(f"  Error during summarizer LLM call: {e}")
            summary = f"Error summarizing conversation segment at turn {turn_count}."
        if summary and summary.lower() not in ["no summary needed.", "no new information to summarize."]:
            print(f"  Generated summary: {summary}")
            if summaries_collection:
                try:
                    summary_id = str(uuid.uuid4())
                    summary_embedding = embedding_model.embed_documents([summary])[0]
                    user_id = state.get("user_id")
                    summaries_collection.add(
                        ids=[summary_id],
                        embeddings=[summary_embedding],
                        documents=[summary],
                        metadatas=[{"source": "conversation_summary", "turn": turn_count, "user_id": user_id }]
                    )
                    print(f"  Summary added to ChromaDB (Collection: {SUMMARIES_COLLECTION_NAME}) with ID: {summary_id}")
                    current_ltm_log = state.get("long_term_memory_session_log", [])
                    new_state["long_term_memory_session_log"] = current_ltm_log + [summary]
                except Exception as e:
                    print(f"  Error adding summary to ChromaDB: {e}")
            else:
                print("  ChromaDB summaries_collection not available. Fact not persisted.")
        else:
            print("  Generated an empty or non-substantive summary.")
    else:
        print(f"  Turn {turn_count}, no summary needed yet (interval {SUMMARY_INTERVAL}).")
    new_state["user_id"] = state.get("user_id")
    return new_state

# --- LangGraph Graph Definition ---

def get_compiled_app():
    print(f"Compiling LangGraph app for {AGENT_NAME}...")
    workflow = StateGraph(AgentState)
    workflow.add_node("entry", entry_node)
    workflow.add_node("assimilate_health", assimilate_health_data_node)
    workflow.add_node("extract_fact", fact_extraction_node)
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve_memory", retrieve_memory_node)
    workflow.add_node("calendar_tool", calendar_tool_node)
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("check_summarize", check_and_summarize_node)

    workflow.set_entry_point("entry")

    workflow.add_edge("entry", "assimilate_health")
    workflow.add_edge("assimilate_health", "extract_fact")
    workflow.add_edge("extract_fact", "router")

    def decide_action_path(state: AgentState):
        decision = state.get("router_decision")
        if decision == RETRIEVE_ACTION:
            return "retrieve_memory"
        if decision == CALENDAR_ACTION:
            return "calendar_tool"
        return "generate_response"

    workflow.add_conditional_edges(
        "router",
        decide_action_path,
        {
            "retrieve_memory": "retrieve_memory",
            "calendar_tool": "calendar_tool",
            "generate_response": "generate_response"
        }
    )

    workflow.add_edge("retrieve_memory", "generate_response")
    workflow.add_edge("calendar_tool", "generate_response")
    workflow.add_edge("generate_response", "check_summarize")
    workflow.add_edge("check_summarize", END)

    _app = workflow.compile()
    return _app

# --- Helper Functions to Fetch Chroma Data for Display ---
def get_chroma_facts_for_display(user_id, limit=10):
    if facts_collection and facts_collection.count() > 0:
        try:
            results = facts_collection.get(
                where={"user_id": user_id},
                limit=min(limit, facts_collection.count()),
                include=["documents", "metadatas"]
            )
            serializable_metadatas = [dict(m) if m is not None else {} for m in results.get("metadatas", [])]
            return {"documents": results.get("documents", []), "metadatas": serializable_metadatas}
        except Exception as e:
            print(f"Error fetching facts for display: {e}")
            return {"documents": [f"Error fetching facts: {e}"], "metadatas":[{}]}
    return {"documents": [], "metadatas":[]}

def get_chroma_summaries_for_display(user_id, limit=10):
    if summaries_collection and summaries_collection.count() > 0:
        try:
            results = summaries_collection.get(
                where={"user_id": user_id},
                limit=min(limit, summaries_collection.count()),
                include=["documents", "metadatas"]
            )
            serializable_metadatas = [dict(m) if m is not None else {} for m in results.get("metadatas", [])]
            return {"documents": results.get("documents", []), "metadatas":serializable_metadatas}
        except Exception as e:
            print(f"Error fetching summaries for display: {e}")
            return {"documents": [f"Error fetching summaries: {e}"], "metadatas":[{}]}
    return {"documents": [], "metadatas":[]}

# --- API Endpoints ---

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    session_id = request.session_id
    user_input = request.user_input
    user_token = request.user_token

    user_id = None
    user_persona = user_persona_data  # Default fallback

    if user_token:
        try:
            payload = verify_token(user_token)
            if payload and payload.get("user_id"):
                user_id = payload["user_id"]
                user_persona = get_user_persona_from_db(user_id, db)
                if user_persona:
                    print(f"Loaded user persona for user {user_id}")
                else:
                    print(f"Could not load user persona, using default")
            else:
                print(f"[WARNING] Token payload missing user_id: {payload}")
        except Exception as e:
            print(f"Error loading user persona: {e}")

    # --- Always update session state with user_id if found from token ---
    if session_id not in session_states:
        session_states[session_id] = {
            "messages": [],
            "long_term_memory_session_log": [],
            "episodic_memory_session_log": [],
            "user_persona": user_persona,
            "user_input": "",
            "turn_count": 0,
            "router_decision": "",
            "retrieved_context": "",
            "tool_result": None,
            "health_alerts": None,
            "user_id": user_id,
            "snoozed_alerts": []  
        }
    else:
        # Always update user_id in session state if we have a new one (even if None, to avoid poisoning)
        if user_id is not None:
            session_states[session_id]["user_id"] = user_id
        else:
            print(f"[WARNING] user_id is missing in /chat endpoint for session {session_id}!")

    current_graph_input_state = session_states[session_id].copy()
    current_graph_input_state["user_input"] = user_input
    current_graph_input_state["user_id"] = session_states[session_id]["user_id"]

    print("--- Checking Health Data (Live from FastAPI endpoint) ---")
    live_health_alerts = check_health_data()
    current_graph_input_state["health_alerts"] = live_health_alerts

    try:
        updated_graph_output_state = app_graph.invoke(current_graph_input_state)
        session_states[session_id] = updated_graph_output_state

        ai_message_content = "Sorry, I had trouble generating a response."
        if updated_graph_output_state.get("messages"):
            last_message_in_graph = updated_graph_output_state["messages"][-1]
            if isinstance(last_message_in_graph, AIMessage):
                ai_message_content = last_message_in_graph.content

        return ChatResponse(
            session_id=session_id,
            ai_response=ai_message_content,
            episodic_memory_log=session_states[session_id].get("episodic_memory_session_log", []),
            long_term_memory_log=session_states[session_id].get("long_term_memory_session_log", []),
            current_router_decision=session_states[session_id].get("router_decision", ""),
            retrieved_context_for_turn=session_states[session_id].get("retrieved_context", ""),
            health_alerts_for_turn=session_states[session_id].get("health_alerts") or [],
            transcribed_text=None
        )
    except Exception as e:
        print(f"Error invoking agent for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@app.post("/transcribe_audio")
async def transcribe_audio_endpoint(audio_file: UploadFile = File(...)):
    """Transcribe audio and return only the text."""
    try:
        transcribed_text = await transcribe_audio_with_whisper(audio_file)
        return {"transcribed_text": transcribed_text}
    except Exception as e:
        print(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Audio transcription failed: {e}")

# Optionally, you can deprecate or remove the /chat_voice endpoint:
# @app.post("/chat_voice")
# async def chat_voice_endpoint(...):
#       raise HTTPException(status_code=410, detail="/chat_voice is deprecated. Use /transcribe_audio and /chat.")

@app.post("/speak_response")
async def speak_response_endpoint(request: VoiceOutputRequest):
    """
    Generates speech from the given text using Kokoro TTS and returns it as an audio file.
    """
    text_to_speak = request.text_to_speak
    voice_name = request.kokoro_voice_name # Get requested voice name from the client

    if not text_to_speak:
        raise HTTPException(status_code=400, detail="No text provided for speech generation.")

    try:
        audio_bytes = await generate_speech_with_kokoro(
            text_to_speak,
            voice_name=voice_name
        )
        return Response(content=audio_bytes, media_type="audio/wav")
    except HTTPException as e:
        # Re-raise HTTPExceptions directly (e.g., from validation errors or model not loaded)
        raise e
    except Exception as e:
        print(f"Error during Kokoro TTS generation from endpoint: {e}")
        # Add more specific error handling based on Kokoro common issues
        if "espeak-ng" in str(e):
            print("  Hint: Kokoro TTS relies on `espeak-ng`. Ensure it's installed as a system dependency (e.g., `sudo apt install espeak-ng`).")
        if "voice" in str(e):
            print("  Hint: The provided `voice_name` might be invalid. Check Kokoro documentation for available voices.")
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {e}. Check Kokoro TTS setup, voice name, and `espeak-ng` installation.")


@app.post("/reset_session")
async def reset_session_endpoint(request: ResetRequest):
    session_id = request.session_id
    if session_id in session_states:
        del session_states[session_id]
        print(f"Session {session_id} reset.")
        return {"message": f"Session {session_id} reset successfully."}
    else:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

@app.get("/get_memories/{session_id}", response_model=MemoryQueryResponse)
async def get_memories_endpoint(session_id: str, limit: int = 10):
    if session_id not in session_states:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")
    user_id = session_states[session_id].get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="No user_id found in session. Please log in again.")
    facts_data = get_chroma_facts_for_display(user_id, limit=limit)
    summaries_data = get_chroma_summaries_for_display(user_id, limit=limit)

    formatted_facts = []
    for doc, meta in zip(facts_data["documents"], facts_data["metadatas"]):
        formatted_facts.append({"document": doc, "metadata": meta})

    formatted_summaries = []
    for doc, meta in zip(summaries_data["documents"], summaries_data["metadatas"]):
        formatted_summaries.append({"document": doc, "metadata": meta})

    return MemoryQueryResponse(facts=formatted_facts, summaries=formatted_summaries)

@app.get("/get_profile/{session_id}")
async def get_profile_endpoint(session_id: str):
    if session_id in session_states:
        persona = session_states[session_id].get("user_persona", user_persona_data)
    else:
        persona = user_persona_data
    return JSONResponse(content={"user_persona": persona})

@app.get("/get_health_data")
def get_health_data():
    data = get_health_from_file()
    return JSONResponse(content=data)

# --- Authentication Endpoints ---

@app.post("/auth/signup", response_model=TokenResponse)
async def signup(request: UserSignupRequest, db: Session = Depends(get_db)):
    """Register a new user"""
    try:
        print(f"Signup request received: {request.email}")
        
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == request.email).first()
        if existing_user:
            print(f"Email already registered: {request.email}")
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create new user
        user_id = generate_user_id()
        hashed_password = get_password_hash(request.password)
        
        print(f"Creating user with ID: {user_id}")
        
        new_user = User(
            id=user_id,
            email=request.email,
            password_hash=hashed_password,
            full_name=request.full_name,
            age=request.age,
            preferred_language=request.preferred_language,
            background=request.background,
            interests=json.dumps(request.interests),
            conversation_preferences=json.dumps(request.conversation_preferences),
            technology_usage=request.technology_usage,
            conversation_goals=json.dumps(request.conversation_goals),
            additional_info=request.additional_info
        )
        
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        print(f"User created successfully: {new_user.email}")

        # --- Add user profile facts to ChromaDB ---
        try:
            profile_facts = []
            if new_user.full_name:
                profile_facts.append(f"User's full name is {new_user.full_name}.")
            if new_user.age:
                profile_facts.append(f"User's age is {new_user.age}.")
            if new_user.preferred_language:
                profile_facts.append(f"User's preferred language is {new_user.preferred_language}.")
            if new_user.background:
                profile_facts.append(f"User's background: {new_user.background}.")
            if new_user.interests:
                try:
                    interests = json.loads(new_user.interests)
                    if interests:
                        profile_facts.append(f"User's interests: {', '.join(interests)}.")
                except Exception:
                    pass
            if new_user.conversation_preferences:
                try:
                    prefs = json.loads(new_user.conversation_preferences)
                    if prefs:
                        profile_facts.append(f"User's conversation preferences: {', '.join(prefs)}.")
                except Exception:
                    pass
            if new_user.technology_usage:
                profile_facts.append(f"User's technology usage: {new_user.technology_usage}.")
            if new_user.conversation_goals:
                try:
                    goals = json.loads(new_user.conversation_goals)
                    if goals:
                        profile_facts.append(f"User's conversation goals: {', '.join(goals)}.")
                except Exception:
                    pass
            if new_user.additional_info:
                profile_facts.append(f"Additional info: {new_user.additional_info}.")
            if facts_collection and profile_facts:
                fact_ids = [str(uuid.uuid4()) for _ in profile_facts]
                fact_embeddings = embedding_model.embed_documents(profile_facts)
                facts_collection.add(
                    ids=fact_ids,
                    embeddings=fact_embeddings,
                    documents=profile_facts,
                    metadatas=[{"source": "user_profile", "user_id": new_user.id}] * len(profile_facts)
                )
                print(f"Added {len(profile_facts)} user profile facts to ChromaDB.")
        except Exception as e:
            print(f"Error adding user profile facts to ChromaDB: {e}")

        # Create access token
        access_token = create_access_token(
            data={"sub": new_user.email, "user_id": new_user.id}
        )
        
        if new_user.id not in session_states:
            session_states[new_user.id] = {
                "messages": [],
                "long_term_memory_session_log": [],
                "episodic_memory_session_log": [],
                "user_persona": new_user.to_dict(),
                "user_input": "",
                "turn_count": 0,
                "router_decision": "",
                "retrieved_context": "",
                "tool_result": None,
                "health_alerts": None,
                "user_id": new_user.id,
                "snoozed_alerts": []
            }

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user=UserResponse(**new_user.to_dict())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Signup error: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/auth/login", response_model=TokenResponse)
async def login(request: UserLoginRequest, db: Session = Depends(get_db)):
    """Authenticate user and return access token"""
    try:
        print(f"Login request received: {request.email}")
        
        # Find user by email
        user = db.query(User).filter(User.email == request.email).first()
        if not user:
            print(f"User not found: {request.email}")
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        print(f"User found: {user.email}")
        
        # Verify password
        if not verify_password(request.password, user.password_hash):
            print(f"Password verification failed for: {request.email}")
            raise HTTPException(status_code=401, detail="Invalid email or password")
        
        print(f"Password verified successfully for: {request.email}")

        # Ensure user profile facts exist in ChromaDB
        try:
            # Check if any facts exist for this user
            existing_facts = facts_collection.get(where={"user_id": user.id}, limit=1, include=["documents"])
            if not existing_facts.get("documents") or len(existing_facts["documents"]) == 0:
                print(f"No profile facts found in ChromaDB for user {user.id}, adding from DB.")
                profile_facts = []
                if user.full_name:
                    profile_facts.append(f"User's full name is {user.full_name}.")
                if user.age:
                    profile_facts.append(f"User's age is {user.age}.")
                if user.preferred_language:
                    profile_facts.append(f"User's preferred language is {user.preferred_language}.")
                if user.background:
                    profile_facts.append(f"User's background: {user.background}.")
                if user.interests:
                    try:
                        interests = json.loads(user.interests)
                        if interests:
                            profile_facts.append(f"User's interests: {', '.join(interests)}.")
                    except Exception:
                        pass
                if user.conversation_preferences:
                    try:
                        prefs = json.loads(user.conversation_preferences)
                        if prefs:
                            profile_facts.append(f"User's conversation preferences: {', '.join(prefs)}.")
                    except Exception:
                        pass
                if user.technology_usage:
                    profile_facts.append(f"User's technology usage: {user.technology_usage}.")
                if user.conversation_goals:
                    try:
                        goals = json.loads(user.conversation_goals)
                        if goals:
                            profile_facts.append(f"User's conversation goals: {', '.join(goals)}.")
                    except Exception:
                        pass
                if user.additional_info:
                    profile_facts.append(f"Additional info: {user.additional_info}.")
                if facts_collection and profile_facts:
                    fact_ids = [str(uuid.uuid4()) for _ in profile_facts]
                    fact_embeddings = embedding_model.embed_documents(profile_facts)
                    facts_collection.add(
                        ids=fact_ids,
                        embeddings=fact_embeddings,
                        documents=profile_facts,
                        metadatas=[{"source": "user_profile", "user_id": user.id}] * len(profile_facts)
                    )
                    print(f"Added {len(profile_facts)} user profile facts to ChromaDB on login.")
        except Exception as e:
            print(f"Error ensuring user profile facts in ChromaDB on login: {e}")

        # Create access token
        access_token = create_access_token(
            data={"sub": user.email, "user_id": user.id}
        )

        # Initialize session_states for this user session (if not already present)
        session_id = str(user.id)  # Use user.id as session_id for login context
        if user.id not in session_states:
            session_states[user.id] = {
                "messages": [],
                "long_term_memory_session_log": [],
                "episodic_memory_session_log": [],
                "user_persona": user.to_dict(),
                "user_input": "",
                "turn_count": 0,
                "router_decision": "",
                "retrieved_context": "",
                "tool_result": None,
                "health_alerts": None,
                "user_id": user.id,
                "snoozed_alerts": []
            }

        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            user=UserResponse(**user.to_dict())
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Login error: {e}")
        print(f"Error type: {type(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/auth/me", response_model=UserResponse)
async def get_current_user(token: str, db: Session = Depends(get_db)):
    """Get current user information"""
    try:
        # Verify token
        payload = verify_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Get user from database
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserResponse(**user.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Get user error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.put("/auth/update-profile", response_model=UserResponse)
async def update_user_profile(
    token: str,
    full_name: str = None,
    age: str = None,
    preferred_language: str = None,
    background: str = None,
    interests: List[str] = None,
    conversation_preferences: List[str] = None,
    technology_usage: str = None,
    conversation_goals: List[str] = None,
    additional_info: str = None,
    db: Session = Depends(get_db)
):
    """Update user profile information"""
    try:
        # Verify token
        payload = verify_token(token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Get user from database
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Update fields if provided
        if full_name is not None:
            user.full_name = full_name
        if age is not None:
            user.age = age
        if preferred_language is not None:
            user.preferred_language = preferred_language
        if background is not None:
            user.background = background
        if interests is not None:
            user.interests = json.dumps(interests)
        if conversation_preferences is not None:
            user.conversation_preferences = json.dumps(conversation_preferences)
        if technology_usage is not None:
            user.technology_usage = technology_usage
        if conversation_goals is not None:
            user.conversation_goals = json.dumps(conversation_goals)
        if additional_info is not None:
            user.additional_info = additional_info
        
        db.commit()
        db.refresh(user)
        
        return UserResponse(**user.to_dict())
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"Update profile error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)