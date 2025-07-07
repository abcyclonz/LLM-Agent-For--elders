# main.py (FastAPI Application)
from fastapi import FastAPI, HTTPException, UploadFile, File, Form # Removed Response as it's not needed without /speak_response
from pydantic import BaseModel
import uvicorn
import os
import uuid
from typing import TypedDict, List, Annotated, Union
import re
from datetime import datetime
import asyncio
import io

# --- Local Whisper Import ---
import whisper
import numpy as np
import torchaudio
import torch
# Removed torch.serialization as it was only used for Coqui TTS safe_globals

# --- Coqui TTS (XTTSv2) Imports ---
# ALL COQUI TTS RELATED IMPORTS REMOVED

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
from health_monitor_tool import check_health_data

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
    description="API for the Senior Assistance Agent, providing conversational, memory, and voice input (Whisper). TTS functionality is disabled, and the /speak_response endpoint has been removed.",
    version="1.0.8", # Updated version to reflect /speak_response removal
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

# --- Pydantic Models for Request/Response ---
class ChatRequest(BaseModel):
    session_id: str
    user_input: str

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

# Removed VoiceOutputRequest Pydantic model as /speak_response is removed


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

# --- Coqui XTTSv2 TTS Globals (removed) ---
# ALL COQUI TTS RELATED GLOBALS REMOVED


@app.on_event("startup")
async def startup_event():
    global router_llm, main_llm, summarizer_llm, fact_extractor_llm
    global embedding_model, chroma_client, facts_collection, summaries_collection
    global app_graph, user_persona_data, whisper_model

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
        facts_collection = chroma_client.get_or_create_collection(name=FACTS_COLLECTION_NAME)
        print(f"Fact collection '{FACTS_COLLECTION_NAME}' loaded/created. Initial count: {facts_collection.count()}")
        summaries_collection = chroma_client.get_or_create_collection(name=SUMMARIES_COLLECTION_NAME)
        print(f"Summaries collection '{SUMMARIES_COLLECTION_NAME}' loaded/created. Initial count: {summaries_collection.count()}")
    except Exception as e:
        print(f"CRITICAL Error initializing ChromaDB collections: {e}")
        raise RuntimeError(f"Failed to initialize ChromaDB: {e}")

    print("Loading local Whisper model ('base' model)... This may take a moment.")
    try:
        whisper_model = whisper.load_model("base")
        print("Whisper model loaded successfully.")
    except Exception as e:
        print(f"CRITICAL Error loading Whisper model: {e}")
        print("Ensure 'ffmpeg' is installed and you have sufficient memory.")
        raise RuntimeError(f"Failed to load Whisper model: {e}")

    # --- TTS Model Loading Block Removed ---
    print("TTS functionality is currently disabled and the /speak_response endpoint has been removed.")


    print("Compiling LangGraph app...")
    app_graph = get_compiled_app()
    print("LangGraph app compiled.")

    user_persona_data = {
        "name": "Aswin",
        "age_group": "Elderly (70s)",
        "preferred_language": "English",
        "background": "Retired history teacher, loves sharing stories from his past.",
        "interests": ["history", "watching old movies", "woodworking", "cricket"],
        "communication_style_preference": "respectful, enjoys a good chat, appreciates when his experiences are acknowledged.",
        "technology_use": "Uses a tablet for news and games.",
        "goals_with_agent": "discuss topics of interest, reminisce,to be a friend, get help finding information online, light-hearted conversation, feel understood and less lonely."
    }

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

# --- Text-to-Speech functions removed ---
# The generate_speech_disabled function is no longer needed as the endpoint itself is gone.


# --- Agent Nodes ---

def entry_node(state: AgentState) -> dict:
    print("\n--- Entry Node ---")
    new_turn_count = state.get('turn_count', 0) + 1
    print(f"  Turn count: {new_turn_count}")
    return {"turn_count": new_turn_count, "retrieved_context": "", "tool_result": None}

def fact_extraction_node(state: AgentState) -> dict:
    global fact_extractor_llm, embedding_model, facts_collection
    print("--- Fact Extraction Node ---")
    if not state["user_input"]:
        print("  No new user input to analyze for facts.")
        return {}
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
    updated_state_dict = {}
    if extracted_fact != NO_FACT_TOKEN and extracted_fact:
        print(f"  Extracted fact: {extracted_fact}")
        if facts_collection:
            try:
                fact_id = str(uuid.uuid4())
                fact_embedding = embedding_model.embed_documents([extracted_fact])[0]
                facts_collection.add(
                    ids=[fact_id],
                    embeddings=[fact_embedding],
                    documents=[extracted_fact],
                    metadatas=[{"source": "user_statement", "turn": state.get("turn_count", 0)}]
                )
                print(f"  Fact added to ChromaDB (Collection: {FACTS_COLLECTION_NAME}) with ID: {fact_id}")
                current_episodic_log = state.get("episodic_memory_session_log", [])
                updated_state_dict["episodic_memory_session_log"] = current_episodic_log + [extracted_fact]
            except Exception as e:
                print(f"  Error adding fact to ChromaDB: {e}")
        else:
            print("  ChromaDB facts_collection not available. Fact not persisted.")
    else:
        print("  No specific fact extracted.")
    return updated_state_dict

def assimilate_health_data_node(state: AgentState) -> dict:
    global facts_collection, embedding_model
    health_alerts = state.get("health_alerts")
    if not health_alerts:
        return {}

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

            facts_collection.add(
                ids=fact_ids,
                embeddings=fact_embeddings,
                documents=facts_to_add,
                metadatas=[{"source": "health_monitor", "date": today_str}] * len(facts_to_add)
            )
            print(f"  Successfully added {len(facts_to_add)} health fact(s) to ChromaDB.")
            current_episodic_log = state.get("episodic_memory_session_log", [])
            return {"episodic_memory_session_log": current_episodic_log + facts_to_add}

        except Exception as e:
            print(f"  Error adding health facts to ChromaDB: {e}")
    return {}

def router_node(state: AgentState) -> dict:
    global router_llm
    print("--- Router Node ---")
    user_input = state["user_input"]

    prompt_template = ChatPromptTemplate.from_messages([
        ("system",
         f"You are an expert router. Your job is to choose the best action to address the user's latest message. You have three choices:\n\n"
         f"1. `{CALENDAR_ACTION}`: Select this to **create or add a new event** to the calendar. Use for requests like 'schedule a meeting' or 'remind me to...'. **Do NOT use this to list existing events.**\n\n"
         f"2. `{RETRIEVE_ACTION}`: Select this to **answer questions from memory**. This is for questions like 'when was my meeting?', 'what did I say?', or **for requests to list or summarize known information like 'what are my upcoming schedules?'.**\n\n"
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
    return {"router_decision": decision}

def retrieve_memory_node(state: AgentState) -> dict:
    global embedding_model, facts_collection, summaries_collection, router_llm
    print("--- Retrieve Memory Node (RAG) ---")
    if state.get("router_decision") != RETRIEVE_ACTION:
        return {"retrieved_context": ""}

    current_user_query = state["user_input"]
    if not current_user_query:
        return {"retrieved_context": ""}

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
                include=["documents", "distances"]
            )
            if fact_results and fact_results.get("documents") and fact_results["documents"][0]:
                docs = fact_results["documents"][0]
                dists = fact_results.get("distances", [[]])[0]
                retrieved_facts_str = "\n".join(f"- Fact: {doc} (Similarity Score: {1 - dist:.4f})" for doc, dist in zip(docs, dists))
                retrieved_context_parts.append(f"Potentially relevant specific facts known:\n{retrieved_facts_str}")
                print(f"  Retrieved {len(docs)} fact(s).")

        if summaries_collection and summaries_collection.count() > 0:
            summary_results = summaries_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(RETRIEVAL_K, summaries_collection.count()),
                include=["documents", "distances"]
            )
            if summary_results and summary_results.get("documents") and summary_results["documents"][0]:
                docs = summary_results["documents"][0]
                dists = summary_results.get("distances", [[]])[0]
                retrieved_summaries_str = "\n".join(f"- Summary: {doc} (Similarity Score: {1 - dist:.4f})" for doc, dist in zip(docs, dists))
                retrieved_context_parts.append(f"Potentially relevant past conversation summaries:\n{retrieved_summaries_str}")
                print(f"  Retrieved {len(docs)} summary(ies).")

    except Exception as e:
        return {"retrieved_context": f"Error retrieving memories: {e}"}

    context_str = "\n\n".join(retrieved_context_parts).strip()
    if not context_str:
        print("  No relevant dynamic memories retrieved from ChromaDB via RAG.")
        return {"retrieved_context": ""}

    print(f"  Retrieved RAG context (first 300 chars):\n{context_str[:300]}...")
    return {"retrieved_context": context_str}

def calendar_tool_node(state: AgentState) -> dict:
    """Executes the calendar tool and puts the result in the state."""
    print("--- Calendar Tool Node ---")
    user_input = state["user_input"]
    conversation_history = state["messages"]
    result_string = schedule_event(user_input, conversation_history)
    print(f"  Calendar tool result: {result_string}")
    return {"tool_result": result_string}

def generate_response_node(state: AgentState) -> dict:
    global main_llm, AGENT_NAME
    print("--- Generate Response Node ---")
    user_input = state["user_input"]

    tool_result = state.get("tool_result")
    retrieved_context_str = state.get("retrieved_context")
    health_alerts = state.get("health_alerts")

    system_prompt_content = (
        f"You are the '{AGENT_NAME}', a kind, patient, and empathetic AI companion. "
        "Your primary role is to be a supportive and engaging conversational partner."
    )

    user_persona_data = state.get("user_persona")
    formatted_user_persona = format_persona_for_prompt(user_persona_data)
    if formatted_user_persona:
        system_prompt_content += f"\n\n--- User Information ---\n{formatted_user_persona}"

    turn_specific_task = ""
    if retrieved_context_str:
        print("  >> Entering Focused Question-Answering Mode.")
        qa_prompt = (
            "You are a Question-Answering engine. Your sole task is to answer the user's question based on the provided 'Context'.\n"
            "1. Analyze the user's question.\n"
            "2. Find the specific answer within the 'Context'.\n"
            "3. State the answer clearly and concisely, starting with a phrase like 'Based on my notes...' or 'I found that...'.\n"
            "4. If the context does not contain the answer, state 'I couldn't find a specific answer to your question in my notes.'\n"
            "5. Do not add any conversational fluff or use information outside the provided 'Context'.\n\n"
            f"--- Context ---\n{retrieved_context_str}\n----------------"
        )
        prompt_parts = [
            SystemMessage(content=qa_prompt),
            HumanMessage(content=user_input)
        ]
    else:
        print("  >> Entering General Conversational Mode.")
        if health_alerts:
            alerts_str = "\n- ".join(health_alerts)
            turn_specific_task = (
                "!!! URGENT HEALTH ALERT !!!\n"
                "Your most important mission is to gently inform the user about the following health observations. "
                "You MUST begin your response by addressing these points. This is your highest priority.\n"
                "Health Observations:\n- "
                f"{alerts_str}"
            )
        elif tool_result:
            turn_specific_task = (
                "Your mission is to report the result of the tool you just used. "
                "Convey this information clearly and conversationally to the user.\n"
                f"Tool Result: '{tool_result}'"
            )
        else:
            turn_specific_task = (
                "Your mission is to be a good listener and conversational partner. "
                "Respond directly to the user's last message in a natural, engaging way."
            )

        system_prompt_content += f"\n\n--- YOUR MISSION FOR THIS TURN ---\n{turn_specific_task}"

        prompt_parts = [SystemMessage(content=system_prompt_content.strip())]
        prompt_parts.extend(state["messages"])
        prompt_parts.append(HumanMessage(content=user_input))

    try:
        response = main_llm.invoke(prompt_parts)
        ai_response_content = response.content
    except Exception as e:
        ai_response_content = f"I'm sorry, I encountered an error: {e}"

    print(f"  AI Response: {ai_response_content}")
    updated_messages = add_messages(state["messages"], [HumanMessage(content=user_input), AIMessage(content=ai_response_content)])
    return {"messages": updated_messages, "user_input": ""}

def check_and_summarize_node(state: AgentState) -> dict:
    global summarizer_llm, summaries_collection, embedding_model
    print("--- Check and Summarize Node ---")
    turn_count = state["turn_count"]
    updated_state_dict = {}
    if turn_count > 0 and turn_count % SUMMARY_INTERVAL == 0:
        print(f"  Turn {turn_count}, triggering summarization.")
        num_messages_to_summarize = SUMMARY_INTERVAL * 2
        messages_to_summarize_candidates = state["messages"]
        start_index_for_summary = max(0, len(messages_to_summarize_candidates) - num_messages_to_summarize)
        messages_to_summarize = messages_to_summarize_candidates[start_index_for_summary:]
        if not messages_to_summarize or len(messages_to_summarize) < 2:
            print("  Not enough new messages to summarize meaningfully.")
            return {}
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
                    summaries_collection.add(
                        ids=[summary_id],
                        embeddings=[summary_embedding],
                        documents=[summary],
                        metadatas=[{"source": "conversation_summary", "turn": turn_count}]
                    )
                    print(f"  Summary added to ChromaDB (Collection: {SUMMARIES_COLLECTION_NAME}) with ID: {summary_id}")
                    current_ltm_log = state.get("long_term_memory_session_log", [])
                    updated_state_dict["long_term_memory_session_log"] = current_ltm_log + [summary]
                except Exception as e:
                    print(f"  Error adding summary to ChromaDB: {e}")
            else:
                print("  ChromaDB summaries_collection not available. Fact not persisted.")
        else:
            print("  Generated an empty or non-substantive summary.")
    else:
        print(f"  Turn {turn_count}, no summary needed yet (interval {SUMMARY_INTERVAL}).")
    return updated_state_dict

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
def get_chroma_facts_for_display(limit=10):
    if facts_collection and facts_collection.count() > 0:
        try:
            results = facts_collection.get(limit=min(limit, facts_collection.count()), include=["documents", "metadatas"])
            serializable_metadatas = [dict(m) if m is not None else {} for m in results.get("metadatas", [])]
            return {"documents": results.get("documents", []), "metadatas": serializable_metadatas}
        except Exception as e:
            print(f"Error fetching facts for display: {e}")
            return {"documents": [f"Error fetching facts: {e}"], "metadatas":[{}]}
    return {"documents": [], "metadatas":[]}

def get_chroma_summaries_for_display(limit=10):
    if summaries_collection and summaries_collection.count() > 0:
        try:
            results = summaries_collection.get(limit=min(limit, summaries_collection.count()), include=["documents", "metadatas"])
            serializable_metadatas = [dict(m) if m is not None else {} for m in results.get("metadatas", [])]
            return {"documents": results.get("documents", []), "metadatas":serializable_metadatas}
        except Exception as e:
            print(f"Error fetching summaries for display: {e}")
            return {"documents": [f"Error fetching summaries: {e}"], "metadatas":[{}]}
    return {"documents": [], "metadatas":[]}

# --- API Endpoints ---

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    user_input = request.user_input

    if session_id not in session_states:
        session_states[session_id] = {
            "messages": [],
            "long_term_memory_session_log": [],
            "episodic_memory_session_log": [],
            "user_persona": user_persona_data,
            "user_input": "",
            "turn_count": 0,
            "router_decision": "",
            "retrieved_context": "",
            "tool_result": None,
            "health_alerts": None
        }

    current_graph_input_state = session_states[session_id].copy()
    current_graph_input_state["user_input"] = user_input

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

@app.post("/chat_voice", response_model=ChatResponse)
async def chat_voice_endpoint(
    session_id: Annotated[str, Form()],
    audio_file: Annotated[UploadFile, File()],
):
    print(f"Received voice input for session: {session_id}")
    transcribed_text = None
    try:
        transcribed_text = await transcribe_audio_with_whisper(audio_file)
        print(f"Transcribed text: '{transcribed_text}'")
    except HTTPException as e:
        print(f"Transcription error: {e.detail}")
        transcribed_text = f"Error transcribing audio: {e.detail}"

    if session_id not in session_states:
        session_states[session_id] = {
            "messages": [],
            "long_term_memory_session_log": [],
            "episodic_memory_session_log": [],
            "user_persona": user_persona_data,
            "user_input": "",
            "turn_count": 0,
            "router_decision": "",
            "retrieved_context": "",
            "tool_result": None,
            "health_alerts": None
        }

    current_graph_input_state = session_states[session_id].copy()
    current_graph_input_state["user_input"] = transcribed_text

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
            transcribed_text=transcribed_text
        )
    except Exception as e:
        print(f"Error invoking agent for session {session_id} after transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error after transcription: {e}")

# Removed /speak_response endpoint


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

    facts_data = get_chroma_facts_for_display(limit=limit)
    summaries_data = get_chroma_summaries_for_display(limit=limit)

    formatted_facts = []
    for doc, meta in zip(facts_data["documents"], facts_data["metadatas"]):
        formatted_facts.append({"document": doc, "metadata": meta})

    formatted_summaries = []
    for doc, meta in zip(summaries_data["documents"], summaries_data["metadatas"]):
        formatted_summaries.append({"document": doc, "metadata": meta})

    return MemoryQueryResponse(facts=formatted_facts, summaries=formatted_summaries)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)