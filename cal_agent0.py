# ==============================================================================
# === Main Agent Script (with Memory and Calendar Tool)                      ===
# ==============================================================================

import streamlit as st
import os
import uuid
from typing import TypedDict, List, Annotated, Union
import re

# --- Streamlit Page Configuration - MUST BE THE FIRST STREAMLIT COMMAND --
st.set_page_config(page_title="Senior Assistance Agent", layout="wide")

# --- Agent Imports ---
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import chromadb

# --- NEW: Import the tool from our separate file ---
from calendar_tool import schedule_event

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

# --- MODIFIED: Added CALENDAR_ACTION ---
RETRIEVE_ACTION = "RETRIEVE_MEMORY"
GENERATE_ACTION = "GENERATE_ONLY"
CALENDAR_ACTION = "USE_CALENDAR_TOOL"

# --- Cached Resource Initializations for Streamlit (Your original code, unchanged) ---
@st.cache_resource
def get_llms():
    # ... (Your original get_llms function is perfect, no changes needed)
    print(f"Initializing LLMs for {AGENT_NAME}...")
    router_llm_ = ChatOllama(model=ROUTER_LLM_MODEL, temperature=0.0)
    main_llm_ = ChatOllama(model=MAIN_LLM_MODEL, temperature=0.7)
    summarizer_llm_ = ChatOllama(model=MAIN_LLM_MODEL, temperature=0.2)
    fact_extractor_llm_ = ChatOllama(model=ROUTER_LLM_MODEL, temperature=0.1)
    print("LLMs initialized.")
    return router_llm_, main_llm_, summarizer_llm_, fact_extractor_llm_

@st.cache_resource
def get_embedding_model():
    # ... (Your original get_embedding_model function is perfect, no changes needed)
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME} for {AGENT_NAME}...")
    _embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )
    print("Embedding model initialized.")
    return _embedding_model

@st.cache_resource
def get_chroma_collections():
    # ... (Your original get_chroma_collections function is perfect, no changes needed)
    print(f"Initializing ChromaDB client for {AGENT_NAME} at: {CHROMA_PERSIST_PATH}")
    _chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
    _facts_collection = None
    _summaries_collection = None
    try:
        _facts_collection = _chroma_client.get_or_create_collection(name=FACTS_COLLECTION_NAME)
        print(f"Fact collection '{FACTS_COLLECTION_NAME}' loaded/created. Initial count: {_facts_collection.count()}")
        _summaries_collection = _chroma_client.get_or_create_collection(name=SUMMARIES_COLLECTION_NAME)
        print(f"Summaries collection '{SUMMARIES_COLLECTION_NAME}' loaded/created. Initial count: {_summaries_collection.count()}")
    except Exception as e:
        st.error(f"CRITICAL Error initializing ChromaDB collections: {e}")
        st.stop()
    return _chroma_client, _facts_collection, _summaries_collection

# --- MODIFIED: LangGraph State Definition ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    long_term_memory_session_log: List[str]
    episodic_memory_session_log: List[str]
    user_persona: Union[dict, str, None]
    user_input: str
    router_decision: str
    retrieved_context: str
    turn_count: int
    # --- NEW: Field to hold result from the calendar tool ---
    tool_result: Union[str, None]

# --- Helper Functions (Your original code, unchanged) ---
def format_messages_for_llm(messages: List[BaseMessage], max_history=10) -> str:
    # ... (Your original format_messages_for_llm function is perfect, no changes needed)
    formatted = []
    start_index = max(0, len(messages) - max_history)
    for msg in messages[start_index:]:
        if isinstance(msg, HumanMessage):
            formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"AI: {msg.content}")
    return "\n".join(formatted)

def format_persona_for_prompt(persona_data: Union[dict, str, None]) -> str:
    # ... (Your original format_persona_for_prompt function is perfect, no changes needed)
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

# --- Agent Nodes (Original, Modified, and New) ---

def entry_node(state: AgentState) -> dict:
    """Resets per-turn state variables."""
    print("\n--- Entry Node ---")
    new_turn_count = state.get('turn_count', 0) + 1
    print(f"  Turn count: {new_turn_count}")
    # Reset transient state for the new turn
    return {"turn_count": new_turn_count, "retrieved_context": "", "tool_result": None}

def fact_extraction_node(state: AgentState) -> dict:
    """This runs on every turn to capture new facts. Unchanged."""
    print("--- Fact Extraction Node ---")
    # ... (Your original fact_extraction_node is perfect, no changes needed here)
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


# --- MODIFIED: The router now has three choices ---
import re # Make sure 'import re' is at the top of your script

def router_node(state: AgentState) -> dict:
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
    """
    Retrieves memories from ChromaDB. This version includes a query rewriting
    step to improve the accuracy of the vector search.
    """
    print("--- Retrieve Memory Node (RAG) ---")
    if state.get("router_decision") != RETRIEVE_ACTION:
        return {"retrieved_context": ""}
    
    current_user_query = state["user_input"]
    if not current_user_query:
        return {"retrieved_context": ""}

    # --- 1. QUERY REWRITING STEP ---
    print("  >> Rewriting user query for better retrieval...")
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at rewriting a user's question into a concise, keyword-focused search query for a vector database. "
                   "Focus on the core nouns and topics. Do not answer the question, just provide the ideal search query. "
                   "For example, if the user asks 'when was my meeting about electronics', the best query is 'user's meeting about electronics'."),
        ("human", "Rewrite the following user question into a search query: '{question}'")
    ])
    
    # Use the low-temperature router_llm for this deterministic task
    query_rewriter_chain = rewrite_prompt | router_llm
    try:
        rewritten_query_response = query_rewriter_chain.invoke({"question": current_user_query})
        rewritten_query = rewritten_query_response.content.strip()
        print(f"  Rewritten search query: '{rewritten_query}'")
    except Exception as e:
        print(f"  Error during query rewriting, falling back to original query. Error: {e}")
        rewritten_query = current_user_query

    # --- 2. EMBED AND SEARCH using the rewritten query ---
    print(f"  Attempting to retrieve relevant memories for query: '{rewritten_query}'")
    try:
        query_embedding = embedding_model.embed_query(rewritten_query)

        retrieved_context_parts = []
        # Query the facts collection
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

        # Query the summaries collection
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


# --- NEW: The node that executes the calendar tool ---
def calendar_tool_node(state: AgentState) -> dict:
    """Executes the calendar tool and puts the result in the state."""
    print("--- Calendar Tool Node ---")
    user_input = state["user_input"]
    conversation_history = state["messages"]
    result_string = schedule_event(user_input, conversation_history)
    print(f"  Calendar tool result: {result_string}")
    return {"tool_result": result_string}

# --- MODIFIED: The generation node is now aware of all contexts ---
def generate_response_node(state: AgentState) -> dict:
    """Generates a response, maintaining persona, using any available context."""
    print("--- Generate Response Node ---")
    user_input = state["user_input"]
    
    # Check for the context that determines the agent's mode for this turn.
    tool_result = state.get("tool_result")
    retrieved_context_str = state.get("retrieved_context")

    # --- NEW, SIMPLIFIED LOGIC FLOW ---

    if retrieved_context_str:
        # --- DEDICATED QA MODE ---
        # In this mode, we IGNORE the general persona and chat history to focus the LLM.
        print("  >> Entering Focused Question-Answering Mode.")
        system_prompt_content = (
            "You are a Question-Answering engine. Your sole task is to answer the user's question based on the provided 'Context'.\n"
            "1. Analyze the user's question.\n"
            "2. Find the specific answer within the 'Context'.\n"
            "3. State the answer clearly and concisely. Begin your response with a phrase like 'Based on my notes,' or 'I found a note that says...'.\n"
            "4. If the context does not contain the answer, state 'I couldn't find a specific answer to your question in my notes.'\n"
            "5. Do not add any conversational fluff, ask follow-up questions, or use information outside the provided 'Context'.\n\n"
            f"--- Context ---\n{retrieved_context_str}\n----------------"
        )
        prompt_parts = [
            SystemMessage(content=system_prompt_content),
            HumanMessage(content=user_input) # The user's direct question
        ]
    else:
        # --- REGULAR CONVERSATIONAL MODE (TOOL-USE OR GENERAL CHAT) ---
        print("  >> Entering General Conversational Mode.")
        # Start with the core persona.
        system_prompt_content = (
            f"You are the '{AGENT_NAME}', a kind, patient, and empathetic AI companion. "
            "Use clear, simple language and a respectful, positive tone."
        )

        user_persona_data = state.get("user_persona")
        formatted_user_persona = format_persona_for_prompt(user_persona_data)
        if formatted_user_persona:
            system_prompt_content += f"\n\nYou are assisting a specific user:\n{formatted_user_persona}"
        
        if tool_result:
            # Handle tool result confirmation
            system_prompt_content += (
                "\n\nYOUR CURRENT TASK: You have just used a tool. "
                f"The result was: '{tool_result}'.\n"
                "Your most important job is to clearly inform the user about this outcome."
            )
        else:
            # Handle general chat
            system_prompt_content += "\n\nYOUR CURRENT TASK: Engage in a helpful and empathetic conversation."

        prompt_parts = [SystemMessage(content=system_prompt_content.strip())]
        prompt_parts.extend(state["messages"])
        prompt_parts.append(HumanMessage(content=user_input))

    # The rest of the function remains the same
    try:
        response = main_llm.invoke(prompt_parts)
        ai_response_content = response.content
    except Exception as e:
        ai_response_content = f"I'm sorry, I encountered an error: {e}"
    
    print(f"  AI Response: {ai_response_content}")
    updated_messages = add_messages(state["messages"], [HumanMessage(content=user_input), AIMessage(content=ai_response_content)])
    return {"messages": updated_messages, "user_input": ""}

# --- FIXED: Your original summarizer node, now correctly included ---
def check_and_summarize_node(state: AgentState) -> dict:
    """This is your original summarization node. Unchanged."""
    print("--- Check and Summarize Node ---")
    # ... (Your original check_and_summarize_node is perfect, no changes needed here)
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
                print("  ChromaDB summaries_collection not available. Summary not persisted.")
        else:
            print("  Generated an empty or non-substantive summary.")
    else:
        print(f"  Turn {turn_count}, no summary needed yet (interval {SUMMARY_INTERVAL}).")
    return updated_state_dict


# --- MODIFIED: Graph Definition with the new tool path ---
@st.cache_resource
def get_compiled_app():
    print(f"Compiling LangGraph app for {AGENT_NAME}...")
    workflow = StateGraph(AgentState)
    workflow.add_node("entry", entry_node)
    workflow.add_node("extract_fact", fact_extraction_node)
    workflow.add_node("router", router_node)
    workflow.add_node("retrieve_memory", retrieve_memory_node)
    workflow.add_node("calendar_tool", calendar_tool_node) # NEW
    workflow.add_node("generate_response", generate_response_node)
    workflow.add_node("check_summarize", check_and_summarize_node)
    
    workflow.set_entry_point("entry")
    workflow.add_edge("entry", "extract_fact")
    workflow.add_edge("extract_fact", "router")
    
    def decide_action_path(state: AgentState):
        decision = state.get("router_decision")
        if decision == RETRIEVE_ACTION:
            return "retrieve_memory"
        if decision == CALENDAR_ACTION:
            return "calendar_tool"
        return "generate_response" # Default to generation
    
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
    workflow.add_edge("calendar_tool", "generate_response") # New path to generation
    workflow.add_edge("generate_response", "check_summarize")
    workflow.add_edge("check_summarize", END)
    
    _app = workflow.compile()
    print("LangGraph app compiled.")
    return _app

# --- User Persona (Your original code, unchanged) ---
@st.cache_data
def get_user_persona():
    # ... (Your original get_user_persona function is perfect, no changes needed)
    return {
        "name": "Aswin",
        "age_group": "Elderly (70s)",
        "preferred_language": "English",
        "background": "Retired history teacher, loves sharing stories from his past.",
        "interests": ["history", "watching old movies", "woodworking", "cricket"],
        "communication_style_preference": "respectful, enjoys a good chat, appreciates when his experiences are acknowledged.",
        "technology_use": "Uses a tablet for news and games.",
        "goals_with_agent": "discuss topics of interest, reminisce,to be a friend, get help finding information online, light-hearted conversation, feel understood and less lonely."
    }

# --- Helper Functions to Fetch Chroma Data for Display (Your original code, unchanged) ---
def get_chroma_facts_for_display(limit=10):
    # ... (Your original get_chroma_facts_for_display function is perfect, no changes needed)
    if facts_collection and facts_collection.count() > 0:
        try:
            results = facts_collection.get(limit=min(limit, facts_collection.count()), include=["documents", "metadatas"])
            return results
        except Exception as e:
            print(f"Error fetching facts for display: {e}")
            return {"documents": [f"Error fetching facts: {e}"], "metadatas":[{}]}
    return {"documents": [], "metadatas":[]}

def get_chroma_summaries_for_display(limit=10):
    # ... (Your original get_chroma_summaries_for_display function is perfect, no changes needed)
    if summaries_collection and summaries_collection.count() > 0:
        try:
            results = summaries_collection.get(limit=min(limit, summaries_collection.count()), include=["documents", "metadatas"])
            return results
        except Exception as e:
            print(f"Error fetching summaries for display: {e}")
            return {"documents": [f"Error fetching summaries: {e}"], "metadatas":[{}]}
    return {"documents": [], "metadatas":[]}


# --- SCRIPT EXECUTION STARTS HERE ---
router_llm, main_llm, summarizer_llm, fact_extractor_llm = get_llms()
embedding_model = get_embedding_model()
chroma_client, facts_collection, summaries_collection = get_chroma_collections()
app = get_compiled_app()
user_persona_data = get_user_persona()

# --- Streamlit UI (Your original code, with one minor addition for the new state field) ---

# --- MODIFIED: Added tool_result to the initial state ---
if "agent_state" not in st.session_state:
    st.session_state.agent_state = {
        "messages": [], "long_term_memory_session_log": [], "episodic_memory_session_log": [],
        "user_persona": user_persona_data, "user_input": "", "turn_count": 0,
        "router_decision": "", "retrieved_context": "", "tool_result": None
    }
    
# ... The rest of your Streamlit UI code is perfect and does not need to be changed.
# It will work seamlessly with the updated graph logic. I'm including it for completeness.

if "chat_display_messages" not in st.session_state:
    st.session_state.chat_display_messages = []
if "facts_to_display_data" not in st.session_state:
    st.session_state.facts_to_display_data = {"documents": [], "metadatas":[]}
if "summaries_to_display_data" not in st.session_state:
    st.session_state.summaries_to_display_data = {"documents": [], "metadatas":[]}
if "facts_to_display_loaded_once" not in st.session_state:
    st.session_state.facts_to_display_loaded_once = False
if "summaries_to_display_loaded_once" not in st.session_state:
    st.session_state.summaries_to_display_loaded_once = False


with st.sidebar:
    st.header(f"{AGENT_NAME} Status")
    with st.expander("👤 User Persona", expanded=False):
        current_persona = st.session_state.agent_state.get("user_persona")
        if isinstance(current_persona, dict):
            for key, value in current_persona.items():
                st.markdown(f"**{key.replace('_',' ').capitalize()}:** {value}")
        elif isinstance(current_persona, str):
            st.write(current_persona)
        else:
            st.write("No user persona loaded.")
    st.markdown("---")
    with st.expander("🔍 Retrieved Context (Current Turn)", expanded=False):
        retrieved_for_turn = st.session_state.agent_state.get("retrieved_context", "")
        if retrieved_for_turn:
            st.markdown(retrieved_for_turn)
        else:
            st.write("No context retrieved for this turn or retrieval was skipped.")
    st.markdown("---")
    st.subheader("Persistent Memory Browser")

    facts_count_val = facts_collection.count() if facts_collection else 0
    with st.expander(f"📖 Chroma Facts ({facts_count_val})", expanded=False):
        if facts_count_val > 0:
            if facts_count_val == 1:
                st.write("Showing 1 fact (the only one available):")
                num_f_disp_val = 1
            else:
                num_f_disp_val = st.slider(
                    "Max facts to show", 
                    min_value=1, 
                    max_value=facts_count_val,
                    value=min(5, facts_count_val), 
                    key="fact_slider_final_unique" # Ensure unique key
                )
            
            if st.button("Refresh Facts", key="refresh_facts_btn_final_unique"): # Ensure unique key
                st.session_state.facts_to_display_data = get_chroma_facts_for_display(limit=num_f_disp_val)
                st.session_state.facts_to_display_loaded_once = True
            
            data_to_show = st.session_state.facts_to_display_data
            # Attempt initial load if button not pressed and no data yet
            if not st.session_state.facts_to_display_loaded_once and not data_to_show.get("documents"):
                data_to_show = get_chroma_facts_for_display(limit=num_f_disp_val)

            docs, metas = data_to_show.get("documents", []), data_to_show.get("metadatas", [])
            if docs:
                for i, doc in enumerate(docs):
                    st.markdown(f"**Fact {i+1}:**")
                    st.code(doc, language=None)
                    if metas and i < len(metas) and metas[i]:
                        st.caption(f"Metadata: {metas[i]}")
            elif st.session_state.facts_to_display_loaded_once: # If button was clicked and list is empty
                 st.write("No facts found or an error occurred during fetch.")
            # No "else" needed for initial state as it implies waiting for button or auto-load
            
        else:
            st.write("No facts stored in ChromaDB.")

    summaries_count_val = summaries_collection.count() if summaries_collection else 0
    with st.expander(f"📑 Chroma Summaries ({summaries_count_val})", expanded=False):
        if summaries_count_val > 0:
            if summaries_count_val == 1:
                st.write("Showing 1 summary (the only one available):")
                num_s_disp_val = 1
            else:
                num_s_disp_val = st.slider(
                    "Max summaries to show", 
                    1, 
                    summaries_count_val, 
                    min(3, summaries_count_val), 
                    key="summary_slider_final_unique" # Ensure unique key
                )

            if st.button("Refresh Summaries", key="refresh_summaries_btn_final_unique"): # Ensure unique key
                st.session_state.summaries_to_display_data = get_chroma_summaries_for_display(limit=num_s_disp_val)
                st.session_state.summaries_to_display_loaded_once = True

            data_to_show = st.session_state.summaries_to_display_data
            if not st.session_state.summaries_to_display_loaded_once and not data_to_show.get("documents"):
                data_to_show = get_chroma_summaries_for_display(limit=num_s_disp_val)

            docs, metas = data_to_show.get("documents", []), data_to_show.get("metadatas", [])
            if docs:
                for i, doc in enumerate(docs):
                    st.markdown(f"**Summary {i+1}:**")
                    st.text_area(f"sum_disp_f_unique_{i}", doc, height=100, disabled=True, key=f"sum_txt_f_final_{i}") # Ensure unique key
                    if metas and i < len(metas) and metas[i]:
                        st.caption(f"Metadata: {metas[i]}")
            elif st.session_state.summaries_to_display_loaded_once:
                 st.write("No summaries found or an error occurred during fetch.")
            
        else:
            st.write("No summaries stored in ChromaDB.")

    st.markdown("---")
    st.caption(f"Agent Turn Count: {st.session_state.agent_state.get('turn_count', 0)}")
    st.subheader("Session Memory Logs")
    st.write(f"Facts added this session: {len(st.session_state.agent_state.get('episodic_memory_session_log',[]))}")
    st.write(f"Summaries added this session: {len(st.session_state.agent_state.get('long_term_memory_session_log',[]))}")

    if st.button("Reset Agent State & Chat"):
        st.session_state.agent_state = {
            "messages": [], 
            "long_term_memory_session_log": [], "episodic_memory_session_log": [],
            "user_persona": user_persona_data, "user_input": "", "turn_count": 0,
            "router_decision": "", "retrieved_context": "", "tool_result": None
        }
        st.session_state.chat_display_messages = []
        st.session_state.facts_to_display_data = {"documents":[], "metadatas":[]}
        st.session_state.summaries_to_display_data = {"documents":[], "metadatas":[]}
        st.session_state.facts_to_display_loaded_once = False
        st.session_state.summaries_to_display_loaded_once = False
        st.rerun()

# Main chat interface
st.title(f"💬 {AGENT_NAME}")

for msg_info in st.session_state.chat_display_messages:
    with st.chat_message(msg_info["role"]):
        st.markdown(msg_info["content"])

if prompt := st.chat_input(f"Chat with {AGENT_NAME}..."):
    st.session_state.chat_display_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    current_graph_input_state = st.session_state.agent_state.copy()
    current_graph_input_state["user_input"] = prompt
    
    with st.spinner(f"{AGENT_NAME} is thinking..."):
        try:
            # We invoke the app with the current state. The 'entry_node' will handle resetting
            # transient fields like 'retrieved_context' and 'tool_result' for this turn.
            updated_graph_output_state = app.invoke(current_graph_input_state)
            st.session_state.agent_state = updated_graph_output_state
            
            ai_message_object = None
            if updated_graph_output_state.get("messages"):
                last_message_in_graph = updated_graph_output_state["messages"][-1]
                if isinstance(last_message_in_graph, AIMessage):
                    ai_message_object = last_message_in_graph
            
            if ai_message_object:
                ai_response_content = ai_message_object.content
                st.session_state.chat_display_messages.append({"role": "assistant", "content": ai_response_content})
                with st.chat_message("assistant"):
                    st.markdown(ai_response_content)
            else:
                st.error(f"{AGENT_NAME} did not return a valid response message.")
                st.session_state.chat_display_messages.append({"role":"assistant","content":"Sorry, I had trouble generating a response."})
        except Exception as e:
            st.error(f"Error invoking {AGENT_NAME}: {e}")
            st.session_state.chat_display_messages.append({"role":"assistant","content":f"An error occurred: {e}"})