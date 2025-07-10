import re
import dateparser
from datetime import datetime, timedelta
from typing import List

# LangChain imports are now needed here
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

# Other imports
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import os
import json

# --- Configuration ---
TOOL_LLM = ChatOllama(model="mistral:7b-instruct-q4_K_M", temperature=0.0)
TOKEN_FILE = "token.json"
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

# --- Helper Functions ---
def _format_history(messages: List[BaseMessage]) -> str:
    """Helper to format message history for the prompt."""
    formatted = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            formatted.append(f"AI: {msg.content}")
    return "\n".join(formatted)

def get_calendar_service():
    """Initializes the Google Calendar API service."""
    if not os.path.exists(TOKEN_FILE):
        return None, f"Error: Authorization file '{TOKEN_FILE}' not found."
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        # Add credential refresh logic if needed
        service = build('calendar', 'v3', credentials=creds)
        return service, None
    except Exception as e:
        return None, f"Error initializing Google Calendar service: {e}"

# --- Main Tool Function ---
def schedule_event(user_input: str, conversation_history: List[BaseMessage]) -> str:
    """
    Takes a user's request and conversation history, extracts details, and creates a calendar event.
    """
    print("--- Invoking Context-Aware Calendar Tool ---")
    
    service, error = get_calendar_service()
    if error:
        return error

    history_str = _format_history(conversation_history)

    # The prompt now includes the conversation history
    prompt_template = PromptTemplate.from_template(
        """You are an expert assistant that extracts event details from a user's request, using the conversation history for context.

        Your goal is to populate a JSON object with:
        - 'title': The core subject of the event.
        - 'date': The specific part of the request describing the date (e.g., "next Friday", "tomorrow", "July 3rd").
        - 'time': The specific part of the request describing the time (e.g., "evening", "4pm", "10:30"). If no time is mentioned, provide an empty string.
        - 'location': The location, if mentioned.

        You MUST use the "Recent Conversation History" to resolve context.

        --- Recent Conversation History ---
        {history}
        ---------------------------------

        --- User's latest request to process ---
        "{user_input}"
        --------------------------------------

        JSON Output:"""
    )
    
    chain = prompt_template | TOOL_LLM
    try:
        response = chain.invoke({"user_input": user_input, "history": history_str})
        llm_output = response.content
        json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if not json_match:
            return "Sorry, I couldn't understand the event details clearly enough to schedule it."
        
        event_data = json.loads(json_match.group(0))
        print(f"  Context-aware extracted data: {event_data}")

        if not event_data.get("title") or not event_data.get("date"):
            return "Error: Could not figure out the title or the date from our conversation."

    except Exception as e:
        return f"I had trouble understanding the event details. Error: {e}"

    try:
        date_part = event_data['date']
        time_part = event_data.get('time') # Get the time part

        # Explicitly check if the time part is empty or just whitespace
        if not time_part or not time_part.strip():
            time_part = '10:00 AM' # If it's empty, set a reasonable default
            print(f"  No time provided, defaulting to {time_part}.")
        
        # Handle fuzzy time terms like "evening"
        fuzzy_map = {"morning": "9am", "afternoon": "2pm", "evening": "6pm", "noon": "12pm"}
        time_part = fuzzy_map.get(time_part.lower().strip(), time_part)
        
        # Combine the parts into one string that dateparser can reliably handle
        combined_string = f"{date_part} at {time_part}"
        print(f"  Combined string for parsing: '{combined_string}'")

        current_year = datetime.now().year
        settings = {
            'PREFER_DATES_FROM': 'future',
            'RELATIVE_BASE': datetime(current_year, 1, 1)
        }

        start_dt = dateparser.parse(combined_string, settings=settings)
        
        if not start_dt:
            raise ValueError(f"Could not parse the combined date and time: '{combined_string}'")
        
        print(f"  Parsed datetime object: {start_dt}")
        
        end_dt = start_dt + timedelta(hours=1)
        
    except (ValueError, TypeError) as e:
        return f"I couldn't figure out the exact date or time for the event. Error: {e}"

    event_payload = {
        'summary': event_data['title'],
        'location': event_data.get('location', ''),
        'description': f"Added by agent from request: '{user_input}'",
        'start': {'dateTime': start_dt.isoformat(), 'timeZone': 'UTC'},
        'end': {'dateTime': end_dt.isoformat(), 'timeZone': 'UTC'},
    }
    
    try:
        service.events().insert(calendarId='primary', body=event_payload).execute()
        return f"Success! I've added '{event_data['title']}' to your Google Calendar schedule."
    except Exception as e:
        return f"I failed to create the event in your calendar. Error: {e}"