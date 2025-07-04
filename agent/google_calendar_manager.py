import os.path
import datetime as dt
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import streamlit as st

SCOPES = ["https://www.googleapis.com/auth/calendar"]
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"

def is_connected():
    return os.path.exists(TOKEN_FILE)

@st.cache_resource
def get_calendar_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                creds = None
        
        if not creds:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
            
    try:
        service = build("calendar", "v3", credentials=creds)
        return service
    except HttpError as error:
        return None

# <-- CRITICAL FIX: This function now correctly handles all-day vs. timed events -->
def create_calendar_event(service, summary: str, start_time: str, end_time: str, description: str = "", **kwargs):
    if not service: return "Error: Calendar service not available."

    # Determine if it's an all-day event or a timed event based on string format
    if 'T' in start_time: # It's a dateTime event
        start_payload = {'dateTime': start_time, 'timeZone': 'UTC'}
        end_payload = {'dateTime': end_time, 'timeZone': 'UTC'}
    else: # It's an all-day event
        start_payload = {'date': start_time}
        # For all-day events, the end date is exclusive. We need to add one day.
        end_date_obj = dt.datetime.strptime(end_time, '%Y-%m-%d').date() + dt.timedelta(days=1)
        end_payload = {'date': end_date_obj.strftime('%Y-%m-%d')}

    event = {
        'summary': summary,
        'description': description,
        'start': start_payload,
        'end': end_payload
    }
    
    try:
        print(f"Attempting to create event with payload: {event}")
        created_event = service.events().insert(calendarId='primary', body=event).execute()
        print("Event created successfully in Google Calendar.")
        return f"Event '{summary}' created successfully! View it here: {created_event.get('htmlLink')}"
    except HttpError as error:
        print(f"Google API Error: {error}")
        return f"A Google Calendar API error occurred: {error}"
    except Exception as e:
        print(f"Generic Error in create_calendar_event: {e}")
        return f"A generic error occurred: {e}"

def get_upcoming_events(service, max_results=10):
    if not service:
        return []
    try:
        now = dt.datetime.utcnow().isoformat() + "Z"
        events_result = service.events().list(calendarId="primary", timeMin=now, maxResults=max_results, singleEvents=True, orderBy="startTime").execute()
        return events_result.get("items", [])
    except HttpError:
        return []