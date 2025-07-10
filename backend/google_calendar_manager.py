import os
import datetime as dt
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import json # Added for handling token.json more explicitly

# If you were to implement a full web flow:
# from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/calendar.events"] # Changed to .events for narrower scope
# Using .events is generally better if you only need to create/manage events.
# If you need to list/read all calendars, use "https://www.googleapis.com/auth/calendar.readonly" or "https://www.googleapis.com/auth/calendar"
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"

# Global variable to store the calendar service instance once initialized
_calendar_service = None

def is_calendar_connected():
    """Checks if a valid token.json exists, indicating a prior successful authorization."""
    return os.path.exists(TOKEN_FILE)

def get_calendar_service():
    """
    Initializes and returns the Google Calendar API service.
    This version focuses on using an existing token.json and refreshing it.
    It does NOT handle the initial interactive authorization flow in a backend context.
    For initial authorization, you would run a separate script or use a web flow.
    """
    global _calendar_service

    if _calendar_service:
        return _calendar_service, None # Return existing service if already initialized

    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception as e:
            print(f"Error loading token.json: {e}")
            return None, f"Error loading existing calendar credentials: {e}"

    if creds and creds.expired and creds.refresh_token:
        try:
            print("Refreshing Google Calendar credentials...")
            creds.refresh(Request())
            with open(TOKEN_FILE, "w") as token:
                token.write(creds.to_json())
            print("Credentials refreshed and saved.")
        except Exception as e:
            print(f"Error refreshing Google Calendar credentials: {e}")
            return None, f"Error refreshing Google Calendar credentials: {e}. Please re-authenticate."
    elif not creds or not creds.valid:
        # This branch means token.json is missing or invalid and cannot be refreshed.
        # In a backend, you typically wouldn't run run_local_server here.
        # You would log an error and require manual intervention or a web OAuth flow.
        print(f"No valid/refreshable '{TOKEN_FILE}' found. Attempting to load from credentials.json directly for a one-time flow or if using a service account.")
        
        # --- IMPORTANT ---
        # For a true backend, you would typically use a Service Account
        # or implement a full OAuth 2.0 Web Application flow (which involves user redirect).
        # This section below is *not* for production backend where user interaction is needed.
        # It's primarily for an existing token.json or a simplified headless setup.
        
        if os.path.exists(CREDENTIALS_FILE):
            try:
                # This part is complex for a backend.
                # If credentials.json is for a *service account*, you'd load it differently:
                # from google.oauth2 import service_account
                # creds = service_account.Credentials.from_service_account_file(
                #     CREDENTIALS_FILE, scopes=SCOPES
                # )
                
                # If credentials.json is for an *installed app* but you want headless auth:
                # This is still tricky. For initial auth, you'd generate token.json once
                # by running a script locally with run_local_server, and then copy token.json to server.
                
                return None, f"Calendar API not authorized. '{TOKEN_FILE}' missing or invalid. Please ensure authentication is set up."
            except Exception as e:
                return None, f"Error with credentials.json: {e}"
        else:
            return None, f"Calendar API not authorized. Neither '{TOKEN_FILE}' nor '{CREDENTIALS_FILE}' found."


    if not creds or not creds.valid:
        return None, "Google Calendar API credentials are not valid after all attempts."

    try:
        _calendar_service = build("calendar", "v3", credentials=creds)
        print("Google Calendar service initialized.")
        return _calendar_service, None
    except HttpError as error:
        print(f"Error building Google Calendar service: {error}")
        return None, f"Error building Google Calendar service: {error}"
    except Exception as e:
        print(f"Generic error initializing Google Calendar service: {e}")
        return None, f"Generic error initializing Google Calendar service: {e}"


def create_calendar_event(summary: str, start_time: str, end_time: str, description: str = "", location: str = "") -> str:
    """
    Creates a Google Calendar event.
    start_time and end_time should be ISO 8601 format.
    For all-day: 'YYYY-MM-DD'
    For timed: 'YYYY-MM-DDTHH:MM:SS' (e.g., '2025-07-05T16:00:00')
    """
    service, error = get_calendar_service()
    if error:
        return error

    # Determine if it's an all-day event or a timed event based on string format
    if 'T' in start_time: # It's a dateTime event
        start_payload = {'dateTime': start_time, 'timeZone': 'UTC'} # Ensure your tool passes UTC or appropriate TZ
        end_payload = {'dateTime': end_time, 'timeZone': 'UTC'}
    else: # It's an all-day event
        start_payload = {'date': start_time}
        # For all-day events, the end date is exclusive. We need to add one day to the end date provided by the tool.
        try:
            end_date_obj = dt.datetime.strptime(end_time, '%Y-%m-%d').date() + dt.timedelta(days=1)
            end_payload = {'date': end_date_obj.strftime('%Y-%m-%d')}
        except ValueError:
            return "Error: Invalid date format for all-day event. Expected YYYY-MM-DD."

    event = {
        'summary': summary,
        'description': description,
        'location': location, # Added location parameter
        'start': start_payload,
        'end': end_payload
    }

    try:
        print(f"Attempting to create event with payload: {event}")
        created_event = service.events().insert(calendarId='primary', body=event).execute()
        print("Event created successfully in Google Calendar.")
        return f"Event '{summary}' created successfully! View it here: {created_event.get('htmlLink')}"
    except HttpError as error:
        print(f"Google API Error: {error.content.decode()}") # Decode for readable error
        return f"A Google Calendar API error occurred: {error.status_code} - {error.content.decode()}"
    except Exception as e:
        print(f"Generic Error in create_calendar_event: {e}")
        return f"A generic error occurred: {e}"


def get_upcoming_events(max_results=10) -> list:
    """
    Retrieves upcoming events from the primary calendar.
    Returns a list of event dictionaries.
    """
    service, error = get_calendar_service()
    if error:
        print(f"Error retrieving calendar service: {error}")
        return []
    try:
        now = dt.datetime.utcnow().isoformat() + "Z" # 'Z' indicates UTC
        print(f"Fetching upcoming events from {now}...")
        events_result = service.events().list(
            calendarId="primary",
            timeMin=now,
            maxResults=max_results,
            singleEvents=True,
            orderBy="startTime"
        ).execute()
        items = events_result.get("items", [])
        print(f"Found {len(items)} upcoming events.")
        return items
    except HttpError as error:
        print(f"Google API Error fetching upcoming events: {error.content.decode()}")
        return []
    except Exception as e:
        print(f"Generic Error fetching upcoming events: {e}")
        return []

# Example of how you might use these functions in a backend script or a simple test
if __name__ == '__main__':
    print("--- Google Calendar Manager Test ---")

    # Test connection
    service, auth_error = get_calendar_service()
    if auth_error:
        print(f"Authentication Problem: {auth_error}")
        print("Please ensure you have a valid 'token.json' file generated from a prior OAuth flow.")
        print("For a backend, this often means generating it once locally and deploying it,")
        print("or using a Google Service Account.")
    else:
        print("\nGoogle Calendar service initialized successfully.")

        # Test creating a timed event (1 hour from now)
        now_utc = dt.datetime.utcnow()
        start_time_str = (now_utc + dt.timedelta(minutes=5)).isoformat(timespec='seconds')
        end_time_str = (now_utc + dt.timedelta(hours=1, minutes=5)).isoformat(timespec='seconds')
        
        print(f"\nAttempting to create a timed event from {start_time_str} to {end_time_str} UTC")
        result = create_calendar_event(
            summary="Test Timed Event from Backend",
            start_time=start_time_str,
            end_time=end_time_str,
            description="Created via FastAPI backend example."
        )
        print(f"Create Timed Event Result: {result}")

        # Test creating an all-day event (tomorrow)
        tomorrow_date = (dt.date.today() + dt.timedelta(days=1)).isoformat()
        print(f"\nAttempting to create an all-day event for {tomorrow_date}")
        result_all_day = create_calendar_event(
            summary="All-Day Test Event from Backend",
            start_time=tomorrow_date,
            end_time=tomorrow_date, # For all-day, start and end date are the same, internally +1 day for exclusive end
            description="All-day event via FastAPI backend example."
        )
        print(f"Create All-Day Event Result: {result_all_day}")


        # Test getting upcoming events
        print("\n--- Upcoming Events ---")
        upcoming_events = get_upcoming_events(max_results=5)
        if upcoming_events:
            for event in upcoming_events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                print(f"  {event['summary']} ({start})")
        else:
            print("No upcoming events found.")