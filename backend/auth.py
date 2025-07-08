import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# This is the same SCOPES you use in your tool. It MUST match.
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]
CREDENTIALS_FILE = "credentials.json"
TOKEN_FILE = "token.json"

def run_authorization():
    """
    Runs the interactive authorization flow to generate a token.json file.
    """
    creds = None
    
    # Check if a token already exists and is valid
    if os.path.exists(TOKEN_FILE):
        print(f"'{TOKEN_FILE}' already exists. Checking if it's valid...")
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            if creds.valid:
                print("Token is valid. No action needed.")
                return
            if creds.expired and creds.refresh_token:
                print("Token is expired, attempting to refresh...")
                creds.refresh(Request())
                print("Token refreshed successfully.")
                # Save the refreshed credentials
                with open(TOKEN_FILE, "w") as token:
                    token.write(creds.to_json())
                print(f"Refreshed token saved to '{TOKEN_FILE}'.")
                return
        except Exception as e:
            print(f"Could not use existing token, it may be invalid. Error: {e}")
            print("Proceeding to re-authorize.")
            creds = None


    # If there are no (valid) credentials available, let the user log in.
    if not creds:
        print("No valid token found. Starting authorization flow...")
        if not os.path.exists(CREDENTIALS_FILE):
            print(f"CRITICAL ERROR: '{CREDENTIALS_FILE}' not found.")
            print("Please make sure you have downloaded your OAuth 2.0 Client ID credentials from Google Cloud Console.")
            return

        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        
        # This line will open a web browser for you to log in.
        creds = flow.run_local_server(port=0)
        
        # Save the credentials for the main app to use
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
            
        print(f"\nAuthorization successful! Credentials saved to '{TOKEN_FILE}'.")
        print("You can now start the main FastAPI server.")

if __name__ == "__main__":
    run_authorization()