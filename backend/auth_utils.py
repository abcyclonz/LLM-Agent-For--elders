import hashlib
import hmac
import os
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
import uuid

# Password hashing using SHA256 with salt
def get_password_hash(password: str) -> str:
    """Hash a password using SHA256 with salt"""
    salt = os.urandom(16).hex()
    hash_obj = hashlib.sha256()
    hash_obj.update((password + salt).encode('utf-8'))
    hashed = hash_obj.hexdigest()
    return f"{salt}${hashed}"

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    try:
        salt, hash_value = hashed_password.split('$', 1)
        hash_obj = hashlib.sha256()
        hash_obj.update((plain_password + salt).encode('utf-8'))
        computed_hash = hash_obj.hexdigest()
        print(f"DEBUG: salt={salt}, hash_value={hash_value}, computed_hash={computed_hash}")
        return hmac.compare_digest(computed_hash, hash_value)
    except Exception as e:
        print(f"DEBUG: Exception in verify_password: {e}")
        return False

# JWT settings
SECRET_KEY = "your-secret-key-here-change-in-production"  # Change this in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    """Verify and decode a JWT token"""
    try:
        print(f"[DEBUG] verify_token: token={token}")
        print(f"[DEBUG] verify_token: SECRET_KEY={SECRET_KEY}, ALGORITHM={ALGORITHM}")
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print(f"[DEBUG] verify_token: decoded payload={payload}")
        return payload
    except JWTError as e:
        print(f"[ERROR] verify_token: JWTError: {e}")
        return None
    except Exception as e:
        print(f"[ERROR] verify_token: Exception: {e}")
        return None

def generate_user_id() -> str:
    """Generate a unique user ID"""
    return str(uuid.uuid4()) 