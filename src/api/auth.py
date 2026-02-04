"""
Authentication and OAuth integration for PaperChat.
Supports Google, Microsoft (Azure AD), and GitHub OAuth providers.
"""

import os
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from authlib.integrations.starlette_client import OAuth
from pydantic import BaseModel
import redis

logger = logging.getLogger(__name__)

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15  # 15 minutes
REFRESH_TOKEN_EXPIRE_MINUTES = 60  # 1 hour (inactivity timeout)
INACTIVITY_TIMEOUT_MINUTES = 60  # Force re-login after 60 min inactivity

# OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
MICROSOFT_CLIENT_ID = os.getenv("MICROSOFT_CLIENT_ID", "")
MICROSOFT_CLIENT_SECRET = os.getenv("MICROSOFT_CLIENT_SECRET", "")
GITHUB_CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")

# Callback URL base (set via environment for production)
CALLBACK_URL_BASE = os.getenv("CALLBACK_URL_BASE", "http://localhost:7860")

# Redis for session storage
REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

security = HTTPBearer(auto_error=False)


class User(BaseModel):
    """User model"""
    user_id: str
    email: str
    name: str
    picture: Optional[str] = None
    provider: str  # google, microsoft, github


class TokenData(BaseModel):
    """JWT token payload"""
    user_id: str
    email: str


def get_redis_client() -> redis.Redis:
    """Get Redis client for session storage"""
    return redis.Redis.from_url(REDIS_URL, decode_responses=True)


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    
    return encoded_jwt


def create_refresh_token(data: Dict[str, Any]) -> str:
    """Create JWT refresh token with longer expiry"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[TokenData]:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        email: str = payload.get("email")
        
        if user_id is None or email is None:
            return None
        
        return TokenData(user_id=user_id, email=email)
    except JWTError as e:
        logger.warning("JWT verification failed: %s", e)
        return None


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """
    Get current authenticated user from JWT token.
    Returns None if no token or invalid token (for optional auth).
    """
    if not credentials:
        return None
    
    token_data = verify_token(credentials.credentials)
    if not token_data:
        return None
    
    # Get user from Redis cache
    redis_client = get_redis_client()
    user_key = f"user:{token_data.user_id}"
    user_data = redis_client.get(user_key)
    
    if not user_data:
        return None
    
    import json
    user_dict = json.loads(user_data)
    
    # Check inactivity timeout
    last_activity_key = f"activity:{token_data.user_id}"
    last_activity = redis_client.get(last_activity_key)
    
    if last_activity:
        last_activity_time = datetime.fromisoformat(last_activity)
        inactive_minutes = (datetime.utcnow() - last_activity_time).total_seconds() / 60
        
        if inactive_minutes > INACTIVITY_TIMEOUT_MINUTES:
            # Session expired due to inactivity
            logger.info("Session expired due to %d minutes of inactivity", int(inactive_minutes))
            redis_client.delete(user_key)
            redis_client.delete(last_activity_key)
            redis_client.delete(f"refresh:{token_data.user_id}")
            return None
    
    # Update last activity timestamp
    redis_client.setex(
        last_activity_key,
        REFRESH_TOKEN_EXPIRE_MINUTES * 60,
        datetime.utcnow().isoformat()
    )
    
    return User(**user_dict)


async def require_auth(user: Optional[User] = Depends(get_current_user)) -> User:
    """
    Require authentication - raises 401 if user not authenticated.
    Use this as a dependency for protected endpoints.
    """
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def save_user_session(user: User, access_token: str, refresh_token: str) -> None:
    """Save user session to Redis with refresh token"""
    redis_client = get_redis_client()
    user_key = f"user:{user.user_id}"
    refresh_key = f"refresh:{user.user_id}"
    activity_key = f"activity:{user.user_id}"
    
    import json
    user_json = user.model_dump_json()
    
    # Store user data (expires with refresh token)
    redis_client.setex(user_key, REFRESH_TOKEN_EXPIRE_MINUTES * 60, user_json)
    
    # Store refresh token
    redis_client.setex(refresh_key, REFRESH_TOKEN_EXPIRE_MINUTES * 60, refresh_token)
    
    # Initialize last activity timestamp
    redis_client.setex(
        activity_key,
        REFRESH_TOKEN_EXPIRE_MINUTES * 60,
        datetime.utcnow().isoformat()
    )
    
    logger.info("Saved session for user: %s", user.email)


def revoke_user_session(user_id: str) -> None:
    """Revoke user session (logout) - delete all session data"""
    redis_client = get_redis_client()
    user_key = f"user:{user_id}"
    refresh_key = f"refresh:{user_id}"
    activity_key = f"activity:{user_id}"
    
    redis_client.delete(user_key)
    redis_client.delete(refresh_key)
    redis_client.delete(activity_key)
    
    logger.info("Revoked session for user: %s", user_id)


# OAuth Setup
oauth = OAuth()

# Google OAuth
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name='google',
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={'scope': 'openid email profile'},
    )

# Microsoft OAuth (Azure AD)
if MICROSOFT_CLIENT_ID and MICROSOFT_CLIENT_SECRET:
    oauth.register(
        name='microsoft',
        client_id=MICROSOFT_CLIENT_ID,
        client_secret=MICROSOFT_CLIENT_SECRET,
        authorize_url='https://login.microsoftonline.com/common/oauth2/v2.0/authorize',
        authorize_params=None,
        access_token_url='https://login.microsoftonline.com/common/oauth2/v2.0/token',
        access_token_params=None,
        refresh_token_url=None,
        client_kwargs={'scope': 'openid email profile'},
    )

# GitHub OAuth
if GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET:
    oauth.register(
        name='github',
        client_id=GITHUB_CLIENT_ID,
        client_secret=GITHUB_CLIENT_SECRET,
        access_token_url='https://github.com/login/oauth/access_token',
        access_token_params=None,
        authorize_url='https://github.com/login/oauth/authorize',
        authorize_params=None,
        api_base_url='https://api.github.com/',
        client_kwargs={'scope': 'user:email'},
    )


async def get_oauth_user(provider: str, token: Dict) -> User:
    """Get user info from OAuth provider"""
    if provider == 'google':
        # Google provides user info in the token
        user_info = token.get('userinfo', {})
        return User(
            user_id=f"google:{user_info['sub']}",
            email=user_info['email'],
            name=user_info.get('name', user_info['email']),
            picture=user_info.get('picture'),
            provider='google'
        )
    
    elif provider == 'microsoft':
        # Microsoft also provides user info in token
        user_info = token.get('userinfo', {})
        return User(
            user_id=f"microsoft:{user_info['sub']}",
            email=user_info['email'],
            name=user_info.get('name', user_info['email']),
            picture=None,
            provider='microsoft'
        )
    
    elif provider == 'github':
        # GitHub requires additional API call for email
        client = oauth.create_client(provider)
        
        # Get user info
        resp = await client.get('user', token=token)
        github_user = resp.json()
        
        # Get email (may need separate API call)
        email = github_user.get('email')
        if not email:
            resp = await client.get('user/emails', token=token)
            emails = resp.json()
            # Get primary email
            for e in emails:
                if e.get('primary'):
                    email = e['email']
                    break
            if not email and emails:
                email = emails[0]['email']
        
        return User(
            user_id=f"github:{github_user['id']}",
            email=email or f"{github_user['login']}@github.local",
            name=github_user.get('name') or github_user['login'],
            picture=github_user.get('avatar_url'),
            provider='github'
        )
    
    else:
        raise ValueError(f"Unknown provider: {provider}")
