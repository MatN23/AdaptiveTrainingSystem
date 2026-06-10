# Copyright (c) 2025 MatN23. All rights reserved.
# Src/Main_Scripts/security/auth.py
# Licensed under the Custom License below.

import hashlib
import secrets
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from dataclasses import dataclass
import jwt


@dataclass
class User:
    """User model with security features."""
    username: str
    password_hash: str
    salt: str
    created_at: datetime
    last_login: Optional[datetime] = None
    failed_login_attempts: int = 0
    is_locked: bool = False
    permissions: List[str] = None
    session_token: Optional[str] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = ['chat:basic']


class SecurityManager:
    """Comprehensive security manager for production deployment."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.secret_key = config.get('secret_key', self._generate_secret_key())
        self.session_timeout = config.get('session_timeout_hours', 24)
        self.max_login_attempts = config.get('max_login_attempts', 5)
        self.lockout_duration = config.get('lockout_duration_minutes', 30)
        
        # In-memory user store (use proper DB in production)
        self.users: Dict[str, User] = {}
        self.active_sessions: Dict[str, Dict] = {}
        
        # Rate limiting
        self.rate_limits: Dict[str, List[float]] = {}
        
        logging.info("Security manager initialized")
    
    def _generate_secret_key(self) -> str:
        """Generate cryptographically secure secret key."""
        return secrets.token_hex(32)
    
    def _hash_password(self, password: str, salt: str) -> str:
        """Hash password with salt using PBKDF2."""
        return hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # 100k iterations
        ).hex()
    
    def _generate_salt(self) -> str:
        """Generate cryptographically secure salt."""
        return secrets.token_hex(16)
    
    def create_user(self, username: str, password: str, permissions: List[str] = None) -> bool:
        """Create new user with secure password hashing."""
        # Input validation
        if not self._validate_username(username):
            return False
        
        if not self._validate_password(password):
            return False
        
        if username in self.users:
            logging.warning(f"Attempt to create duplicate user: {username}")
            return False
        
        salt = self._generate_salt()
        password_hash = self._hash_password(password, salt)
        
        user = User(
            username=username,
            password_hash=password_hash,
            salt=salt,
            created_at=datetime.now(),
            permissions=permissions or ['chat:basic']
        )
        
        self.users[username] = user
        logging.info(f"User created: {username}")
        return True
    
    def authenticate(self, username: str, password: str, client_ip: str) -> Optional[str]:
        """Authenticate user and return session token."""
        # Rate limiting check
        if not self._check_rate_limit(client_ip, 'login', 5, 300):  # 5 attempts per 5 minutes
            logging.warning(f"Rate limit exceeded for login from {client_ip}")
            return None
        
        # User existence check
        if username not in self.users:
            logging.warning(f"Login attempt for non-existent user: {username}")
            self._record_rate_limit_event(client_ip, 'login')
            return None
        
        user = self.users[username]
        
        # Account lockout check
        if user.is_locked:
            if self._is_lockout_expired(user):
                user.is_locked = False
                user.failed_login_attempts = 0
                logging.info(f"Account unlock due to timeout: {username}")
            else:
                logging.warning(f"Login attempt on locked account: {username}")
                return None
        
        # Password verification
        password_hash = self._hash_password(password, user.salt)
        if password_hash != user.password_hash:
            user.failed_login_attempts += 1
            
            if user.failed_login_attempts >= self.max_login_attempts:
                user.is_locked = True
                logging.warning(f"Account locked due to failed attempts: {username}")
            
            logging.warning(f"Failed login attempt for user: {username}")
            self._record_rate_limit_event(client_ip, 'login')
            return None
        
        # Successful authentication
        user.failed_login_attempts = 0
        user.last_login = datetime.now()
        
        session_token = self._generate_session_token(user)
        user.session_token = session_token
        
        # Store active session
        self.active_sessions[session_token] = {
            'username': username,
            'created_at': datetime.now(),
            'client_ip': client_ip,
            'permissions': user.permissions
        }
        
        logging.info(f"Successful login: {username} from {client_ip}")
        return session_token
    
    def _generate_session_token(self, user: User) -> str:
        """Generate JWT session token."""
        payload = {
            'username': user.username,
            'permissions': user.permissions,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(hours=self.session_timeout)
        }
        
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def validate_session(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate session token and return user info."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            if token not in self.active_sessions:
                return None
            
            session = self.active_sessions[token]
            return {
                'username': payload['username'],
                'permissions': payload['permissions'],
                'session_info': session
            }
            
        except jwt.ExpiredSignatureError:
            logging.info("Session token expired")
            if token in self.active_sessions:
                del self.active_sessions[token]
            return None
        except jwt.InvalidTokenError:
            logging.warning("Invalid session token")
            return None
    
    def logout(self, token: str) -> bool:
        """Logout user and invalidate session."""
        if token in self.active_sessions:
            username = self.active_sessions[token]['username']
            del self.active_sessions[token]
            
            # Clear user session token
            if username in self.users:
                self.users[username].session_token = None
            
            logging.info(f"User logged out: {username}")
            return True
        return False
    
    def _validate_username(self, username: str) -> bool:
        """Validate username format and security."""
        if not username or len(username) < 3 or len(username) > 50:
            return False
        
        # Allow only alphanumeric and safe characters
        allowed_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.')
        if not all(c in allowed_chars for c in username):
            return False
        
        return True
    
    def _validate_password(self, password: str) -> bool:
        """Validate password strength."""
        if len(password) < 8:
            return False
        
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password)
        
        return sum([has_upper, has_lower, has_digit, has_special]) >= 3
    
    def _is_lockout_expired(self, user: User) -> bool:
        """Check if account lockout has expired."""
        if not user.last_login:
            return False
        
        lockout_end = user.last_login + timedelta(minutes=self.lockout_duration)
        return datetime.now() > lockout_end
    
    def _check_rate_limit(self, client_ip: str, action: str, limit: int, window: int) -> bool:
        """Check if action is within rate limits."""
        key = f"{client_ip}:{action}"
        now = time.time()
        
        if key not in self.rate_limits:
            self.rate_limits[key] = []
        
        # Clean old entries
        self.rate_limits[key] = [t for t in self.rate_limits[key] if now - t < window]
        
        if len(self.rate_limits[key]) >= limit:
            return False
        
        return True
    
    def _record_rate_limit_event(self, client_ip: str, action: str):
        """Record a rate limit event."""
        key = f"{client_ip}:{action}"
        now = time.time()
        
        if key not in self.rate_limits:
            self.rate_limits[key] = []
        
        self.rate_limits[key].append(now)
    
    def check_permission(self, session_info: Dict, required_permission: str) -> bool:
        """Check if user has required permission."""
        user_permissions = session_info.get('permissions', [])
        return required_permission in user_permissions or 'admin' in user_permissions