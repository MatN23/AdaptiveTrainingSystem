import time
import threading
from collections import defaultdict, deque
from typing import Dict, Tuple, Optional
import logging


class RateLimiter:
    """Thread-safe rate limiter with multiple strategies."""
    
    def __init__(self):
        self.buckets: Dict[str, deque] = defaultdict(deque)
        self.lock = threading.RLock()
        
        # Default limits (requests per time window in seconds)
        self.limits = {
            'chat_message': (10, 60),      # 10 messages per minute
            'login_attempt': (5, 300),      # 5 attempts per 5 minutes
            'api_request': (100, 3600),     # 100 requests per hour
            'generation': (20, 300),        # 20 generations per 5 minutes
        }
        
        logging.info("Rate limiter initialized")
    
    def is_allowed(self, identifier: str, action: str, custom_limit: Optional[Tuple[int, int]] = None) -> bool:
        """Check if action is allowed for identifier."""
        limit, window = custom_limit or self.limits.get(action, (10, 60))
        
        with self.lock:
            key = f"{identifier}:{action}"
            bucket = self.buckets[key]
            now = time.time()
            
            # Clean old entries
            while bucket and bucket[0] <= now - window:
                bucket.popleft()
            
            # Check limit
            if len(bucket) >= limit:
                logging.warning(f"Rate limit exceeded for {identifier}:{action}")
                return False
            
            # Record request
            bucket.append(now)
            return True
    
    def get_remaining_requests(self, identifier: str, action: str) -> int:
        """Get number of remaining requests in current window."""
        limit, window = self.limits.get(action, (10, 60))
        
        with self.lock:
            key = f"{identifier}:{action}"
            bucket = self.buckets[key]
            now = time.time()
            
            # Clean old entries
            while bucket and bucket[0] <= now - window:
                bucket.popleft()
            
            return max(0, limit - len(bucket))
    
    def get_reset_time(self, identifier: str, action: str) -> Optional[float]:
        """Get timestamp when rate limit resets."""
        _, window = self.limits.get(action, (10, 60))
        
        with self.lock:
            key = f"{identifier}:{action}"
            bucket = self.buckets[key]
            
            if not bucket:
                return None
            
            return bucket[0] + window
    
    def cleanup_old_buckets(self):
        """Clean up old bucket entries (call periodically)."""
        with self.lock:
            now = time.time()
            keys_to_remove = []
            
            for key, bucket in self.buckets.items():
                action = key.split(':', 1)[1] if ':' in key else 'default'
                _, window = self.limits.get(action, (10, 60))
                
                # Clean old entries
                while bucket and bucket[0] <= now - window:
                    bucket.popleft()
                
                # Remove empty buckets
                if not bucket:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.buckets[key]


# Src/Main_Scripts/security/secure_chat.py
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from security.auth import SecurityManager, User
from security.input_validator import InputValidator, ValidationResult
from security.rate_limiter import RateLimiter


class SecureConversationalChat:
    """Security-enhanced chat interface."""
    
    def __init__(self, chat_instance, security_config: Dict[str, Any]):
        self.chat = chat_instance
        self.security_manager = SecurityManager(security_config)
        self.input_validator = InputValidator()
        self.rate_limiter = RateLimiter()
        
        # Security settings
        self.require_authentication = security_config.get('require_auth', True)
        self.log_all_interactions = security_config.get('log_interactions', True)
        self.enable_content_filtering = security_config.get('content_filtering', True)
        
        logging.info("Secure chat interface initialized")
    
    def authenticate_user(self, username: str, password: str, client_ip: str) -> Optional[str]:
        """Authenticate user and return session token."""
        if not self.require_authentication:
            return "no_auth_required"
        
        return self.security_manager.authenticate(username, password, client_ip)
    
    def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Validate session token."""
        if not self.require_authentication:
            return {
                'username': 'anonymous',
                'permissions': ['chat:basic'],
                'session_info': {'client_ip': 'unknown'}
            }
        
        return self.security_manager.validate_session(session_token)
    
    def secure_generate_response(self, user_input: str, session_token: str, 
                                client_ip: str) -> Dict[str, Any]:
        """Generate response with full security checks."""
        try:
            # Validate session
            session_info = self.validate_session(session_token)
            if not session_info:
                return {
                    'success': False,
                    'error': 'Invalid or expired session',
                    'error_code': 'AUTH_FAILED'
                }
            
            username = session_info['username']
            
            # Check permissions
            if not self.security_manager.check_permission(session_info, 'chat:basic'):
                logging.warning(f"Permission denied for user {username}")
                return {
                    'success': False,
                    'error': 'Insufficient permissions',
                    'error_code': 'PERMISSION_DENIED'
                }
            
            # Rate limiting
            if not self.rate_limiter.is_allowed(client_ip, 'chat_message'):
                logging.warning(f"Rate limit exceeded for {client_ip}")
                return {
                    'success': False,
                    'error': 'Rate limit exceeded. Please wait before sending another message.',
                    'error_code': 'RATE_LIMITED',
                    'retry_after': self.rate_limiter.get_reset_time(client_ip, 'chat_message')
                }
            
            # Input validation
            if self.enable_content_filtering:
                validation_result = self.input_validator.validate_user_input(user_input)
                if not validation_result.is_valid:
                    logging.warning(f"Invalid input from {username}: {validation_result.errors}")
                    return {
                        'success': False,
                        'error': 'Invalid input: ' + '; '.join(validation_result.errors),
                        'error_code': 'INVALID_INPUT'
                    }
                
                # Use sanitized input
                user_input = validation_result.sanitized_input
                
                # Log warnings
                for warning in validation_result.warnings:
                    logging.warning(f"Input warning for {username}: {warning}")
            
            # Generation rate limiting
            if not self.rate_limiter.is_allowed(username, 'generation'):
                return {
                    'success': False,
                    'error': 'Generation rate limit exceeded',
                    'error_code': 'GENERATION_RATE_LIMITED'
                }
            
            # Log interaction
            if self.log_all_interactions:
                logging.info(f"Chat interaction: {username} from {client_ip}")
            
            # Generate response using the underlying chat system
            response, metrics = self.chat.generate_response(user_input)
            
            return {
                'success': True,
                'response': response,
                'metrics': metrics,
                'username': username,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Secure chat error: {e}")
            return {
                'success': False,
                'error': 'Internal server error',
                'error_code': 'SERVER_ERROR'
            }
    
    def create_user(self, username: str, password: str, permissions: list = None) -> bool:
        """Create new user account."""
        return self.security_manager.create_user(username, password, permissions)
    
    def logout_user(self, session_token: str) -> bool:
        """Logout user and invalidate session."""
        return self.security_manager.logout(session_token)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security system status."""
        return {
            'authentication_enabled': self.require_authentication,
            'content_filtering_enabled': self.enable_content_filtering,
            'active_sessions': len(self.security_manager.active_sessions),
            'total_users': len(self.security_manager.users),
            'rate_limit_buckets': len(self.rate_limiter.buckets)
        }


# Example usage and integration
"""
# Src/Main_Scripts/Chat.py - Updated with security

from security.secure_chat import SecureConversationalChat

class ConversationalChat:
    # ... existing code ...
    
    def __init__(self, config: Config, checkpoint_path: Optional[str] = None):
        # ... existing initialization ...
        
        # Initialize security
        security_config = {
            'secret_key': os.environ.get('CHAT_SECRET_KEY', 'development-key-change-me'),
            'require_auth': config.get('require_authentication', True),
            'log_interactions': True,
            'content_filtering': True,
            'session_timeout_hours': 24,
            'max_login_attempts': 5,
            'lockout_duration_minutes': 30
        }
        
        self.secure_chat = SecureConversationalChat(self, security_config)
    
    def run_secure_chat(self):
        '''Run chat with security enabled.'''
        self._print_header()
        
        # Authentication
        if self.secure_chat.require_authentication:
            session_token = self._handle_authentication()
            if not session_token:
                print(" Authentication failed")
                return
        else:
            session_token = "no_auth_required"
        
        client_ip = "127.0.0.1"  # Get from request in web deployment
        
        try:
            while True:
                user_input = input(" You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith('/'):
                    if user_input in ['/quit', '/exit']:
                        break
                    continue
                
                # Use secure generation
                result = self.secure_chat.secure_generate_response(
                    user_input, session_token, client_ip
                )
                
                if result['success']:
                    print(f" Assistant: {result['response']}")
                else:
                    print(f" Error: {result['error']}")
                    if result.get('error_code') == 'RATE_LIMITED':
                        print(f"   Please wait until: {result.get('retry_after')}")
        
        finally:
            self.secure_chat.logout_user(session_token)
    
    def _handle_authentication(self) -> Optional[str]:
        '''Handle user authentication.'''
        print(" Authentication Required")
        
        for attempt in range(3):
            username = input("Username: ").strip()
            password = input("Password: ").strip()
            
            if not username or not password:
                print(" Username and password required")
                continue
            
            session_token = self.secure_chat.authenticate_user(
                username, password, "127.0.0.1"
            )
            
            if session_token:
                print(f" Welcome, {username}!")
                return session_token
            else:
                print(" Invalid credentials")
        
        return None
"""