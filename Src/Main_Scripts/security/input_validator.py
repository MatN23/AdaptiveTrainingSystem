import re
import html
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_input: Any
    errors: List[str]
    warnings: List[str]


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    def __init__(self):
        # Content size limits
        self.max_message_length = 10000
        self.max_messages_per_conversation = 100
        self.max_conversation_tokens = 4000
        
        # Patterns for detection
        self.suspicious_patterns = [
            # SQL injection patterns
            re.compile(r"('|(\\')|(;)|(\\\\)|(\\x)|(\\0))", re.IGNORECASE),
            # Script injection patterns
            re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
            # Command injection patterns
            re.compile(r"[;&|`${}()]", re.IGNORECASE),
            # Path traversal patterns
            re.compile(r"(\.\./)|(\.\.\\\\)|(\.\./)|(\.\.\\\)", re.IGNORECASE),
        ]
        
        # Blocked words/phrases (configurable)
        self.blocked_content = [
            # Add sensitive terms as needed
        ]
        
        logging.info("Input validator initialized")
    
    def validate_conversation(self, conversation: Dict[str, Any]) -> ValidationResult:
        """Validate entire conversation input."""
        errors = []
        warnings = []
        
        # Basic structure validation
        if not isinstance(conversation, dict):
            return ValidationResult(False, None, ["Input must be a dictionary"], [])
        
        if 'messages' not in conversation:
            return ValidationResult(False, None, ["Missing 'messages' field"], [])
        
        messages = conversation['messages']
        if not isinstance(messages, list):
            return ValidationResult(False, None, ["Messages must be a list"], [])
        
        if len(messages) > self.max_messages_per_conversation:
            errors.append(f"Too many messages: {len(messages)} > {self.max_messages_per_conversation}")
        
        sanitized_messages = []
        for i, message in enumerate(messages):
            msg_result = self._validate_message(message)
            
            if not msg_result.is_valid:
                errors.extend([f"Message {i}: {error}" for error in msg_result.errors])
            else:
                sanitized_messages.append(msg_result.sanitized_input)
            
            warnings.extend([f"Message {i}: {warning}" for warning in msg_result.warnings])
        
        if errors:
            return ValidationResult(False, None, errors, warnings)
        
        sanitized_conversation = conversation.copy()
        sanitized_conversation['messages'] = sanitized_messages
        
        return ValidationResult(True, sanitized_conversation, [], warnings)
    
    def _validate_message(self, message: Dict[str, Any]) -> ValidationResult:
        """Validate individual message."""
        errors = []
        warnings = []
        
        if not isinstance(message, dict):
            return ValidationResult(False, None, ["Message must be a dictionary"], [])
        
        role = message.get('role', '').lower().strip()
        valid_roles = ['user', 'assistant', 'system', 'prompter']
        
        if not role:
            errors.append("Missing or empty role field")
        elif role not in valid_roles:
            warnings.append(f"Unknown role '{role}', treating as 'user'")
            role = 'user'
        
        content = message.get('content', '')
        if not isinstance(content, str):
            errors.append("Content must be a string")
        else:
            content_result = self._validate_content(content)
            if not content_result.is_valid:
                errors.extend(content_result.errors)
            else:
                content = content_result.sanitized_input
            warnings.extend(content_result.warnings)
        
        if errors:
            return ValidationResult(False, None, errors, warnings)
        
        sanitized_message = {
            'role': role,
            'content': content
        }
        
        return ValidationResult(True, sanitized_message, [], warnings)
    
    def _validate_content(self, content: str) -> ValidationResult:
        """Validate and sanitize message content."""
        errors = []
        warnings = []
        
        if not content.strip():
            errors.append("Content cannot be empty")
            return ValidationResult(False, None, errors, warnings)
        
        # Length validation
        if len(content) > self.max_message_length:
            errors.append(f"Content too long: {len(content)} > {self.max_message_length}")
            return ValidationResult(False, None, errors, warnings)
        
        # Security pattern detection
        for pattern in self.suspicious_patterns:
            if pattern.search(content):
                warnings.append(f"Suspicious pattern detected: {pattern.pattern}")
        
        # Blocked content check
        content_lower = content.lower()
        for blocked in self.blocked_content:
            if blocked.lower() in content_lower:
                errors.append(f"Content contains blocked material")
                return ValidationResult(False, None, errors, warnings)
        
        # Sanitization
        sanitized_content = self._sanitize_content(content)
        
        return ValidationResult(True, sanitized_content, [], warnings)
    
    def _sanitize_content(self, content: str) -> str:
        """Sanitize content while preserving readability."""
        sanitized = html.escape(content)
        
        # Remove or escape potentially dangerous characters
        # Remove null bytes
        sanitized = sanitized.replace('\x00', '')
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized
    
    def validate_user_input(self, user_input: str) -> ValidationResult:
        """Validate direct user input from chat interface."""
        errors = []
        warnings = []
        
        if not isinstance(user_input, str):
            return ValidationResult(False, None, ["Input must be a string"], [])
        
        # Basic length check
        if len(user_input.strip()) == 0:
            return ValidationResult(False, None, ["Input cannot be empty"], [])
        
        if len(user_input) > self.max_message_length:
            errors.append(f"Input too long: {len(user_input)} > {self.max_message_length}")
        
        for pattern in self.suspicious_patterns:
            if pattern.search(user_input):
                warnings.append("Input contains suspicious patterns")
                break
        
        sanitized = self._sanitize_content(user_input)
        
        if errors:
            return ValidationResult(False, None, errors, warnings)
        
        return ValidationResult(True, sanitized, [], warnings)