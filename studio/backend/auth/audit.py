"""
Enterprise Security & RBAC Audit System
Provides comprehensive audit logging, role-based access control, and OAuth2/OIDC integration.
"""

import json
import time
import uuid
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Union
from enum import Enum
from dataclasses import dataclass, asdict, field
from functools import wraps
import threading
from pathlib import Path

from fastapi import HTTPException, Request, Depends, status
from fastapi.security import OAuth2PasswordBearer
import jwt
from jwt.exceptions import PyJWTError

from studio.backend.auth.hashing import Hasher
from studio.backend.auth.storage import UserStorage
from studio.backend.auth.authentication import AuthenticationError

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sovereign.audit")


class AuditEventType(Enum):
    """Types of audit events for comprehensive logging."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    TOKEN_REFRESH = "token_refresh"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"
    PERMISSION_CHECKED = "permission_checked"
    SECRET_ROTATED = "secret_rotated"
    USER_CREATED = "user_created"
    USER_DELETED = "user_deleted"
    PASSWORD_CHANGED = "password_changed"
    OAUTH_GRANT = "oauth_grant"
    OAUTH_REVOKE = "oauth_revoke"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


class Role(Enum):
    """RBAC roles with hierarchical permissions."""
    ADMIN = "admin"
    TRAINER = "trainer"
    VIEWER = "viewer"


class Permission(Enum):
    """Fine-grained permissions for policy engine."""
    # User management
    MANAGE_USERS = "manage_users"
    VIEW_USERS = "view_users"
    
    # Model operations
    TRAIN_MODELS = "train_models"
    DEPLOY_MODELS = "deploy_models"
    VIEW_MODELS = "view_models"
    
    # Data operations
    UPLOAD_DATA = "upload_data"
    VIEW_DATA = "view_data"
    DELETE_DATA = "delete_data"
    
    # System operations
    MANAGE_SYSTEM = "manage_system"
    VIEW_AUDIT_LOGS = "view_audit_logs"
    ROTATE_SECRETS = "rotate_secrets"


@dataclass
class AuditEvent:
    """Structured audit event for comprehensive logging."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    event_type: AuditEventType = AuditEventType.LOGIN_SUCCESS
    user_id: Optional[str] = None
    username: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    outcome: str = "success"
    metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    role: Optional[Role] = None
    permissions_checked: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary for serialization."""
        data = asdict(self)
        data['event_type'] = self.event_type.value
        if self.role:
            data['role'] = self.role.value
        return data
    
    def to_json(self) -> str:
        """Serialize audit event to JSON."""
        return json.dumps(self.to_dict())


class PolicyEngine:
    """Fine-grained permission policy engine for RBAC."""
    
    def __init__(self):
        self.role_permissions: Dict[Role, Set[Permission]] = {
            Role.ADMIN: {
                Permission.MANAGE_USERS,
                Permission.VIEW_USERS,
                Permission.TRAIN_MODELS,
                Permission.DEPLOY_MODELS,
                Permission.VIEW_MODELS,
                Permission.UPLOAD_DATA,
                Permission.VIEW_DATA,
                Permission.DELETE_DATA,
                Permission.MANAGE_SYSTEM,
                Permission.VIEW_AUDIT_LOGS,
                Permission.ROTATE_SECRETS,
            },
            Role.TRAINER: {
                Permission.VIEW_USERS,
                Permission.TRAIN_MODELS,
                Permission.VIEW_MODELS,
                Permission.UPLOAD_DATA,
                Permission.VIEW_DATA,
            },
            Role.VIEWER: {
                Permission.VIEW_USERS,
                Permission.VIEW_MODELS,
                Permission.VIEW_DATA,
            }
        }
        
        # Resource-based policies for fine-grained control
        self.resource_policies: Dict[str, Dict[str, List[Permission]]] = {
            "models": {
                "read": [Permission.VIEW_MODELS],
                "write": [Permission.TRAIN_MODELS, Permission.DEPLOY_MODELS],
                "delete": [Permission.MANAGE_SYSTEM],
            },
            "users": {
                "read": [Permission.VIEW_USERS],
                "write": [Permission.MANAGE_USERS],
                "delete": [Permission.MANAGE_USERS],
            },
            "audit_logs": {
                "read": [Permission.VIEW_AUDIT_LOGS],
            },
            "secrets": {
                "rotate": [Permission.ROTATE_SECRETS],
            }
        }
    
    def check_permission(self, role: Role, resource: str, action: str) -> bool:
        """Check if role has permission for resource action."""
        required_permissions = self.resource_policies.get(resource, {}).get(action, [])
        if not required_permissions:
            return False
        
        user_permissions = self.role_permissions.get(role, set())
        return any(perm in user_permissions for perm in required_permissions)
    
    def get_role_permissions(self, role: Role) -> Set[Permission]:
        """Get all permissions for a role."""
        return self.role_permissions.get(role, set())


class SecretManager:
    """Handles secret rotation and secure storage."""
    
    def __init__(self, storage_path: str = "secrets.json"):
        self.storage_path = Path(storage_path)
        self.secrets: Dict[str, Dict[str, Any]] = self._load_secrets()
        self.rotation_interval = timedelta(days=90)  # Default 90-day rotation
        self._lock = threading.RLock()
    
    def _load_secrets(self) -> Dict[str, Dict[str, Any]]:
        """Load secrets from secure storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning("Failed to load secrets, initializing empty store")
        return {}
    
    def _save_secrets(self) -> None:
        """Save secrets to secure storage."""
        with self._lock:
            try:
                with open(self.storage_path, 'w') as f:
                    json.dump(self.secrets, f, indent=2)
                # Set restrictive permissions
                self.storage_path.chmod(0o600)
            except IOError as e:
                logger.error(f"Failed to save secrets: {e}")
                raise
    
    def generate_secret(self, name: str, length: int = 64) -> str:
        """Generate and store a new secret."""
        with self._lock:
            secret = secrets.token_urlsafe(length)
            self.secrets[name] = {
                "value": secret,
                "created_at": datetime.utcnow().isoformat(),
                "last_rotated": datetime.utcnow().isoformat(),
                "rotation_due": (datetime.utcnow() + self.rotation_interval).isoformat()
            }
            self._save_secrets()
            return secret
    
    def rotate_secret(self, name: str) -> str:
        """Rotate an existing secret."""
        with self._lock:
            if name not in self.secrets:
                raise ValueError(f"Secret {name} not found")
            
            old_secret = self.secrets[name]["value"]
            new_secret = secrets.token_urlsafe(64)
            
            self.secrets[name].update({
                "value": new_secret,
                "last_rotated": datetime.utcnow().isoformat(),
                "rotation_due": (datetime.utcnow() + self.rotation_interval).isoformat(),
                "previous_hash": Hasher.hash(old_secret)  # Store hash for verification
            })
            
            self._save_secrets()
            logger.info(f"Secret {name} rotated successfully")
            return new_secret
    
    def get_secret(self, name: str) -> Optional[str]:
        """Retrieve a secret by name."""
        with self._lock:
            secret_data = self.secrets.get(name)
            if secret_data:
                # Check if rotation is due
                rotation_due = datetime.fromisoformat(secret_data["rotation_due"])
                if datetime.utcnow() > rotation_due:
                    logger.warning(f"Secret {name} is overdue for rotation")
                return secret_data["value"]
            return None
    
    def check_rotation_needed(self) -> List[str]:
        """Check which secrets need rotation."""
        with self._lock:
            needs_rotation = []
            now = datetime.utcnow()
            for name, data in self.secrets.items():
                rotation_due = datetime.fromisoformat(data["rotation_due"])
                if now > rotation_due:
                    needs_rotation.append(name)
            return needs_rotation


class AuditLogger:
    """Centralized audit logging system with structured events."""
    
    def __init__(self, log_file: str = "audit.log"):
        self.log_file = Path(log_file)
        self._ensure_log_file()
        self._lock = threading.RLock()
        self.logger = logging.getLogger("sovereign.audit.file")
        
        # Configure file handler for audit logs
        handler = logging.FileHandler(self.log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _ensure_log_file(self) -> None:
        """Ensure audit log file exists with proper permissions."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_file.exists():
            self.log_file.touch(mode=0o600)
    
    def log_event(self, event: AuditEvent) -> None:
        """Log an audit event to file and system logger."""
        with self._lock:
            # Write to structured log file
            self.logger.info(event.to_json())
            
            # Also log to system logger for immediate visibility
            logger.info(
                f"AUDIT: {event.event_type.value} - "
                f"User: {event.username or 'anonymous'} - "
                f"Outcome: {event.outcome} - "
                f"Resource: {event.resource or 'N/A'}"
            )
    
    def log_authentication(self, username: str, success: bool, 
                          ip_address: str = None, user_agent: str = None,
                          reason: str = None) -> None:
        """Log authentication attempt."""
        event = AuditEvent(
            event_type=AuditEventType.LOGIN_SUCCESS if success else AuditEventType.LOGIN_FAILURE,
            username=username,
            ip_address=ip_address,
            user_agent=user_agent,
            outcome="success" if success else "failure",
            metadata={"reason": reason} if reason else {}
        )
        self.log_event(event)
    
    def log_access(self, user_id: str, username: str, role: Role,
                  resource: str, action: str, granted: bool,
                  ip_address: str = None, session_id: str = None) -> None:
        """Log resource access attempt."""
        event = AuditEvent(
            event_type=AuditEventType.ACCESS_GRANTED if granted else AuditEventType.ACCESS_DENIED,
            user_id=user_id,
            username=username,
            role=role,
            resource=resource,
            action=action,
            outcome="success" if granted else "denied",
            ip_address=ip_address,
            session_id=session_id,
            permissions_checked=[f"{role.value}:{resource}:{action}"]
        )
        self.log_event(event)
    
    def log_role_change(self, admin_user: str, target_user: str,
                       role: Role, assigned: bool) -> None:
        """Log role assignment or revocation."""
        event = AuditEvent(
            event_type=AuditEventType.ROLE_ASSIGNED if assigned else AuditEventType.ROLE_REVOKED,
            username=admin_user,
            resource="users",
            action="role_change",
            outcome="success",
            metadata={
                "target_user": target_user,
                "role": role.value,
                "action": "assigned" if assigned else "revoked"
            }
        )
        self.log_event(event)
    
    def log_secret_rotation(self, secret_name: str, rotated_by: str) -> None:
        """Log secret rotation event."""
        event = AuditEvent(
            event_type=AuditEventType.SECRET_ROTATED,
            username=rotated_by,
            resource="secrets",
            action="rotate",
            outcome="success",
            metadata={"secret_name": secret_name}
        )
        self.log_event(event)
    
    def get_recent_events(self, limit: int = 100, 
                         event_type: AuditEventType = None) -> List[Dict[str, Any]]:
        """Retrieve recent audit events (admin only)."""
        events = []
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()[-limit:]  # Get last N lines
                for line in lines:
                    try:
                        # Extract JSON from log line
                        json_str = line.split(' - INFO - ')[-1]
                        event_data = json.loads(json_str)
                        if event_type is None or event_data.get('event_type') == event_type.value:
                            events.append(event_data)
                    except (json.JSONDecodeError, IndexError):
                        continue
        except IOError as e:
            logger.error(f"Failed to read audit log: {e}")
        
        return events


class OAuth2Provider:
    """OAuth2/OIDC provider integration."""
    
    def __init__(self, secret_manager: SecretManager):
        self.secret_manager = secret_manager
        self.jwt_secret = self.secret_manager.get_secret("jwt_secret") or \
                         self.secret_manager.generate_secret("jwt_secret")
        self.algorithm = "HS256"
        self.token_expiry = timedelta(hours=1)
        self.refresh_token_expiry = timedelta(days=30)
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
    
    def create_access_token(self, data: Dict[str, Any], 
                           expires_delta: timedelta = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or self.token_expiry)
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4()),
            "type": "access"
        })
        return jwt.encode(to_encode, self.jwt_secret, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create refresh token for token renewal."""
        to_encode = {
            "sub": user_id,
            "exp": datetime.utcnow() + self.refresh_token_expiry,
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4()),
            "type": "refresh"
        }
        return jwt.encode(to_encode, self.jwt_secret, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.algorithm])
            return payload
        except PyJWTError as e:
            raise AuthenticationError(f"Invalid token: {e}")
    
    def rotate_jwt_secret(self, rotated_by: str) -> None:
        """Rotate JWT signing secret."""
        self.jwt_secret = self.secret_manager.rotate_secret("jwt_secret")
        logger.info(f"JWT secret rotated by {rotated_by}")


class RBACMiddleware:
    """Middleware for role-based access control."""
    
    def __init__(self, policy_engine: PolicyEngine, audit_logger: AuditLogger):
        self.policy_engine = policy_engine
        self.audit_logger = audit_logger
    
    def require_permission(self, resource: str, action: str):
        """Decorator to require specific permission for endpoint access."""
        def decorator(func):
            @wraps(func)
            async def wrapper(request: Request, *args, **kwargs):
                # Extract user from request (set by authentication middleware)
                user = getattr(request.state, 'user', None)
                if not user:
                    self.audit_logger.log_access(
                        user_id="anonymous",
                        username="anonymous",
                        role=Role.VIEWER,
                        resource=resource,
                        action=action,
                        granted=False,
                        ip_address=request.client.host if request.client else None
                    )
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )
                
                # Check permission
                has_permission = self.policy_engine.check_permission(
                    user['role'], resource, action
                )
                
                # Log access attempt
                self.audit_logger.log_access(
                    user_id=user['id'],
                    username=user['username'],
                    role=user['role'],
                    resource=resource,
                    action=action,
                    granted=has_permission,
                    ip_address=request.client.host if request.client else None,
                    session_id=user.get('session_id')
                )
                
                if not has_permission:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Insufficient permissions for {action} on {resource}"
                    )
                
                return await func(request, *args, **kwargs)
            return wrapper
        return decorator
    
    def require_role(self, required_role: Role):
        """Decorator to require specific role for endpoint access."""
        def decorator(func):
            @wraps(func)
            async def wrapper(request: Request, *args, **kwargs):
                user = getattr(request.state, 'user', None)
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )
                
                if user['role'] != required_role and user['role'] != Role.ADMIN:
                    self.audit_logger.log_access(
                        user_id=user['id'],
                        username=user['username'],
                        role=user['role'],
                        resource="role_check",
                        action="access",
                        granted=False,
                        ip_address=request.client.host if request.client else None
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Role {required_role.value} required"
                    )
                
                return await func(request, *args, **kwargs)
            return wrapper
        return decorator


class EnterpriseAuthSystem:
    """Main enterprise authentication and authorization system."""
    
    def __init__(self, storage_path: str = "auth_data"):
        self.storage = UserStorage(storage_path)
        self.hasher = Hasher()
        self.secret_manager = SecretManager(Path(storage_path) / "secrets.json")
        self.audit_logger = AuditLogger(Path(storage_path) / "audit.log")
        self.policy_engine = PolicyEngine()
        self.oauth2_provider = OAuth2Provider(self.secret_manager)
        self.rbac_middleware = RBACMiddleware(self.policy_engine, self.audit_logger)
        
        # Initialize default admin user if none exists
        self._initialize_default_users()
        
        # Start secret rotation monitor
        self._start_rotation_monitor()
    
    def _initialize_default_users(self) -> None:
        """Create default admin user if no users exist."""
        if not self.storage.list_users():
            admin_password = self.hasher.hash_password("admin")
            self.storage.create_user(
                username="admin",
                password_hash=admin_password,
                role=Role.ADMIN.value,
                email="admin@sovereign.ai"
            )
            logger.info("Default admin user created")
    
    def _start_rotation_monitor(self) -> None:
        """Start background thread to monitor secret rotation."""
        def monitor():
            while True:
                time.sleep(86400)  # Check daily
                needs_rotation = self.secret_manager.check_rotation_needed()
                for secret_name in needs_rotation:
                    logger.warning(f"Secret {secret_name} needs rotation")
                    # Auto-rotate non-critical secrets
                    if secret_name != "jwt_secret":  # JWT secret rotation requires downtime
                        try:
                            self.secret_manager.rotate_secret(secret_name)
                            self.audit_logger.log_secret_rotation(
                                secret_name, "system_auto_rotation"
                            )
                        except Exception as e:
                            logger.error(f"Auto-rotation failed for {secret_name}: {e}")
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
    
    def authenticate_user(self, username: str, password: str, 
                         ip_address: str = None, user_agent: str = None) -> Dict[str, Any]:
        """Authenticate user with Argon2id hashing."""
        user = self.storage.get_user(username)
        if not user:
            self.audit_logger.log_authentication(
                username, False, ip_address, user_agent, "user_not_found"
            )
            raise AuthenticationError("Invalid credentials")
        
        if not self.hasher.verify_password(password, user['password_hash']):
            self.audit_logger.log_authentication(
                username, False, ip_address, user_agent, "invalid_password"
            )
            raise AuthenticationError("Invalid credentials")
        
        # Create tokens
        token_data = {
            "sub": user['id'],
            "username": user['username'],
            "role": user['role'],
            "permissions": list(self.policy_engine.get_role_permissions(Role(user['role'])))
        }
        
        access_token = self.oauth2_provider.create_access_token(token_data)
        refresh_token = self.oauth2_provider.create_refresh_token(user['id'])
        
        # Store refresh token
        session_id = str(uuid.uuid4())
        self.storage.store_session(
            user_id=user['id'],
            session_id=session_id,
            refresh_token=self.hasher.hash(refresh_token),
            expires_at=datetime.utcnow() + self.oauth2_provider.refresh_token_expiry
        )
        
        self.audit_logger.log_authentication(username, True, ip_address, user_agent)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": int(self.oauth2_provider.token_expiry.total_seconds()),
            "user": {
                "id": user['id'],
                "username": user['username'],
                "email": user['email'],
                "role": user['role']
            },
            "session_id": session_id
        }
    
    def verify_access_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT access token and return user data."""
        try:
            payload = self.oauth2_provider.verify_token(token)
            if payload.get('type') != 'access':
                raise AuthenticationError("Invalid token type")
            
            # Get fresh user data
            user = self.storage.get_user_by_id(payload['sub'])
            if not user:
                raise AuthenticationError("User not found")
            
            return {
                "id": user['id'],
                "username": user['username'],
                "email": user['email'],
                "role": Role(user['role']),
                "session_id": payload.get('jti')
            }
        except PyJWTError as e:
            raise AuthenticationError(f"Token verification failed: {e}")
    
    def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token using refresh token."""
        try:
            payload = self.oauth2_provider.verify_token(refresh_token)
            if payload.get('type') != 'refresh':
                raise AuthenticationError("Invalid token type")
            
            # Verify refresh token exists in storage
            user_id = payload['sub']
            sessions = self.storage.get_user_sessions(user_id)
            
            token_found = False
            for session in sessions:
                if self.hasher.verify(refresh_token, session['refresh_token_hash']):
                    token_found = True
                    break
            
            if not token_found:
                raise AuthenticationError("Refresh token not found")
            
            # Get user data
            user = self.storage.get_user_by_id(user_id)
            if not user:
                raise AuthenticationError("User not found")
            
            # Create new access token
            token_data = {
                "sub": user['id'],
                "username": user['username'],
                "role": user['role'],
                "permissions": list(self.policy_engine.get_role_permissions(Role(user['role'])))
            }
            
            new_access_token = self.oauth2_provider.create_access_token(token_data)
            
            self.audit_logger.log_event(AuditEvent(
                event_type=AuditEventType.TOKEN_REFRESH,
                user_id=user['id'],
                username=user['username'],
                outcome="success"
            ))
            
            return {
                "access_token": new_access_token,
                "token_type": "bearer",
                "expires_in": int(self.oauth2_provider.token_expiry.total_seconds())
            }
        except PyJWTError as e:
            raise AuthenticationError(f"Token refresh failed: {e}")
    
    def assign_role(self, admin_username: str, target_username: str, role: Role) -> None:
        """Assign role to user (admin only)."""
        admin_user = self.storage.get_user(admin_username)
        if not admin_user or Role(admin_user['role']) != Role.ADMIN:
            raise AuthenticationError("Insufficient permissions")
        
        target_user = self.storage.get_user(target_username)
        if not target_user:
            raise AuthenticationError("Target user not found")
        
        self.storage.update_user_role(target_username, role.value)
        self.audit_logger.log_role_change(admin_username, target_username, role, True)
    
    def rotate_secrets(self, admin_username: str) -> Dict[str, str]:
        """Rotate all secrets (admin only)."""
        admin_user = self.storage.get_user(admin_username)
        if not admin_user or Role(admin_user['role']) != Role.ADMIN:
            raise AuthenticationError("Insufficient permissions")
        
        rotated = {}
        for secret_name in self.secret_manager.secrets.keys():
            try:
                new_secret = self.secret_manager.rotate_secret(secret_name)
                rotated[secret_name] = "rotated"
                self.audit_logger.log_secret_rotation(secret_name, admin_username)
            except Exception as e:
                rotated[secret_name] = f"failed: {e}"
                logger.error(f"Failed to rotate {secret_name}: {e}")
        
        return rotated
    
    def get_rbac_middleware(self) -> RBACMiddleware:
        """Get RBAC middleware for FastAPI integration."""
        return self.rbac_middleware
    
    def get_audit_logs(self, admin_username: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve audit logs (admin only)."""
        admin_user = self.storage.get_user(admin_username)
        if not admin_user or Role(admin_user['role']) != Role.ADMIN:
            raise AuthenticationError("Insufficient permissions")
        
        return self.audit_logger.get_recent_events(limit)


# Global instance for application-wide use
enterprise_auth = EnterpriseAuthSystem()

# Export key components for easy access
__all__ = [
    'EnterpriseAuthSystem',
    'AuditLogger',
    'AuditEvent',
    'AuditEventType',
    'Role',
    'Permission',
    'PolicyEngine',
    'SecretManager',
    'OAuth2Provider',
    'RBACMiddleware',
    'enterprise_auth'
]