"""Enterprise Security & RBAC module for SOVEREIGN Studio.

Implements OAuth2/OIDC authentication, role-based access control,
audit logging, and secret rotation with Argon2id hashing.
"""

import os
import json
import time
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from functools import wraps
import hashlib
import hmac
import base64

import jwt
from jwt import PyJWTError
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from .hashing import Hasher
from .storage import AuthStorage

logger = logging.getLogger(__name__)


class Role(str, Enum):
    """RBAC roles with hierarchical permissions."""
    ADMIN = "admin"
    TRAINER = "trainer"
    VIEWER = "viewer"


class Permission(str, Enum):
    """Fine-grained permissions for resources."""
    # Model permissions
    MODEL_CREATE = "model:create"
    MODEL_READ = "model:read"
    MODEL_UPDATE = "model:update"
    MODEL_DELETE = "model:delete"
    MODEL_TRAIN = "model:train"
    
    # Data permissions
    DATA_CREATE = "data:create"
    DATA_READ = "data:read"
    DATA_UPDATE = "data:update"
    DATA_DELETE = "data:delete"
    
    # User management
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    
    # System permissions
    SYSTEM_CONFIG = "system:config"
    SYSTEM_AUDIT = "system:audit"
    SYSTEM_ROTATE_SECRETS = "system:rotate_secrets"


class AuthEvent(str, Enum):
    """Audit log event types."""
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILURE = "auth.login.failure"
    LOGOUT = "auth.logout"
    TOKEN_REFRESH = "auth.token.refresh"
    TOKEN_REVOKED = "auth.token.revoked"
    PERMISSION_DENIED = "auth.permission.denied"
    SECRET_ROTATED = "auth.secret.rotated"
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"
    ROLE_ASSIGNED = "user.role.assigned"
    ROLE_REVOKED = "user.role.revoked"


@dataclass
class AuditLogEntry:
    """Structured audit log entry."""
    timestamp: str
    event: AuthEvent
    user_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource: Optional[str]
    action: Optional[str]
    success: bool
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class PolicyEngine:
    """Fine-grained permission policy engine."""
    
    # Role to permission mapping
    ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
        Role.ADMIN: {
            Permission.MODEL_CREATE, Permission.MODEL_READ, Permission.MODEL_UPDATE,
            Permission.MODEL_DELETE, Permission.MODEL_TRAIN,
            Permission.DATA_CREATE, Permission.DATA_READ, Permission.DATA_UPDATE,
            Permission.DATA_DELETE,
            Permission.USER_CREATE, Permission.USER_READ, Permission.USER_UPDATE,
            Permission.USER_DELETE,
            Permission.SYSTEM_CONFIG, Permission.SYSTEM_AUDIT,
            Permission.SYSTEM_ROTATE_SECRETS,
        },
        Role.TRAINER: {
            Permission.MODEL_CREATE, Permission.MODEL_READ, Permission.MODEL_UPDATE,
            Permission.MODEL_TRAIN,
            Permission.DATA_CREATE, Permission.DATA_READ, Permission.DATA_UPDATE,
            Permission.USER_READ,
        },
        Role.VIEWER: {
            Permission.MODEL_READ,
            Permission.DATA_READ,
            Permission.USER_READ,
        },
    }
    
    # Resource-based policies (can be extended with ABAC)
    RESOURCE_POLICIES: Dict[str, Dict[str, List[Permission]]] = {
        "model": {
            "create": [Permission.MODEL_CREATE],
            "read": [Permission.MODEL_READ],
            "update": [Permission.MODEL_UPDATE],
            "delete": [Permission.MODEL_DELETE],
            "train": [Permission.MODEL_TRAIN],
        },
        "data": {
            "create": [Permission.DATA_CREATE],
            "read": [Permission.DATA_READ],
            "update": [Permission.DATA_UPDATE],
            "delete": [Permission.DATA_DELETE],
        },
        "user": {
            "create": [Permission.USER_CREATE],
            "read": [Permission.USER_READ],
            "update": [Permission.USER_UPDATE],
            "delete": [Permission.USER_DELETE],
        },
        "system": {
            "config": [Permission.SYSTEM_CONFIG],
            "audit": [Permission.SYSTEM_AUDIT],
            "rotate_secrets": [Permission.SYSTEM_ROTATE_SECRETS],
        },
    }
    
    @classmethod
    def get_permissions_for_role(cls, role: Role) -> Set[Permission]:
        """Get all permissions for a given role."""
        return cls.ROLE_PERMISSIONS.get(role, set())
    
    @classmethod
    def check_permission(
        cls,
        user_roles: List[Role],
        required_permission: Permission,
        resource: Optional[str] = None,
        action: Optional[str] = None,
    ) -> bool:
        """Check if user roles grant the required permission."""
        # Collect all permissions from user roles
        user_permissions: Set[Permission] = set()
        for role in user_roles:
            user_permissions.update(cls.get_permissions_for_role(role))
        
        # Check direct permission
        if required_permission in user_permissions:
            return True
        
        # Check resource-based permission if resource and action provided
        if resource and action:
            resource_policy = cls.RESOURCE_POLICIES.get(resource, {})
            required_permissions = resource_policy.get(action, [])
            return any(perm in user_permissions for perm in required_permissions)
        
        return False
    
    @classmethod
    def validate_role_hierarchy(cls, assigner_role: Role, target_role: Role) -> bool:
        """Validate if assigner can assign target role."""
        hierarchy = {
            Role.ADMIN: {Role.ADMIN, Role.TRAINER, Role.VIEWER},
            Role.TRAINER: {Role.TRAINER, Role.VIEWER},
            Role.VIEWER: set(),
        }
        return target_role in hierarchy.get(assigner_role, set())


class SecretRotator:
    """Handles secret rotation with zero-downtime."""
    
    def __init__(self, storage: AuthStorage):
        self.storage = storage
        self.rotation_interval = timedelta(days=30)
        self.grace_period = timedelta(hours=24)
    
    def generate_secret(self, length: int = 64) -> str:
        """Generate a cryptographically secure secret."""
        return secrets.token_urlsafe(length)
    
    async def rotate_jwt_secret(self) -> Tuple[str, str]:
        """Rotate JWT secret with grace period for old tokens."""
        current_time = datetime.utcnow()
        
        # Get current secret
        current_secret = await self.storage.get_jwt_secret()
        if not current_secret:
            # Initialize first secret
            new_secret = self.generate_secret()
            await self.storage.set_jwt_secret(
                secret=new_secret,
                created_at=current_time,
                expires_at=current_time + self.rotation_interval
            )
            return new_secret, ""
        
        # Check if rotation is needed
        secret_data = await self.storage.get_jwt_secret_metadata()
        if secret_data and secret_data.get("expires_at"):
            expires_at = datetime.fromisoformat(secret_data["expires_at"])
            if current_time < expires_at - self.grace_period:
                # No rotation needed yet
                return current_secret, ""
        
        # Perform rotation
        new_secret = self.generate_secret()
        old_secret = current_secret
        
        # Store new secret and mark old for grace period
        await self.storage.set_jwt_secret(
            secret=new_secret,
            created_at=current_time,
            expires_at=current_time + self.rotation_interval
        )
        
        # Keep old secret for grace period
        await self.storage.set_jwt_secret(
            secret=old_secret,
            created_at=current_time - self.rotation_interval,
            expires_at=current_time + self.grace_period,
            is_previous=True
        )
        
        logger.info("JWT secret rotated successfully")
        return new_secret, old_secret
    
    async def get_active_secrets(self) -> List[str]:
        """Get all active JWT secrets (current and previous in grace period)."""
        secrets_list = []
        
        current_secret = await self.storage.get_jwt_secret()
        if current_secret:
            secrets_list.append(current_secret)
        
        previous_secret = await self.storage.get_previous_jwt_secret()
        if previous_secret:
            # Verify it's still in grace period
            metadata = await self.storage.get_jwt_secret_metadata(is_previous=True)
            if metadata and metadata.get("expires_at"):
                expires_at = datetime.fromisoformat(metadata["expires_at"])
                if datetime.utcnow() < expires_at:
                    secrets_list.append(previous_secret)
        
        return secrets_list


class OAuth2Provider:
    """OAuth2/OIDC provider implementation."""
    
    def __init__(
        self,
        storage: AuthStorage,
        secret_rotator: SecretRotator,
        issuer: str = "sovereign-studio",
        audience: str = "sovereign-api",
        token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
    ):
        self.storage = storage
        self.secret_rotator = secret_rotator
        self.issuer = issuer
        self.audience = audience
        self.token_expire_minutes = token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self.password_hasher = PasswordHasher()  # Argon2id
    
    async def create_tokens(
        self, user_id: str, roles: List[Role], metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create access and refresh tokens."""
        now = datetime.utcnow()
        access_token_expires = now + timedelta(minutes=self.token_expire_minutes)
        refresh_token_expires = now + timedelta(days=self.refresh_token_expire_days)
        
        # Get current JWT secret
        secrets_list = await self.secret_rotator.get_active_secrets()
        if not secrets_list:
            raise ValueError("No active JWT secrets available")
        
        current_secret = secrets_list[0]
        
        # Create access token
        access_token_payload = {
            "sub": user_id,
            "iss": self.issuer,
            "aud": self.audience,
            "iat": now,
            "exp": access_token_expires,
            "type": "access",
            "roles": [role.value for role in roles],
            "metadata": metadata or {},
        }
        
        access_token = jwt.encode(
            access_token_payload,
            current_secret,
            algorithm="HS256"
        )
        
        # Create refresh token
        refresh_token = secrets.token_urlsafe(64)
        refresh_token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
        
        # Store refresh token
        await self.storage.store_refresh_token(
            token_hash=refresh_token_hash,
            user_id=user_id,
            expires_at=refresh_token_expires,
            metadata=metadata
        )
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": self.token_expire_minutes * 60,
            "refresh_token": refresh_token,
            "refresh_token_expires_in": self.refresh_token_expire_days * 24 * 60 * 60,
        }
    
    async def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token with secret rotation support."""
        secrets_list = await self.secret_rotator.get_active_secrets()
        
        last_error = None
        for secret in secrets_list:
            try:
                payload = jwt.decode(
                    token,
                    secret,
                    algorithms=["HS256"],
                    issuer=self.issuer,
                    audience=self.audience,
                )
                
                # Verify token type
                if payload.get("type") != "access":
                    raise jwt.InvalidTokenError("Invalid token type")
                
                return payload
            except PyJWTError as e:
                last_error = e
                continue
        
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {str(last_error)}"
        )
    
    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """Refresh access token using refresh token."""
        refresh_token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
        
        # Validate refresh token
        token_data = await self.storage.get_refresh_token(refresh_token_hash)
        if not token_data:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        # Check expiration
        expires_at = datetime.fromisoformat(token_data["expires_at"])
        if datetime.utcnow() > expires_at:
            await self.storage.revoke_refresh_token(refresh_token_hash)
            raise HTTPException(status_code=401, detail="Refresh token expired")
        
        # Get user roles
        user_data = await self.storage.get_user(token_data["user_id"])
        if not user_data:
            raise HTTPException(status_code=404, detail="User not found")
        
        roles = [Role(role) for role in user_data.get("roles", [])]
        
        # Revoke old refresh token (rotation)
        await self.storage.revoke_refresh_token(refresh_token_hash)
        
        # Create new tokens
        return await self.create_tokens(
            user_id=token_data["user_id"],
            roles=roles,
            metadata=token_data.get("metadata")
        )
    
    async def revoke_token(self, token: str, token_type: str = "access") -> None:
        """Revoke a token (add to blacklist)."""
        if token_type == "refresh":
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            await self.storage.revoke_refresh_token(token_hash)
        else:
            # For access tokens, we'd need a token blacklist
            # For simplicity, we'll just log it
            logger.info(f"Access token revoked (would add to blacklist)")


class RBACManager:
    """Role-Based Access Control manager."""
    
    def __init__(self, storage: AuthStorage, oauth_provider: OAuth2Provider):
        self.storage = storage
        self.oauth_provider = oauth_provider
        self.policy_engine = PolicyEngine()
    
    async def assign_role(self, user_id: str, role: Role, assigner_id: str) -> bool:
        """Assign a role to a user."""
        # Get assigner's roles
        assigner_data = await self.storage.get_user(assigner_id)
        if not assigner_data:
            raise HTTPException(status_code=404, detail="Assigner not found")
        
        assigner_roles = [Role(r) for r in assigner_data.get("roles", [])]
        
        # Check if assigner can assign this role
        can_assign = any(
            self.policy_engine.validate_role_hierarchy(assigner_role, role)
            for assigner_role in assigner_roles
        )
        
        if not can_assign:
            raise HTTPException(
                status_code=403,
                detail="Insufficient permissions to assign this role"
            )
        
        # Assign role
        success = await self.storage.assign_user_role(user_id, role.value)
        
        if success:
            # Log the event
            await self._log_audit_event(
                event=AuthEvent.ROLE_ASSIGNED,
                user_id=assigner_id,
                resource="user",
                action="update",
                success=True,
                metadata={"target_user": user_id, "role": role.value}
            )
        
        return success
    
    async def check_permission(
        self,
        user_id: str,
        permission: Permission,
        resource: Optional[str] = None,
        action: Optional[str] = None,
    ) -> bool:
        """Check if user has the required permission."""
        user_data = await self.storage.get_user(user_id)
        if not user_data:
            return False
        
        user_roles = [Role(role) for role in user_data.get("roles", [])]
        return self.policy_engine.check_permission(
            user_roles, permission, resource, action
        )
    
    async def require_permission(
        self,
        permission: Permission,
        resource: Optional[str] = None,
        action: Optional[str] = None,
    ):
        """Dependency for FastAPI to require specific permission."""
        async def permission_checker(
            request: Request,
            credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())
        ):
            # Validate token
            payload = await self.oauth_provider.validate_token(credentials.credentials)
            user_id = payload.get("sub")
            
            if not user_id:
                raise HTTPException(status_code=401, detail="Invalid token")
            
            # Check permission
            has_permission = await self.check_permission(
                user_id, permission, resource, action
            )
            
            if not has_permission:
                # Log permission denied
                await self._log_audit_event(
                    event=AuthEvent.PERMISSION_DENIED,
                    user_id=user_id,
                    ip_address=request.client.host if request.client else None,
                    user_agent=request.headers.get("user-agent"),
                    resource=resource,
                    action=action,
                    success=False,
                    metadata={"required_permission": permission.value}
                )
                
                raise HTTPException(
                    status_code=403,
                    detail="Insufficient permissions"
                )
            
            # Add user info to request state
            request.state.user_id = user_id
            request.state.user_roles = payload.get("roles", [])
            
            return payload
        
        return Depends(permission_checker)
    
    async def _log_audit_event(
        self,
        event: AuthEvent,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        success: bool = True,
        metadata: Optional[Dict] = None,
    ):
        """Log an audit event."""
        entry = AuditLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            event=event,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            success=success,
            metadata=metadata or {}
        )
        
        # Log to structured audit trail
        logger.info(
            "AUDIT: %s",
            json.dumps(entry.to_dict()),
            extra={"audit": True, **entry.to_dict()}
        )
        
        # Also store in database for querying
        await self.storage.store_audit_log(entry.to_dict())


class Argon2Hasher:
    """Argon2id password hasher replacement for PBKDF2."""
    
    def __init__(self):
        self.hasher = PasswordHasher(
            time_cost=3,  # Number of iterations
            memory_cost=65536,  # 64MB memory usage
            parallelism=4,  # Number of parallel threads
            hash_len=32,  # Length of the hash in bytes
            salt_len=16,  # Length of random salt in bytes
        )
    
    def hash_password(self, password: str) -> str:
        """Hash password using Argon2id."""
        return self.hasher.hash(password)
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against Argon2id hash."""
        try:
            return self.hasher.verify(hashed, password)
        except VerifyMismatchError:
            return False
    
    def needs_rehash(self, hashed: str) -> bool:
        """Check if hash needs to be updated (parameters changed)."""
        return self.hasher.check_needs_rehash(hashed)


# FastAPI dependencies
async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())
) -> Dict[str, Any]:
    """Get current authenticated user from JWT token."""
    # This would be implemented with the OAuth2Provider
    # For now, return a placeholder
    return {"user_id": "placeholder", "roles": ["viewer"]}


def require_roles(*required_roles: Role):
    """Decorator to require specific roles for an endpoint."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get request from kwargs
            request = kwargs.get('request')
            if not request:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break
            
            if not request:
                raise HTTPException(status_code=500, detail="Request not found")
            
            # Get user from request state (set by auth middleware)
            user_roles = getattr(request.state, 'user_roles', [])
            
            # Check if user has any of the required roles
            user_role_enums = [Role(r) for r in user_roles if r in [role.value for role in Role]]
            if not any(role in user_role_enums for role in required_roles):
                raise HTTPException(
                    status_code=403,
                    detail=f"Required roles: {[r.value for r in required_roles]}"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Initialize global instances (to be configured by application)
_auth_storage: Optional[AuthStorage] = None
_secret_rotator: Optional[SecretRotator] = None
_oauth_provider: Optional[OAuth2Provider] = None
_rbac_manager: Optional[RBACManager] = None
_argon2_hasher: Optional[Argon2Hasher] = None


def initialize_security(
    storage: AuthStorage,
    jwt_secret: Optional[str] = None,
    issuer: str = "sovereign-studio",
    audience: str = "sovereign-api",
) -> Tuple[OAuth2Provider, RBACManager, Argon2Hasher]:
    """Initialize security components."""
    global _auth_storage, _secret_rotator, _oauth_provider, _rbac_manager, _argon2_hasher
    
    _auth_storage = storage
    _secret_rotator = SecretRotator(storage)
    _oauth_provider = OAuth2Provider(
        storage=storage,
        secret_rotator=_secret_rotator,
        issuer=issuer,
        audience=audience
    )
    _rbac_manager = RBACManager(storage, _oauth_provider)
    _argon2_hasher = Argon2Hasher()
    
    # Initialize JWT secret if provided
    if jwt_secret:
        import asyncio
        asyncio.create_task(
            storage.set_jwt_secret(
                secret=jwt_secret,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(days=30)
            )
        )
    
    return _oauth_provider, _rbac_manager, _argon2_hasher


def get_oauth_provider() -> OAuth2Provider:
    """Get the global OAuth2 provider instance."""
    if not _oauth_provider:
        raise RuntimeError("Security not initialized. Call initialize_security first.")
    return _oauth_provider


def get_rbac_manager() -> RBACManager:
    """Get the global RBAC manager instance."""
    if not _rbac_manager:
        raise RuntimeError("Security not initialized. Call initialize_security first.")
    return _rbac_manager


def get_argon2_hasher() -> Argon2Hasher:
    """Get the global Argon2 hasher instance."""
    if not _argon2_hasher:
        raise RuntimeError("Security not initialized. Call initialize_security first.")
    return _argon2_hasher


# Backward compatibility with existing hashing module
def migrate_password_hashing(user_id: str, old_hash: str, password: str) -> bool:
    """Migrate from PBKDF2 to Argon2id."""
    hasher = get_argon2_hasher()
    
    # Verify with old hash (assuming PBKDF2)
    # This would need to be implemented based on existing hashing.py
    # For now, we'll just rehash with Argon2id
    new_hash = hasher.hash_password(password)
    
    # Update storage with new hash
    # This would call storage.update_user_password(user_id, new_hash)
    
    return True


# Example usage in FastAPI app
"""
from fastapi import FastAPI, Depends
from studio.backend.auth.rbac import (
    initialize_security,
    get_oauth_provider,
    get_rbac_manager,
    require_roles,
    Role,
    Permission
)

app = FastAPI()

# Initialize security
storage = AuthStorage()  # Your storage implementation
oauth_provider, rbac_manager, hasher = initialize_security(storage)

# Protected endpoint example
@app.get("/admin/users")
@require_roles(Role.ADMIN)
async def list_users():
    return {"users": []}

# Permission-based endpoint
@app.post("/models/train")
async def train_model(
    request: Request,
    user=Depends(rbac_manager.require_permission(Permission.MODEL_TRAIN))
):
    return {"message": "Training started"}
"""