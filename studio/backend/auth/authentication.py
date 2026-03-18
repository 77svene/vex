# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer
import jwt
from pydantic import BaseModel

from .storage import (
    get_jwt_secret,
    get_user_and_secret,
    load_jwt_secret,
    save_refresh_token,
    verify_refresh_token,
    log_auth_event,
    get_user_role,
    rotate_jwt_secret,
    verify_argon2id_hash,
    hash_argon2id,
)

# OAuth2/OIDC Configuration
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
REFRESH_TOKEN_EXPIRE_DAYS = 7
ISSUER = "vex-studio"
AUDIENCE = "vex-api"

# RBAC Roles
class UserRole(str, Enum):
    ADMIN = "admin"
    TRAINER = "trainer"
    VIEWER = "viewer"

# Audit Event Types
class AuditEventType(str, Enum):
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILURE = "auth.login.failure"
    TOKEN_REFRESH = "auth.token.refresh"
    TOKEN_REVOKED = "auth.token.revoked"
    ROLE_CHANGE = "auth.role.change"
    SECRET_ROTATION = "auth.secret.rotation"
    ACCESS_DENIED = "auth.access.denied"
    PASSWORD_CHANGE = "auth.password.change"

# Security schemes
security = HTTPBearer()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# Policy engine for fine-grained permissions
PERMISSIONS_POLICY = {
    UserRole.ADMIN: {
        "users": ["create", "read", "update", "delete"],
        "models": ["create", "read", "update", "delete", "train"],
        "datasets": ["create", "read", "update", "delete"],
        "jobs": ["create", "read", "update", "delete", "cancel"],
        "system": ["read", "update", "configure"],
        "audit": ["read"],
    },
    UserRole.TRAINER: {
        "users": ["read"],
        "models": ["create", "read", "update", "train"],
        "datasets": ["create", "read", "update"],
        "jobs": ["create", "read", "update", "cancel"],
        "system": ["read"],
        "audit": [],
    },
    UserRole.VIEWER: {
        "users": [],
        "models": ["read"],
        "datasets": ["read"],
        "jobs": ["read"],
        "system": ["read"],
        "audit": [],
    },
}

class TokenData(BaseModel):
    sub: str
    exp: datetime
    iat: datetime
    iss: str
    aud: Union[str, List[str]]
    roles: List[str] = []
    permissions: Dict[str, List[str]] = {}

class AuditLogEntry(BaseModel):
    timestamp: datetime
    event_type: str
    user_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    resource: Optional[str]
    action: Optional[str]
    success: bool
    metadata: Dict[str, Any] = {}

def _get_secret_for_subject(subject: str) -> str:
    """Get JWT secret with support for rotation."""
    secrets_list = get_jwt_secret(subject)
    if not secrets_list:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    # Use the first (current) secret for signing
    return secrets_list[0]

def _get_all_secrets_for_subject(subject: str) -> List[str]:
    """Get all valid JWT secrets for token verification (supports rotation)."""
    secrets_list = get_jwt_secret(subject)
    if not secrets_list:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return secrets_list

def _decode_subject_without_verification(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(
            token,
            options={"verify_signature": False, "verify_exp": False},
        )
    except jwt.InvalidTokenError:
        return None

    subject = payload.get("sub")
    return subject if isinstance(subject, str) else None

def _create_audit_log(
    event_type: AuditEventType,
    user_id: Optional[str] = None,
    request: Optional[Request] = None,
    resource: Optional[str] = None,
    action: Optional[str] = None,
    success: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Create structured audit log entry."""
    entry = AuditLogEntry(
        timestamp=datetime.now(timezone.utc),
        event_type=event_type.value,
        user_id=user_id,
        ip_address=request.client.host if request and request.client else None,
        user_agent=request.headers.get("user-agent") if request else None,
        resource=resource,
        action=action,
        success=success,
        metadata=metadata or {},
    )
    log_auth_event(entry.dict())

def _check_permission(role: UserRole, resource: str, action: str) -> bool:
    """Check if a role has permission to perform an action on a resource."""
    role_permissions = PERMISSIONS_POLICY.get(role, {})
    allowed_actions = role_permissions.get(resource, [])
    return action in allowed_actions

def create_access_token(
    subject: str,
    roles: Optional[List[UserRole]] = None,
    expires_delta: Optional[timedelta] = None,
    request: Optional[Request] = None,
) -> str:
    """
    Create a signed JWT for the given subject with RBAC claims.
    Supports OAuth2/OIDC standard claims.
    """
    if roles is None:
        # Get roles from storage
        user_role = get_user_role(subject)
        roles = [user_role] if user_role else [UserRole.VIEWER]
    
    # Build permissions from role
    permissions = {}
    for role in roles:
        role_perms = PERMISSIONS_POLICY.get(role, {})
        for resource, actions in role_perms.items():
            if resource not in permissions:
                permissions[resource] = []
            permissions[resource].extend(actions)
    
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    
    to_encode = {
        "sub": subject,
        "exp": expire,
        "iat": now,
        "iss": ISSUER,
        "aud": AUDIENCE,
        "roles": [role.value for role in roles],
        "permissions": permissions,
    }
    
    token = jwt.encode(
        to_encode,
        _get_secret_for_subject(subject),
        algorithm=ALGORITHM,
    )
    
    _create_audit_log(
        AuditEventType.LOGIN_SUCCESS,
        user_id=subject,
        request=request,
        metadata={"token_type": "access", "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60},
    )
    
    return token

def create_refresh_token(
    subject: str,
    request: Optional[Request] = None,
) -> str:
    """
    Create a random refresh token with Argon2id hashing.
    """
    token = secrets.token_urlsafe(48)
    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    # Hash the refresh token with Argon2id before storing
    token_hash = hash_argon2id(token)
    save_refresh_token(token_hash, subject, expires_at.isoformat())
    
    _create_audit_log(
        AuditEventType.LOGIN_SUCCESS,
        user_id=subject,
        request=request,
        metadata={"token_type": "refresh", "expires_in": REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60},
    )
    
    return token

def refresh_access_token(
    refresh_token: str,
    request: Optional[Request] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Validate a refresh token and issue a new access token.
    Uses Argon2id verification for refresh tokens.
    """
    # Hash the provided token and verify against stored hash
    token_hash = hash_argon2id(refresh_token)
    username = verify_refresh_token(token_hash)
    
    if username is None:
        _create_audit_log(
            AuditEventType.LOGIN_FAILURE,
            request=request,
            success=False,
            metadata={"reason": "invalid_refresh_token"},
        )
        return None, None
    
    _create_audit_log(
        AuditEventType.TOKEN_REFRESH,
        user_id=username,
        request=request,
    )
    
    return create_access_token(subject=username, request=request), username

def rotate_secrets(subject: str, request: Optional[Request] = None) -> bool:
    """
    Rotate JWT secrets for a user.
    Maintains previous secret for token verification during rotation.
    """
    success = rotate_jwt_secret(subject)
    if success:
        _create_audit_log(
            AuditEventType.SECRET_ROTATION,
            user_id=subject,
            request=request,
            metadata={"action": "jwt_secret_rotation"},
        )
    return success

def reload_secret() -> None:
    """Reload JWT secrets from storage."""
    load_jwt_secret()

async def get_current_subject(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request: Request = None,
) -> str:
    """Validate JWT and return the subject with audit logging."""
    return await _get_current_subject(
        credentials,
        request=request,
        allow_password_change=False,
    )

async def get_current_subject_allow_password_change(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request: Request = None,
) -> str:
    """Validate JWT but allow access to the password-change endpoint."""
    return await _get_current_subject(
        credentials,
        request=request,
        allow_password_change=True,
    )

async def get_current_user_with_role(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    request: Request = None,
) -> Tuple[str, UserRole]:
    """Validate JWT and return subject with role."""
    subject, roles = await _get_current_subject_with_roles(
        credentials,
        request=request,
    )
    # Return the highest privilege role
    if UserRole.ADMIN in roles:
        return subject, UserRole.ADMIN
    elif UserRole.TRAINER in roles:
        return subject, UserRole.TRAINER
    else:
        return subject, UserRole.VIEWER

async def require_role(required_role: UserRole):
    """Dependency factory for role-based access control."""
    async def role_checker(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        request: Request = None,
    ) -> Tuple[str, UserRole]:
        subject, user_role = await get_current_user_with_role(credentials, request)
        
        # Role hierarchy: admin > trainer > viewer
        role_hierarchy = {
            UserRole.ADMIN: 3,
            UserRole.TRAINER: 2,
            UserRole.VIEWER: 1,
        }
        
        if role_hierarchy.get(user_role, 0) < role_hierarchy.get(required_role, 0):
            _create_audit_log(
                AuditEventType.ACCESS_DENIED,
                user_id=subject,
                request=request,
                resource="role",
                action="access",
                success=False,
                metadata={
                    "required_role": required_role.value,
                    "user_role": user_role.value,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {required_role.value} required",
            )
        
        return subject, user_role
    
    return role_checker

async def require_permission(resource: str, action: str):
    """Dependency factory for fine-grained permission checks."""
    async def permission_checker(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        request: Request = None,
    ) -> str:
        subject, roles = await _get_current_subject_with_roles(credentials, request)
        
        # Check if any of the user's roles grant the required permission
        has_permission = False
        for role in roles:
            if _check_permission(role, resource, action):
                has_permission = True
                break
        
        if not has_permission:
            _create_audit_log(
                AuditEventType.ACCESS_DENIED,
                user_id=subject,
                request=request,
                resource=resource,
                action=action,
                success=False,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied for {action} on {resource}",
            )
        
        return subject
    
    return permission_checker

async def _get_current_subject(
    credentials: HTTPAuthorizationCredentials,
    request: Optional[Request] = None,
    *,
    allow_password_change: bool,
) -> str:
    """
    FastAPI dependency to validate the JWT and return the subject.
    """
    token = credentials.credentials
    subject = _decode_subject_without_verification(token)
    if subject is None:
        _create_audit_log(
            AuditEventType.LOGIN_FAILURE,
            request=request,
            success=False,
            metadata={"reason": "invalid_token_payload"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    record = get_user_and_secret(subject)
    if record is None:
        _create_audit_log(
            AuditEventType.LOGIN_FAILURE,
            user_id=subject,
            request=request,
            success=False,
            metadata={"reason": "user_not_found"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    # Unpack user record (assuming format: salt, pwd_hash, must_change_password, ...)
    _salt, _pwd_hash, must_change_password = record[:3]
    
    # Try all valid secrets for verification (supports rotation)
    secrets_list = _get_all_secrets_for_subject(subject)
    payload = None
    
    for secret in secrets_list:
        try:
            payload = jwt.decode(
                token,
                secret,
                algorithms=[ALGORITHM],
                issuer=ISSUER,
                audience=AUDIENCE,
            )
            break
        except jwt.InvalidTokenError:
            continue
    
    if payload is None:
        _create_audit_log(
            AuditEventType.LOGIN_FAILURE,
            user_id=subject,
            request=request,
            success=False,
            metadata={"reason": "invalid_token_signature"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    
    if payload.get("sub") != subject:
        _create_audit_log(
            AuditEventType.LOGIN_FAILURE,
            user_id=subject,
            request=request,
            success=False,
            metadata={"reason": "subject_mismatch"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    
    if must_change_password and not allow_password_change:
        _create_audit_log(
            AuditEventType.ACCESS_DENIED,
            user_id=subject,
            request=request,
            resource="password",
            action="change_required",
            success=False,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Password change required",
        )
    
    return subject

async def _get_current_subject_with_roles(
    credentials: HTTPAuthorizationCredentials,
    request: Optional[Request] = None,
) -> Tuple[str, List[UserRole]]:
    """Validate JWT and return subject with roles."""
    token = credentials.credentials
    subject = _decode_subject_without_verification(token)
    if subject is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    record = get_user_and_secret(subject)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    # Verify token with all valid secrets
    secrets_list = _get_all_secrets_for_subject(subject)
    payload = None
    
    for secret in secrets_list:
        try:
            payload = jwt.decode(
                token,
                secret,
                algorithms=[ALGORITHM],
                issuer=ISSUER,
                audience=AUDIENCE,
            )
            break
        except jwt.InvalidTokenError:
            continue
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    
    # Extract roles from token
    roles_str = payload.get("roles", [])
    roles = []
    for role_str in roles_str:
        try:
            roles.append(UserRole(role_str))
        except ValueError:
            continue
    
    if not roles:
        roles = [UserRole.VIEWER]  # Default role
    
    return subject, roles

async def verify_oauth2_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify an OAuth2/OIDC token from external provider.
    This can be used to integrate with Authelia or other OAuth2 providers.
    """
    try:
        # This would be configured with the provider's JWKS endpoint
        # For now, we'll decode without verification as an example
        payload = jwt.decode(
            token,
            options={
                "verify_signature": False,  # Would be True with proper JWKS
                "verify_exp": True,
                "verify_iss": True,
                "verify_aud": True,
            },
            issuer="external-provider",  # Configure with actual issuer
            audience="vex-studio",   # Configure with actual audience
        )
        return payload
    except jwt.InvalidTokenError:
        return None