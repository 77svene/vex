"""
Enterprise Security & RBAC Module for SOVEREIGN Studio
Implements OAuth2/OIDC, Role-Based Access Control, Audit Logging, and Secret Rotation
"""

import os
import json
import time
import uuid
import secrets
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Set, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from functools import wraps

from fastapi import FastAPI, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2AuthorizationCodeBearer
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, DateTime, Boolean, JSON, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import httpx

from studio.backend.auth.storage import get_db, UserDB
from studio.backend.auth.hashing import hash_password, verify_password
from studio.backend.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
AUDIT_LOG_RETENTION_DAYS = 365
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_MINUTES = 15

# Argon2id configuration (replacing PBKDF2)
ARGON2_TIME_COST = 3
ARGON2_MEMORY_COST = 65536  # 64MB
ARGON2_PARALLELISM = 4
ARGON2_HASH_LENGTH = 32
ARGON2_SALT_LENGTH = 16

# OAuth2/OIDC Configuration
OAUTH2_PROVIDERS = {
    "authelia": {
        "authorization_url": os.getenv("AUTHELIA_AUTH_URL", "https://auth.example.com/api/oidc/authorization"),
        "token_url": os.getenv("AUTHELIA_TOKEN_URL", "https://auth.example.com/api/oidc/token"),
        "userinfo_url": os.getenv("AUTHELIA_USERINFO_URL", "https://auth.example.com/api/oidc/userinfo"),
        "client_id": os.getenv("AUTHELIA_CLIENT_ID"),
        "client_secret": os.getenv("AUTHELIA_CLIENT_SECRET"),
        "scopes": ["openid", "profile", "email"],
    },
    "internal": {
        "authorization_url": "/oauth/authorize",
        "token_url": "/oauth/token",
        "userinfo_url": "/oauth/userinfo",
        "client_id": os.getenv("INTERNAL_OAUTH_CLIENT_ID", "sovereign-studio"),
        "client_secret": os.getenv("INTERNAL_OAUTH_CLIENT_SECRET", secrets.token_urlsafe(32)),
        "scopes": ["openid", "profile", "email", "admin"],
    }
}

# RBAC Roles and Permissions
class Role(str, Enum):
    ADMIN = "admin"
    TRAINER = "trainer"
    VIEWER = "viewer"
    SYSTEM = "system"

class Permission(str, Enum):
    # User Management
    CREATE_USER = "user:create"
    READ_USER = "user:read"
    UPDATE_USER = "user:update"
    DELETE_USER = "user:delete"
    
    # Model Management
    CREATE_MODEL = "model:create"
    READ_MODEL = "model:read"
    UPDATE_MODEL = "model:update"
    DELETE_MODEL = "model:delete"
    TRAIN_MODEL = "model:train"
    DEPLOY_MODEL = "model:deploy"
    
    # Data Management
    CREATE_DATA = "data:create"
    READ_DATA = "data:read"
    UPDATE_DATA = "data:update"
    DELETE_DATA = "data:delete"
    
    # System Administration
    VIEW_AUDIT_LOGS = "audit:read"
    MANAGE_SECRETS = "secrets:manage"
    MANAGE_ROLES = "roles:manage"
    SYSTEM_CONFIG = "system:config"
    
    # API Access
    API_ACCESS = "api:access"
    API_ADMIN = "api:admin"

# Role-Permission Mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: {
        Permission.CREATE_USER, Permission.READ_USER, Permission.UPDATE_USER, Permission.DELETE_USER,
        Permission.CREATE_MODEL, Permission.READ_MODEL, Permission.UPDATE_MODEL, Permission.DELETE_MODEL,
        Permission.TRAIN_MODEL, Permission.DEPLOY_MODEL,
        Permission.CREATE_DATA, Permission.READ_DATA, Permission.UPDATE_DATA, Permission.DELETE_DATA,
        Permission.VIEW_AUDIT_LOGS, Permission.MANAGE_SECRETS, Permission.MANAGE_ROLES, Permission.SYSTEM_CONFIG,
        Permission.API_ACCESS, Permission.API_ADMIN,
    },
    Role.TRAINER: {
        Permission.READ_USER,
        Permission.CREATE_MODEL, Permission.READ_MODEL, Permission.UPDATE_MODEL, Permission.TRAIN_MODEL,
        Permission.CREATE_DATA, Permission.READ_DATA, Permission.UPDATE_DATA,
        Permission.API_ACCESS,
    },
    Role.VIEWER: {
        Permission.READ_USER,
        Permission.READ_MODEL,
        Permission.READ_DATA,
        Permission.API_ACCESS,
    },
    Role.SYSTEM: {
        Permission.API_ACCESS, Permission.API_ADMIN,
        Permission.MANAGE_SECRETS, Permission.SYSTEM_CONFIG,
    }
}

# Database Models
Base = declarative_base()

class OAuthClient(Base):
    """OAuth2 Client Registration"""
    __tablename__ = "oauth_clients"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(String(255), unique=True, nullable=False, index=True)
    client_secret_hash = Column(String(512), nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    redirect_uris = Column(JSON, default=list)
    allowed_scopes = Column(JSON, default=list)
    allowed_roles = Column(JSON, default=[Role.VIEWER.value])
    is_confidential = Column(Boolean, default=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    secret_expires_at = Column(DateTime)
    metadata_ = Column("metadata", JSON, default=dict)

class UserSession(Base):
    """Active User Sessions"""
    __tablename__ = "user_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    session_token = Column(String(512), unique=True, nullable=False, index=True)
    refresh_token_hash = Column(String(512), unique=True, nullable=False)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    scopes = Column(JSON, default=list)
    roles = Column(JSON, default=list)
    expires_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class AuditLog(Base):
    """Structured Audit Trail"""
    __tablename__ = "audit_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_id = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    event_type = Column(String(100), nullable=False, index=True)
    user_id = Column(String(36), index=True)
    username = Column(String(255))
    ip_address = Column(String(45))
    user_agent = Column(Text)
    resource_type = Column(String(100))
    resource_id = Column(String(255))
    action = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False)  # success, failure, error
    details = Column(JSON, default=dict)
    risk_score = Column(Integer, default=0)  # 0-100
    tags = Column(JSON, default=list)
    session_id = Column(String(36))

class RoleAssignment(Base):
    """User Role Assignments"""
    __tablename__ = "role_assignments"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    role = Column(SQLEnum(Role), nullable=False)
    assigned_by = Column(String(36))
    assigned_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)
    conditions = Column(JSON, default=dict)  # For attribute-based access control

class SecretRotation(Base):
    """Secret Rotation History"""
    __tablename__ = "secret_rotations"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    secret_type = Column(String(100), nullable=False)  # jwt_key, oauth_client, encryption_key
    secret_id = Column(String(255))
    old_hash = Column(String(512))
    new_hash = Column(String(512))
    rotated_by = Column(String(36))
    rotated_at = Column(DateTime, default=datetime.utcnow)
    rotation_reason = Column(String(255))
    metadata_ = Column("metadata", JSON, default=dict)

# Pydantic Models
class TokenData(BaseModel):
    sub: str
    exp: datetime
    iat: datetime
    iss: str = "sovereign-studio"
    aud: str = "sovereign-api"
    scope: str = ""
    roles: List[str] = []
    client_id: Optional[str] = None
    session_id: Optional[str] = None

class OAuth2TokenRequest(BaseModel):
    grant_type: str
    code: Optional[str] = None
    redirect_uri: Optional[str] = None
    client_id: str
    client_secret: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    scope: Optional[str] = None
    refresh_token: Optional[str] = None

class OAuth2TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None
    scope: str
    roles: List[str] = []

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    password: str = Field(..., min_length=12)
    full_name: Optional[str] = None
    roles: List[Role] = [Role.VIEWER]

class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    full_name: Optional[str]
    roles: List[str]
    is_active: bool
    created_at: datetime
    last_login: Optional[datetime]

class AuditEvent(BaseModel):
    event_type: str
    user_id: Optional[str] = None
    username: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: str
    status: str = "success"
    details: Dict[str, Any] = {}
    risk_score: int = 0
    tags: List[str] = []

# Security Utilities
class Argon2Hasher:
    """Argon2id password hashing (replacing PBKDF2)"""
    
    def __init__(self):
        self.context = CryptContext(
            schemes=["argon2"],
            default="argon2",
            argon2__time_cost=ARGON2_TIME_COST,
            argon2__memory_cost=ARGON2_MEMORY_COST,
            argon2__parallelism=ARGON2_PARALLELISM,
            argon2__hash_len=ARGON2_HASH_LENGTH,
            argon2__salt_len=ARGON2_SALT_LENGTH,
        )
    
    def hash(self, password: str) -> str:
        return self.context.hash(password)
    
    def verify(self, password: str, hashed: str) -> bool:
        return self.context.verify(password, hashed)
    
    def needs_update(self, hashed: str) -> bool:
        return self.context.needs_update(hashed)

class AuditLogger:
    """Structured audit logging system"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.logger = logging.getLogger("audit")
    
    def log(self, event: AuditEvent, request: Request = None):
        """Log an audit event"""
        try:
            # Enrich event with request context
            if request:
                event.details.update({
                    "method": request.method,
                    "path": request.url.path,
                    "query_params": dict(request.query_params),
                    "headers": {
                        "user-agent": request.headers.get("user-agent"),
                        "x-forwarded-for": request.headers.get("x-forwarded-for"),
                    }
                })
                event.ip_address = request.client.host if request.client else None
                event.user_agent = request.headers.get("user-agent")
            
            # Calculate risk score
            event.risk_score = self._calculate_risk_score(event)
            
            # Store in database
            db_event = AuditLog(
                event_type=event.event_type,
                user_id=event.user_id,
                username=event.username,
                ip_address=event.ip_address,
                user_agent=event.user_agent,
                resource_type=event.resource_type,
                resource_id=event.resource_id,
                action=event.action,
                status=event.status,
                details=event.details,
                risk_score=event.risk_score,
                tags=event.tags,
            )
            self.db.add(db_event)
            self.db.commit()
            
            # Log to application logs
            log_data = asdict(event)
            if event.status == "failure":
                self.logger.warning(f"AUDIT: {json.dumps(log_data)}")
            elif event.risk_score > 70:
                self.logger.error(f"AUDIT HIGH RISK: {json.dumps(log_data)}")
            else:
                self.logger.info(f"AUDIT: {json.dumps(log_data)}")
                
        except Exception as e:
            self.logger.error(f"Failed to log audit event: {e}")
            self.db.rollback()
    
    def _calculate_risk_score(self, event: AuditEvent) -> int:
        """Calculate risk score for an event (0-100)"""
        score = 0
        
        # Failed authentication attempts
        if event.event_type == "authentication" and event.status == "failure":
            score += 30
        
        # Admin actions
        if "admin" in event.action.lower():
            score += 20
        
        # Sensitive operations
        sensitive_ops = ["delete", "update", "create", "rotate", "config"]
        if any(op in event.action.lower() for op in sensitive_ops):
            score += 15
        
        # Multiple rapid events (would need rate limiting context)
        if "rapid" in event.tags:
            score += 25
        
        # Unusual hours (outside business hours)
        hour = datetime.utcnow().hour
        if hour < 6 or hour > 22:
            score += 10
        
        return min(score, 100)
    
    def get_events(self, 
                   start_time: datetime = None,
                   end_time: datetime = None,
                   user_id: str = None,
                   event_type: str = None,
                   min_risk_score: int = None,
                   limit: int = 100) -> List[AuditLog]:
        """Query audit logs with filters"""
        query = self.db.query(AuditLog)
        
        if start_time:
            query = query.filter(AuditLog.timestamp >= start_time)
        if end_time:
            query = query.filter(AuditLog.timestamp <= end_time)
        if user_id:
            query = query.filter(AuditLog.user_id == user_id)
        if event_type:
            query = query.filter(AuditLog.event_type == event_type)
        if min_risk_score:
            query = query.filter(AuditLog.risk_score >= min_risk_score)
        
        return query.order_by(AuditLog.timestamp.desc()).limit(limit).all()

class PolicyEngine:
    """Fine-grained permission policy engine"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def check_permission(self, 
                        user_id: str, 
                        permission: Permission,
                        resource_type: str = None,
                        resource_id: str = None,
                        context: Dict[str, Any] = None) -> bool:
        """Check if user has permission for an action"""
        # Get user's active roles
        roles = self._get_user_roles(user_id)
        
        # Check role-based permissions
        for role in roles:
            if permission in ROLE_PERMISSIONS.get(role, set()):
                # Check for resource-specific conditions
                if self._check_conditions(user_id, role, permission, resource_type, resource_id, context):
                    return True
        
        return False
    
    def _get_user_roles(self, user_id: str) -> List[Role]:
        """Get active roles for a user"""
        assignments = self.db.query(RoleAssignment).filter(
            RoleAssignment.user_id == user_id,
            RoleAssignment.is_active == True,
            (RoleAssignment.expires_at == None) | (RoleAssignment.expires_at > datetime.utcnow())
        ).all()
        
        return [assignment.role for assignment in assignments]
    
    def _check_conditions(self,
                         user_id: str,
                         role: Role,
                         permission: Permission,
                         resource_type: str,
                         resource_id: str,
                         context: Dict[str, Any]) -> bool:
        """Check attribute-based access control conditions"""
        # Get role assignment with conditions
        assignment = self.db.query(RoleAssignment).filter(
            RoleAssignment.user_id == user_id,
            RoleAssignment.role == role,
            RoleAssignment.is_active == True
        ).first()
        
        if not assignment or not assignment.conditions:
            return True  # No additional conditions
        
        conditions = assignment.conditions
        
        # Example conditions implementation
        # Time-based access
        if "allowed_hours" in conditions:
            current_hour = datetime.utcnow().hour
            allowed_hours = conditions["allowed_hours"]
            if not (allowed_hours["start"] <= current_hour <= allowed_hours["end"]):
                return False
        
        # IP-based restrictions
        if "allowed_ips" in conditions and context and "ip_address" in context:
            if context["ip_address"] not in conditions["allowed_ips"]:
                return False
        
        # Resource ownership
        if "owner_only" in conditions and conditions["owner_only"]:
            if resource_type and resource_id:
                # Check if user owns the resource
                # This would need to be implemented per resource type
                pass
        
        return True

class SecretManager:
    """Secret rotation and management"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.hasher = Argon2Hasher()
    
    def rotate_jwt_secret(self, reason: str = "scheduled_rotation") -> str:
        """Rotate JWT signing secret"""
        old_secret = SECRET_KEY
        new_secret = secrets.token_urlsafe(32)
        
        # Store rotation history
        rotation = SecretRotation(
            secret_type="jwt_key",
            secret_id="main_jwt_secret",
            old_hash=self.hasher.hash(old_secret),
            new_hash=self.hasher.hash(new_secret),
            rotated_by="system",
            rotation_reason=reason,
            metadata_={"algorithm": ALGORITHM}
        )
        self.db.add(rotation)
        self.db.commit()
        
        # Update environment (in production, this would update a secure vault)
        os.environ["SECRET_KEY"] = new_secret
        global SECRET_KEY
        SECRET_KEY = new_secret
        
        logger.info(f"JWT secret rotated: {reason}")
        return new_secret
    
    def rotate_oauth_client_secret(self, client_id: str, reason: str = "scheduled_rotation") -> str:
        """Rotate OAuth client secret"""
        client = self.db.query(OAuthClient).filter(
            OAuthClient.client_id == client_id
        ).first()
        
        if not client:
            raise ValueError(f"OAuth client not found: {client_id}")
        
        old_secret_hash = client.client_secret_hash
        new_secret = secrets.token_urlsafe(32)
        new_secret_hash = self.hasher.hash(new_secret)
        
        # Update client
        client.client_secret_hash = new_secret_hash
        client.secret_expires_at = datetime.utcnow() + timedelta(days=90)
        client.updated_at = datetime.utcnow()
        
        # Store rotation history
        rotation = SecretRotation(
            secret_type="oauth_client",
            secret_id=client_id,
            old_hash=old_secret_hash,
            new_hash=new_secret_hash,
            rotated_by="system",
            rotation_reason=reason,
            metadata_={"client_name": client.name}
        )
        self.db.add(rotation)
        self.db.commit()
        
        logger.info(f"OAuth client secret rotated for {client_id}: {reason}")
        return new_secret
    
    def schedule_rotation(self):
        """Schedule automatic secret rotation"""
        # Rotate JWT secret every 30 days
        jwt_rotations = self.db.query(SecretRotation).filter(
            SecretRotation.secret_type == "jwt_key",
            SecretRotation.rotated_at >= datetime.utcnow() - timedelta(days=30)
        ).count()
        
        if jwt_rotations == 0:
            self.rotate_jwt_secret("scheduled_30day_rotation")
        
        # Rotate OAuth client secrets expiring in next 7 days
        expiring_clients = self.db.query(OAuthClient).filter(
            OAuthClient.secret_expires_at <= datetime.utcnow() + timedelta(days=7),
            OAuthClient.is_active == True
        ).all()
        
        for client in expiring_clients:
            try:
                self.rotate_oauth_client_secret(client.client_id, "expiring_secret")
            except Exception as e:
                logger.error(f"Failed to rotate secret for client {client.client_id}: {e}")

class OAuth2Provider:
    """OAuth2/OIDC Provider Implementation"""
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self.hasher = Argon2Hasher()
        self.audit = AuditLogger(db_session)
        self.policy = PolicyEngine(db_session)
        self.secrets = SecretManager(db_session)
        
        # OAuth2 configuration
        self.oauth2_scheme = OAuth2PasswordBearer(tokenUrl="oauth/token")
        self.oauth2_code_scheme = OAuth2AuthorizationCodeBearer(
            authorizationUrl="oauth/authorize",
            tokenUrl="oauth/token",
            refreshUrl="oauth/token",
        )
    
    def create_access_token(self, 
                           data: Dict[str, Any], 
                           expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "iss": "sovereign-studio",
            "aud": "sovereign-api",
            "jti": str(uuid.uuid4()),  # JWT ID for revocation
        })
        
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, user_id: str, session_id: str) -> str:
        """Create refresh token"""
        token_data = {
            "sub": user_id,
            "session_id": session_id,
            "type": "refresh",
            "exp": datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4()),
        }
        
        return jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
    
    def verify_token(self, token: str) -> TokenData:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token, 
                SECRET_KEY, 
                algorithms=[ALGORITHM],
                audience="sovereign-api",
                issuer="sovereign-studio"
            )
            
            # Check token revocation (would need a revocation list in production)
            # For now, we'll just validate the structure
            
            return TokenData(**payload)
            
        except JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def authenticate_user(self, 
                         username: str, 
                         password: str,
                         client_id: str = None,
                         ip_address: str = None,
                         user_agent: str = None) -> Optional[Dict[str, Any]]:
        """Authenticate user with username/password"""
        # Get user from database
        user = self.db.query(UserDB).filter(UserDB.username == username).first()
        
        if not user:
            self._log_auth_event(
                username=username,
                event_type="authentication",
                action="login_attempt",
                status="failure",
                details={"reason": "user_not_found"},
                ip_address=ip_address,
                user_agent=user_agent
            )
            return None
        
        # Check if account is locked
        if self._is_account_locked(user):
            self._log_auth_event(
                user_id=str(user.id),
                username=username,
                event_type="authentication",
                action="login_attempt",
                status="failure",
                details={"reason": "account_locked"},
                ip_address=ip_address,
                user_agent=user_agent
            )
            return None
        
        # Verify password
        if not self.hasher.verify(password, user.password_hash):
            self._increment_failed_attempts(user)
            self._log_auth_event(
                user_id=str(user.id),
                username=username,
                event_type="authentication",
                action="login_attempt",
                status="failure",
                details={"reason": "invalid_password"},
                ip_address=ip_address,
                user_agent=user_agent
            )
            return None
        
        # Reset failed attempts on successful login
        self._reset_failed_attempts(user)
        
        # Get user roles
        roles = self.policy._get_user_roles(user.id)
        
        # Create session
        session_id = str(uuid.uuid4())
        session_token = secrets.token_urlsafe(32)
        refresh_token = self.create_refresh_token(str(user.id), session_id)
        
        user_session = UserSession(
            user_id=user.id,
            session_token=session_token,
            refresh_token_hash=self.hasher.hash(refresh_token),
            ip_address=ip_address,
            user_agent=user_agent,
            scopes=["openid", "profile", "email"],
            roles=[role.value for role in roles],
            expires_at=datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        )
        self.db.add(user_session)
        
        # Update last login
        user.last_login = datetime.utcnow()
        self.db.commit()
        
        # Create access token
        access_token = self.create_access_token({
            "sub": str(user.id),
            "scope": "openid profile email",
            "roles": [role.value for role in roles],
            "session_id": session_id,
            "client_id": client_id or "internal",
        })
        
        # Log successful authentication
        self._log_auth_event(
            user_id=str(user.id),
            username=username,
            event_type="authentication",
            action="login_success",
            status="success",
            details={"session_id": session_id, "client_id": client_id},
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "scope": "openid profile email",
            "roles": [role.value for role in roles],
            "user_id": str(user.id),
            "session_id": session_id,
        }
    
    def refresh_access_token(self, 
                            refresh_token: str,
                            client_id: str = None,
                            ip_address: str = None) -> Dict[str, Any]:
        """Refresh access token using refresh token"""
        try:
            # Decode refresh token
            payload = jwt.decode(
                refresh_token,
                SECRET_KEY,
                algorithms=[ALGORITHM]
            )
            
            user_id = payload.get("sub")
            session_id = payload.get("session_id")
            
            if not user_id or not session_id:
                raise HTTPException(status_code=400, detail="Invalid refresh token")
            
            # Get session from database
            session = self.db.query(UserSession).filter(
                UserSession.session_id == session_id,
                UserSession.user_id == user_id,
                UserSession.is_active == True
            ).first()
            
            if not session:
                raise HTTPException(status_code=400, detail="Session not found or expired")
            
            # Verify refresh token hash
            if not self.hasher.verify(refresh_token, session.refresh_token_hash):
                raise HTTPException(status_code=400, detail="Invalid refresh token")
            
            # Check session expiration
            if session.expires_at < datetime.utcnow():
                session.is_active = False
                self.db.commit()
                raise HTTPException(status_code=400, detail="Session expired")
            
            # Update session activity
            session.last_activity = datetime.utcnow()
            
            # Create new access token
            user = self.db.query(UserDB).filter(UserDB.id == user_id).first()
            roles = self.policy._get_user_roles(user_id)
            
            access_token = self.create_access_token({
                "sub": str(user_id),
                "scope": " ".join(session.scopes),
                "roles": [role.value for role in roles],
                "session_id": session_id,
                "client_id": client_id or "internal",
            })
            
            self.db.commit()
            
            # Log token refresh
            self._log_auth_event(
                user_id=str(user_id),
                username=user.username,
                event_type="authentication",
                action="token_refresh",
                status="success",
                details={"session_id": session_id},
                ip_address=ip_address
            )
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
                "scope": " ".join(session.scopes),
                "roles": [role.value for role in roles],
            }
            
        except JWTError:
            raise HTTPException(status_code=400, detail="Invalid refresh token")
    
    def revoke_session(self, session_id: str, user_id: str = None):
        """Revoke a user session"""
        query = self.db.query(UserSession).filter(UserSession.session_id == session_id)
        
        if user_id:
            query = query.filter(UserSession.user_id == user_id)
        
        session = query.first()
        
        if session:
            session.is_active = False
            self.db.commit()
            
            # Log session revocation
            user = self.db.query(UserDB).filter(UserDB.id == session.user_id).first()
            self._log_auth_event(
                user_id=str(session.user_id),
                username=user.username if user else None,
                event_type="session",
                action="session_revoked",
                status="success",
                details={"session_id": session_id}
            )
    
    def _is_account_locked(self, user: UserDB) -> bool:
        """Check if account is locked due to failed attempts"""
        if hasattr(user, 'failed_login_attempts') and user.failed_login_attempts >= MAX_LOGIN_ATTEMPTS:
            if hasattr(user, 'last_failed_login'):
                lockout_time = user.last_failed_login + timedelta(minutes=LOCKOUT_MINUTES)
                if datetime.utcnow() < lockout_time:
                    return True
        return False
    
    def _increment_failed_attempts(self, user: UserDB):
        """Increment failed login attempts"""
        if hasattr(user, 'failed_login_attempts'):
            user.failed_login_attempts += 1
        else:
            user.failed_login_attempts = 1
        
        user.last_failed_login = datetime.utcnow()
        self.db.commit()
    
    def _reset_failed_attempts(self, user: UserDB):
        """Reset failed login attempts"""
        if hasattr(user, 'failed_login_attempts'):
            user.failed_login_attempts = 0
        self.db.commit()
    
    def _log_auth_event(self, 
                       event_type: str,
                       action: str,
                       status: str,
                       details: Dict[str, Any] = None,
                       user_id: str = None,
                       username: str = None,
                       ip_address: str = None,
                       user_agent: str = None):
        """Log authentication event"""
        event = AuditEvent(
            event_type=event_type,
            user_id=user_id,
            username=username,
            action=action,
            status=status,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
        )
        self.audit.log(event)

# FastAPI Dependencies
def get_current_user(token: str = Depends(OAuth2PasswordBearer(tokenUrl="oauth/token"))):
    """Dependency to get current authenticated user"""
    provider = OAuth2Provider(next(get_db()))
    token_data = provider.verify_token(token)
    
    # Get user from database
    db = next(get_db())
    user = db.query(UserDB).filter(UserDB.id == token_data.sub).first()
    
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return user

def require_permissions(*permissions: Permission):
    """Dependency to require specific permissions"""
    def permission_checker(
        current_user: UserDB = Depends(get_current_user),
        request: Request = None
    ):
        db = next(get_db())
        policy = PolicyEngine(db)
        
        for permission in permissions:
            if not policy.check_permission(
                user_id=str(current_user.id),
                permission=permission,
                context={"ip_address": request.client.host if request else None}
            ):
                # Log permission denied
                audit = AuditLogger(db)
                audit.log(AuditEvent(
                    event_type="authorization",
                    user_id=str(current_user.id),
                    username=current_user.username,
                    action=f"permission_denied:{permission.value}",
                    status="failure",
                    details={"required_permission": permission.value},
                    risk_score=40,
                ), request)
                
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission denied: {permission.value}"
                )
        
        return current_user
    
    return permission_checker

def require_role(*roles: Role):
    """Dependency to require specific roles"""
    def role_checker(
        current_user: UserDB = Depends(get_current_user),
        request: Request = None
    ):
        db = next(get_db())
        policy = PolicyEngine(db)
        user_roles = policy._get_user_roles(str(current_user.id))
        
        if not any(role in user_roles for role in roles):
            # Log role denied
            audit = AuditLogger(db)
            audit.log(AuditEvent(
                event_type="authorization",
                user_id=str(current_user.id),
                username=current_user.username,
                action="role_denied",
                status="failure",
                details={
                    "required_roles": [r.value for r in roles],
                    "user_roles": [r.value for r in user_roles]
                },
                risk_score=50,
            ), request)
            
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient role privileges"
            )
        
        return current_user
    
    return role_checker

# FastAPI Router
from fastapi import APIRouter

router = APIRouter(prefix="/oauth", tags=["oauth"])

@router.post("/token", response_model=OAuth2TokenResponse)
async def oauth_token(
    request: Request,
    form_data: OAuth2TokenRequest,
    db: Session = Depends(get_db)
):
    """OAuth2 token endpoint"""
    provider = OAuth2Provider(db)
    
    if form_data.grant_type == "password":
        # Resource Owner Password Credentials Grant
        result = provider.authenticate_user(
            username=form_data.username,
            password=form_data.password,
            client_id=form_data.client_id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return OAuth2TokenResponse(
            access_token=result["access_token"],
            token_type="bearer",
            expires_in=result["expires_in"],
            refresh_token=result["refresh_token"],
            scope=result["scope"],
            roles=result["roles"]
        )
    
    elif form_data.grant_type == "refresh_token":
        # Refresh Token Grant
        result = provider.refresh_access_token(
            refresh_token=form_data.refresh_token,
            client_id=form_data.client_id,
            ip_address=request.client.host if request.client else None
        )
        
        return OAuth2TokenResponse(
            access_token=result["access_token"],
            token_type="bearer",
            expires_in=result["expires_in"],
            scope=result["scope"],
            roles=result["roles"]
        )
    
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported grant type"
        )

@router.post("/revoke")
async def revoke_token(
    request: Request,
    token: str = None,
    token_type_hint: str = "access_token",
    current_user: UserDB = Depends(require_permissions(Permission.MANAGE_SECRETS)),
    db: Session = Depends(get_db)
):
    """Revoke an OAuth2 token"""
    provider = OAuth2Provider(db)
    
    if token_type_hint == "refresh_token":
        # For refresh tokens, we need to find and deactivate the session
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            session_id = payload.get("session_id")
            user_id = payload.get("sub")
            
            if session_id:
                provider.revoke_session(session_id, user_id)
        except JWTError:
            pass
    
    # Log revocation
    audit = AuditLogger(db)
    audit.log(AuditEvent(
        event_type="token",
        action="token_revoked",
        status="success",
        details={
            "token_type": token_type_hint,
            "revoked_by": str(current_user.id)
        }
    ), request)
    
    return {"status": "ok"}

@router.get("/authorize")
async def oauth_authorize(
    request: Request,
    response_type: str,
    client_id: str,
    redirect_uri: str,
    scope: str = "openid",
    state: str = None,
    current_user: UserDB = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """OAuth2 authorization endpoint"""
    # Validate client
    client = db.query(OAuthClient).filter(
        OAuthClient.client_id == client_id,
        OAuthClient.is_active == True
    ).first()
    
    if not client:
        raise HTTPException(status_code=400, detail="Invalid client")
    
    # Validate redirect URI
    if redirect_uri not in client.redirect_uris:
        raise HTTPException(status_code=400, detail="Invalid redirect URI")
    
    # Generate authorization code
    code = secrets.token_urlsafe(32)
    
    # Store authorization code (in production, use Redis with TTL)
    # For now, we'll just return it
    
    # Log authorization
    audit = AuditLogger(db)
    audit.log(AuditEvent(
        event_type="authorization",
        user_id=str(current_user.id),
        username=current_user.username,
        action="authorization_granted",
        status="success",
        details={
            "client_id": client_id,
            "scope": scope,
            "redirect_uri": redirect_uri
        }
    ), request)
    
    # In production, redirect with code and state
    return {
        "code": code,
        "state": state,
        "redirect_uri": redirect_uri,
        "scope": scope
    }

@router.post("/rotate-secrets")
async def rotate_secrets(
    request: Request,
    secret_type: str = "all",
    current_user: UserDB = Depends(require_role(Role.ADMIN)),
    db: Session = Depends(get_db)
):
    """Rotate secrets (admin only)"""
    secret_manager = SecretManager(db)
    
    rotated = []
    
    if secret_type in ["all", "jwt"]:
        try:
            new_secret = secret_manager.rotate_jwt_secret("manual_admin_rotation")
            rotated.append({"type": "jwt", "status": "rotated"})
        except Exception as e:
            rotated.append({"type": "jwt", "status": "failed", "error": str(e)})
    
    if secret_type in ["all", "oauth_clients"]:
        # Rotate all active OAuth client secrets
        clients = db.query(OAuthClient).filter(OAuthClient.is_active == True).all()
        for client in clients:
            try:
                new_secret = secret_manager.rotate_oauth_client_secret(
                    client.client_id, 
                    "manual_admin_rotation"
                )
                rotated.append({
                    "type": "oauth_client",
                    "client_id": client.client_id,
                    "status": "rotated"
                })
            except Exception as e:
                rotated.append({
                    "type": "oauth_client",
                    "client_id": client.client_id,
                    "status": "failed",
                    "error": str(e)
                })
    
    # Log secret rotation
    audit = AuditLogger(db)
    audit.log(AuditEvent(
        event_type="security",
        user_id=str(current_user.id),
        username=current_user.username,
        action="secrets_rotated",
        status="success",
        details={
            "secret_type": secret_type,
            "rotated_items": len(rotated),
            "results": rotated
        },
        risk_score=60,
    ), request)
    
    return {
        "status": "completed",
        "rotated": rotated,
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/audit-logs")
async def get_audit_logs(
    request: Request,
    start_time: datetime = None,
    end_time: datetime = None,
    user_id: str = None,
    event_type: str = None,
    min_risk_score: int = None,
    limit: int = 100,
    current_user: UserDB = Depends(require_permissions(Permission.VIEW_AUDIT_LOGS)),
    db: Session = Depends(get_db)
):
    """Query audit logs (requires audit:read permission)"""
    audit = AuditLogger(db)
    
    # Log that someone accessed audit logs
    audit.log(AuditEvent(
        event_type="audit",
        user_id=str(current_user.id),
        username=current_user.username,
        action="audit_logs_accessed",
        status="success",
        details={
            "filters": {
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": end_time.isoformat() if end_time else None,
                "user_id": user_id,
                "event_type": event_type,
                "min_risk_score": min_risk_score,
            }
        }
    ), request)
    
    events = audit.get_events(
        start_time=start_time,
        end_time=end_time,
        user_id=user_id,
        event_type=event_type,
        min_risk_score=min_risk_score,
        limit=limit
    )
    
    return {
        "events": [
            {
                "id": str(event.id),
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "user_id": event.user_id,
                "username": event.username,
                "action": event.action,
                "status": event.status,
                "risk_score": event.risk_score,
                "details": event.details,
                "tags": event.tags,
            }
            for event in events
        ],
        "total": len(events),
        "filters_applied": {
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "user_id": user_id,
            "event_type": event_type,
            "min_risk_score": min_risk_score,
        }
    }

# Middleware for automatic audit logging
class AuditMiddleware:
    """Middleware to automatically log API requests"""
    
    def __init__(self, app: FastAPI):
        self.app = app
        
        # Add middleware
        @app.middleware("http")
        async def audit_log_middleware(request: Request, call_next):
            start_time = time.time()
            
            # Get user from token if present
            user_id = None
            username = None
            token = request.headers.get("Authorization", "").replace("Bearer ", "")
            
            if token:
                try:
                    provider = OAuth2Provider(next(get_db()))
                    token_data = provider.verify_token(token)
                    user_id = token_data.sub
                    
                    # Get username
                    db = next(get_db())
                    user = db.query(UserDB).filter(UserDB.id == user_id).first()
                    if user:
                        username = user.username
                except:
                    pass
            
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = (time.time() - start_time) * 1000
            
            # Log API request (skip for health checks and static files)
            if not request.url.path.startswith(("/health", "/static", "/favicon.ico")):
                db = next(get_db())
                audit = AuditLogger(db)
                
                # Determine action based on method and path
                action = f"api_{request.method.lower()}"
                resource_type = request.url.path.split("/")[1] if len(request.url.path.split("/")) > 1 else "unknown"
                
                # Determine risk score based on status code
                risk_score = 0
                if response.status_code >= 400:
                    risk_score = 30
                if response.status_code >= 500:
                    risk_score = 70
                
                # Check for sensitive operations
                sensitive_paths = ["/users", "/admin", "/secrets", "/config"]
                if any(sensitive in request.url.path for sensitive in sensitive_paths):
                    risk_score += 20
                
                audit.log(AuditEvent(
                    event_type="api_request",
                    user_id=user_id,
                    username=username,
                    action=action,
                    status="success" if response.status_code < 400 else "failure",
                    resource_type=resource_type,
                    resource_id=request.path_params.get("id"),
                    details={
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                        "processing_time_ms": process_time,
                        "query_params": dict(request.query_params),
                    },
                    risk_score=min(risk_score, 100),
                    tags=["api", request.method.lower()],
                ), request)
            
            return response

# Initialize database tables
def init_oauth_db():
    """Initialize OAuth-related database tables"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    # Use the same database URL as the main application
    engine = create_engine(settings.DATABASE_URL)
    
    # Create tables
    Base.metadata.create_all(engine)
    
    # Create default OAuth client if it doesn't exist
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        # Check if internal client exists
        internal_client = db.query(OAuthClient).filter(
            OAuthClient.client_id == "sovereign-studio"
        ).first()
        
        if not internal_client:
            hasher = Argon2Hasher()
            client_secret = os.getenv("INTERNAL_OAUTH_CLIENT_SECRET", secrets.token_urlsafe(32))
            
            internal_client = OAuthClient(
                client_id="sovereign-studio",
                client_secret_hash=hasher.hash(client_secret),
                name="SOVEREIGN Studio Internal",
                description="Internal OAuth2 client for SOVEREIGN Studio",
                redirect_uris=["http://localhost:8000/oauth/callback", "https://studio.sovereign.ai/oauth/callback"],
                allowed_scopes=["openid", "profile", "email", "admin"],
                allowed_roles=[Role.ADMIN.value, Role.TRAINER.value, Role.VIEWER.value],
                is_confidential=True,
                secret_expires_at=datetime.utcnow() + timedelta(days=90),
            )
            
            db.add(internal_client)
            db.commit()
            
            logger.info(f"Created internal OAuth client with secret: {client_secret}")
            logger.info("Store this secret securely - it won't be shown again!")
    
    except Exception as e:
        logger.error(f"Failed to initialize OAuth database: {e}")
        db.rollback()
    finally:
        db.close()

# Export main components
__all__ = [
    "OAuth2Provider",
    "AuditLogger",
    "PolicyEngine",
    "SecretManager",
    "Argon2Hasher",
    "router",
    "init_oauth_db",
    "get_current_user",
    "require_permissions",
    "require_role",
    "Role",
    "Permission",
    "AuditMiddleware",
]