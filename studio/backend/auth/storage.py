# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
SQLite storage for authentication data (user credentials + JWT secret).
"""

import hashlib
import secrets
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict, Any

from utils.paths import auth_db_path, ensure_dir

DB_PATH = auth_db_path()
DEFAULT_ADMIN_USERNAME = "vex"

# Plaintext bootstrap password file — lives beside auth.db, deleted on
# first password change so the credential never lingers on disk.
_BOOTSTRAP_PW_PATH = DB_PATH.parent / ".bootstrap_password"

# In-process cache so we don't re-read the file on every HTML serve.
_bootstrap_password: Optional[str] = None


def generate_bootstrap_password() -> str:
    """Generate a 4-word diceware passphrase and persist it to disk.

    The passphrase is written to ``_BOOTSTRAP_PW_PATH`` so that it
    survives server restarts (the DB only stores the *hash*).  On
    subsequent calls / restarts, the persisted value is returned.
    """
    global _bootstrap_password

    # 1. Already cached in this process?
    if _bootstrap_password is not None:
        return _bootstrap_password

    # 2. Already persisted from a previous run?
    if _BOOTSTRAP_PW_PATH.is_file():
        _bootstrap_password = _BOOTSTRAP_PW_PATH.read_text().strip()
        if _bootstrap_password:
            return _bootstrap_password

    # 3. First-ever startup — generate a fresh passphrase.
    import diceware

    _bootstrap_password = diceware.get_passphrase(
        options = diceware.handle_options(args = ["-n", "4", "-d", "", "-c"])
    )

    # Persist so the *same* passphrase is used if the server restarts
    # before the user changes the password.
    ensure_dir(_BOOTSTRAP_PW_PATH.parent)
    _BOOTSTRAP_PW_PATH.write_text(_bootstrap_password)

    return _bootstrap_password


def get_bootstrap_password() -> Optional[str]:
    """Return the cached bootstrap password, or None if not yet generated."""
    return _bootstrap_password


def clear_bootstrap_password() -> None:
    """Delete the persisted bootstrap password file (called after password change)."""
    global _bootstrap_password
    _bootstrap_password = None
    if _BOOTSTRAP_PW_PATH.is_file():
        _BOOTSTRAP_PW_PATH.unlink(missing_ok = True)


def _hash_token(token: str) -> str:
    """SHA-256 hash helper used for refresh token storage."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def get_connection() -> sqlite3.Connection:
    """Get a connection to the auth database, creating tables if needed."""
    ensure_dir(DB_PATH.parent)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Create auth_user table with RBAC support
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS auth_user (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'viewer',
            jwt_secret TEXT NOT NULL,
            must_change_password INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            last_login TEXT,
            failed_attempts INTEGER NOT NULL DEFAULT 0,
            locked_until TEXT,
            oauth_provider TEXT,
            oauth_id TEXT,
            mfa_secret TEXT,
            mfa_enabled INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    
    # Create refresh_tokens table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS refresh_tokens (
            id INTEGER PRIMARY KEY,
            token_hash TEXT NOT NULL,
            username TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            user_agent TEXT,
            ip_address TEXT,
            revoked INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    
    # Create audit_logs table for security events
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL,
            username TEXT,
            ip_address TEXT,
            user_agent TEXT,
            details TEXT,
            success INTEGER NOT NULL DEFAULT 1
        );
        """
    )
    
    # Create roles table for RBAC
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS roles (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            permissions TEXT NOT NULL DEFAULT '[]'
        );
        """
    )
    
    # Create user_sessions table for OAuth2/OIDC
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_sessions (
            id INTEGER PRIMARY KEY,
            session_id TEXT UNIQUE NOT NULL,
            username TEXT NOT NULL,
            provider TEXT NOT NULL,
            access_token TEXT,
            refresh_token TEXT,
            id_token TEXT,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_activity TEXT NOT NULL
        );
        """
    )
    
    # Create secret_rotation table for JWT secret history
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS secret_rotation (
            id INTEGER PRIMARY KEY,
            username TEXT NOT NULL,
            old_secret_hash TEXT NOT NULL,
            new_secret_hash TEXT NOT NULL,
            rotated_at TEXT NOT NULL,
            reason TEXT
        );
        """
    )
    
    # Create policy_engine table for fine-grained permissions
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS policies (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            description TEXT,
            effect TEXT NOT NULL CHECK(effect IN ('allow', 'deny')),
            actions TEXT NOT NULL,
            resources TEXT NOT NULL,
            conditions TEXT,
            priority INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        """
    )
    
    # Create indexes for performance
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_username ON audit_logs(username)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_refresh_tokens_username ON refresh_tokens(username)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_refresh_tokens_expires ON refresh_tokens(expires_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user_sessions_username ON user_sessions(username)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user_sessions_provider ON user_sessions(provider)")
    
    # Schema migrations for existing installations
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(auth_user)")}
    
    if "must_change_password" not in columns:
        conn.execute("ALTER TABLE auth_user ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0")
    
    if "role" not in columns:
        conn.execute("ALTER TABLE auth_user ADD COLUMN role TEXT NOT NULL DEFAULT 'viewer'")
    
    if "created_at" not in columns:
        conn.execute("ALTER TABLE auth_user ADD COLUMN created_at TEXT NOT NULL DEFAULT ''")
    
    if "last_login" not in columns:
        conn.execute("ALTER TABLE auth_user ADD COLUMN last_login TEXT")
    
    if "failed_attempts" not in columns:
        conn.execute("ALTER TABLE auth_user ADD COLUMN failed_attempts INTEGER NOT NULL DEFAULT 0")
    
    if "locked_until" not in columns:
        conn.execute("ALTER TABLE auth_user ADD COLUMN locked_until TEXT")
    
    if "oauth_provider" not in columns:
        conn.execute("ALTER TABLE auth_user ADD COLUMN oauth_provider TEXT")
    
    if "oauth_id" not in columns:
        conn.execute("ALTER TABLE auth_user ADD COLUMN oauth_id TEXT")
    
    if "mfa_secret" not in columns:
        conn.execute("ALTER TABLE auth_user ADD COLUMN mfa_secret TEXT")
    
    if "mfa_enabled" not in columns:
        conn.execute("ALTER TABLE auth_user ADD COLUMN mfa_enabled INTEGER NOT NULL DEFAULT 0")
    
    # Migrate from PBKDF2 to Argon2id if needed
    if "password_salt" in columns:
        # We'll handle migration during password verification
        pass
    
    # Initialize default roles if they don't exist
    _initialize_default_roles(conn)
    
    conn.commit()
    return conn


def _initialize_default_roles(conn: sqlite3.Connection) -> None:
    """Initialize default RBAC roles if they don't exist."""
    default_roles = [
        ("admin", "Full system access", '["*"]'),
        ("trainer", "Can manage models and training", '["model:*", "training:*", "dataset:read"]'),
        ("viewer", "Read-only access", '["model:read", "training:read", "dataset:read"]'),
    ]
    
    for role_name, description, permissions in default_roles:
        conn.execute(
            "INSERT OR IGNORE INTO roles (name, description, permissions) VALUES (?, ?, ?)",
            (role_name, description, permissions)
        )


def log_audit_event(
    event_type: str,
    username: Optional[str] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    details: Optional[str] = None,
    success: bool = True
) -> None:
    """Log an authentication/authorization event to the audit trail."""
    timestamp = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO audit_logs (timestamp, event_type, username, ip_address, user_agent, details, success)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, event_type, username, ip_address, user_agent, details, int(success))
        )
        conn.commit()
    finally:
        conn.close()


def get_audit_logs(
    username: Optional[str] = None,
    event_type: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Retrieve audit logs with optional filtering."""
    conn = get_connection()
    try:
        query = "SELECT * FROM audit_logs WHERE 1=1"
        params = []
        
        if username:
            query += " AND username = ?"
            params.append(username)
        
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        cur = conn.execute(query, params)
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def rotate_jwt_secret(username: str, reason: str = "scheduled_rotation") -> str:
    """Rotate JWT secret for a user and store history."""
    from .hashing import hash_secret
    
    new_secret = secrets.token_urlsafe(64)
    new_secret_hash = hash_secret(new_secret)
    
    conn = get_connection()
    try:
        # Get current secret
        cur = conn.execute(
            "SELECT jwt_secret FROM auth_user WHERE username = ?",
            (username,)
        )
        row = cur.fetchone()
        if not row:
            raise ValueError(f"User {username} not found")
        
        old_secret = row["jwt_secret"]
        old_secret_hash = hash_secret(old_secret)
        
        # Update with new secret
        conn.execute(
            "UPDATE auth_user SET jwt_secret = ? WHERE username = ?",
            (new_secret, username)
        )
        
        # Record rotation
        rotated_at = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO secret_rotation (username, old_secret_hash, new_secret_hash, rotated_at, reason)
            VALUES (?, ?, ?, ?, ?)
            """,
            (username, old_secret_hash, new_secret_hash, rotated_at, reason)
        )
        
        conn.commit()
        
        log_audit_event(
            "jwt_secret_rotation",
            username=username,
            details=f"JWT secret rotated. Reason: {reason}"
        )
        
        return new_secret
    finally:
        conn.close()


def check_permission(username: str, action: str, resource: str) -> bool:
    """Check if a user has permission to perform an action on a resource."""
    conn = get_connection()
    try:
        # Get user's role
        cur = conn.execute(
            "SELECT role FROM auth_user WHERE username = ?",
            (username,)
        )
        row = cur.fetchone()
        if not row:
            return False
        
        role_name = row["role"]
        
        # Get role permissions
        cur = conn.execute(
            "SELECT permissions FROM roles WHERE name = ?",
            (role_name,)
        )
        row = cur.fetchone()
        if not row:
            return False
        
        import json
        permissions = json.loads(row["permissions"])
        
        # Check for wildcard permission
        if "*" in permissions:
            return True
        
        # Check specific permission
        permission_pattern = f"{action}:{resource}"
        for perm in permissions:
            if perm == permission_pattern:
                return True
            # Check wildcard patterns like "model:*"
            if perm.endswith(":*") and perm[:-2] == action:
                return True
            if perm.startswith("*:") and perm[2:] == resource:
                return True
        
        # Check policies (more fine-grained)
        cur = conn.execute(
            """
            SELECT effect FROM policies 
            WHERE actions LIKE ? AND resources LIKE ?
            ORDER BY priority DESC
            """,
            (f"%{action}%", f"%{resource}%")
        )
        
        for policy_row in cur.fetchall():
            if policy_row["effect"] == "allow":
                return True
            elif policy_row["effect"] == "deny":
                return False
        
        return False
    finally:
        conn.close()


def update_user_role(username: str, new_role: str) -> bool:
    """Update a user's role."""
    conn = get_connection()
    try:
        # Verify role exists
        cur = conn.execute(
            "SELECT name FROM roles WHERE name = ?",
            (new_role,)
        )
        if not cur.fetchone():
            raise ValueError(f"Role {new_role} does not exist")
        
        # Get current role for audit
        cur = conn.execute(
            "SELECT role FROM auth_user WHERE username = ?",
            (username,)
        )
        row = cur.fetchone()
        if not row:
            return False
        
        old_role = row["role"]
        
        # Update role
        conn.execute(
            "UPDATE auth_user SET role = ? WHERE username = ?",
            (new_role, username)
        )
        conn.commit()
        
        log_audit_event(
            "role_change",
            username=username,
            details=f"Role changed from {old_role} to {new_role}"
        )
        
        return True
    finally:
        conn.close()


def is_initialized() -> bool:
    """Check if auth is ready for login (at least one user exists in DB)."""
    conn = get_connection()
    cur = conn.execute("SELECT COUNT(*) AS c FROM auth_user")
    row = cur.fetchone()
    conn.close()
    return bool(row["c"])


def create_initial_user(
    username: str,
    password: str,
    jwt_secret: str,
    *,
    must_change_password: bool = False,
    role: str = "viewer"
) -> None:
    """
    Create the initial admin user in the database.

    Raises sqlite3.IntegrityError if username already exists.
    """
    from .hashing import hash_password_argon2id
    
    password_hash = hash_password_argon2id(password)
    created_at = datetime.now(timezone.utc).isoformat()
    
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO auth_user (
                username,
                password_hash,
                role,
                jwt_secret,
                must_change_password,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (username, password_hash, role, jwt_secret, int(must_change_password), created_at),
        )
        conn.commit()
        
        log_audit_event(
            "user_created",
            username=username,
            details=f"User created with role: {role}"
        )
    finally:
        conn.close()


def delete_user(username: str) -> None:
    """
    Delete a user from the database.

    Used for rollback when user creation fails partway through bootstrap.
    """
    conn = get_connection()
    try:
        conn.execute("DELETE FROM auth_user WHERE username = ?", (username,))
        conn.execute("DELETE FROM refresh_tokens WHERE username = ?", (username,))
        conn.execute("DELETE FROM user_sessions WHERE username = ?", (username,))
        conn.commit()
        
        log_audit_event(
            "user_deleted",
            username=username,
            details="User account deleted"
        )
    finally:
        conn.close()


def get_user_and_secret(username: str) -> Optional[Tuple[str, str, str, bool, str]]:
    """
    Get user's password hash, JWT secret, must_change_password flag, and role.

    Returns (password_hash, jwt_secret, must_change_password, role)
    or None if user not found.
    """
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            SELECT password_hash, jwt_secret, must_change_password, role
            FROM auth_user
            WHERE username = ?
            """,
            (username,),
        )
        row = cur.fetchone()
        if not row:
            return None
        return (
            row["password_hash"],
            row["jwt_secret"],
            bool(row["must_change_password"]),
            row["role"],
        )
    finally:
        conn.close()


def get_jwt_secret(username: str) -> Optional[str]:
    """Return the current JWT signing secret for a user."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT jwt_secret FROM auth_user WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
        return row["jwt_secret"] if row else None
    finally:
        conn.close()


def requires_password_change(username: str) -> bool:
    """Return whether the user must change the seeded default password."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT must_change_password FROM auth_user WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
        return bool(row and row["must_change_password"])
    finally:
        conn.close()


def load_jwt_secret() -> str:
    """
    Load the JWT secret from the database.

    Raises RuntimeError if no auth user has been created yet.
    """
    conn = get_connection()
    try:
        cur = conn.execute("SELECT jwt_secret FROM auth_user LIMIT 1")
        row = cur.fetchone()
        if not row:
            raise RuntimeError(
                "Auth is not initialized. Wait for the seeded admin bootstrap to complete."
            )
        return row["jwt_secret"]
    finally:
        conn.close()


def ensure_default_admin() -> bool:
    """Seed the default admin account on first startup.

    Uses a randomly generated diceware passphrase as the bootstrap password.
    Returns True when the default admin was created in this call.
    """
    bootstrap_pw = generate_bootstrap_password()
    try:
        create_initial_user(
            username = DEFAULT_ADMIN_USERNAME,
            password = bootstrap_pw,
            jwt_secret = secrets.token_urlsafe(64),
            must_change_password = True,
            role = "admin"
        )
        return True
    except sqlite3.IntegrityError:
        return False


def update_password(username: str, new_password: str) -> bool:
    """Update password, clear first-login requirement, rotate JWT secret."""
    from .hashing import hash_password_argon2id
    
    password_hash = hash_password_argon2id(new_password)
    new_jwt_secret = secrets.token_urlsafe(64)
    
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            UPDATE auth_user 
            SET password_hash = ?,
                jwt_secret = ?,
                must_change_password = 0
            WHERE username = ?
            """,
            (password_hash, new_jwt_secret, username),
        )
        conn.commit()
        
        if cur.rowcount > 0:
            log_audit_event(
                "password_change",
                username=username,
                details="Password changed successfully"
            )
            clear_bootstrap_password()
            return True
        return False
    finally:
        conn.close()


def verify_password_with_migration(username: str, password: str) -> bool:
    """Verify password and migrate from PBKDF2 to Argon2id if needed."""
    from .hashing import verify_password_argon2id, hash_password_argon2id
    
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT password_hash FROM auth_user WHERE username = ?",
            (username,)
        )
        row = cur.fetchone()
        if not row:
            return False
        
        stored_hash = row["password_hash"]
        
        # Check if it's an old PBKDF2 hash (64 chars hex)
        if len(stored_hash) == 64 and all(c in '0123456789abcdef' for c in stored_hash):
            # This is a PBKDF2 hash, need to migrate
            from .hashing import verify_password_pbkdf2
            if verify_password_pbkdf2(stored_hash, password):
                # Migrate to Argon2id
                new_hash = hash_password_argon2id(password)
                conn.execute(
                    "UPDATE auth_user SET password_hash = ? WHERE username = ?",
                    (new_hash, username)
                )
                conn.commit()
                log_audit_event(
                    "password_migration",
                    username=username,
                    details="Migrated from PBKDF2 to Argon2id"
                )
                return True
            return False
        else:
            # Argon2id hash
            return verify_password_argon2id(stored_hash, password)
    finally:
        conn.close()


def update_last_login(username: str) -> None:
    """Update the last login timestamp for a user."""
    last_login = datetime.now(timezone.utc).isoformat()
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE auth_user SET last_login = ? WHERE username = ?",
            (last_login, username)
        )
        conn.commit()
    finally:
        conn.close()


def lock_account(username: str, duration_minutes: int = 30) -> None:
    """Lock a user account for a specified duration."""
    from datetime import timedelta
    locked_until = (datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)).isoformat()
    
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE auth_user SET locked_until = ? WHERE username = ?",
            (locked_until, username)
        )
        conn.commit()
        
        log_audit_event(
            "account_locked",
            username=username,
            details=f"Account locked for {duration_minutes} minutes"
        )
    finally:
        conn.close()


def is_account_locked(username: str) -> bool:
    """Check if a user account is currently locked."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "SELECT locked_until FROM auth_user WHERE username = ?",
            (username,)
        )
        row = cur.fetchone()
        if not row or not row["locked_until"]:
            return False
        
        locked_until = datetime.fromisoformat(row["locked_until"])
        return datetime.now(timezone.utc) < locked_until
    finally:
        conn.close()


def increment_failed_attempts(username: str) -> int:
    """Increment failed login attempts and lock if threshold exceeded."""
    conn = get_connection()
    try:
        # Get current attempts
        cur = conn.execute(
            "SELECT failed_attempts FROM auth_user WHERE username = ?",
            (username,)
        )
        row = cur.fetchone()
        if not row:
            return 0
        
        attempts = row["failed_attempts"] + 1
        
        # Update attempts
        conn.execute(
            "UPDATE auth_user SET failed_attempts = ? WHERE username = ?",
            (attempts, username)
        )
        conn.commit()
        
        # Lock account after 5 failed attempts
        if attempts >= 5:
            lock_account(username, 30)
            log_audit_event(
                "account_locked_brute_force",
                username=username,
                details=f"Account locked after {attempts} failed attempts"
            )
        
        return attempts
    finally:
        conn.close()


def reset_failed_attempts(username: str) -> None:
    """Reset failed login attempts after successful login."""
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE auth_user SET failed_attempts = 0 WHERE username = ?",
            (username,)
        )
        conn.commit()
    finally:
        conn.close()


def create_user_session(
    username: str,
    provider: str,
    access_token: str,
    refresh_token: Optional[str] = None,
    id_token: Optional[str] = None,
    expires_in: int = 3600
) -> str:
    """Create a new OAuth2/OIDC user session."""
    import uuid
    from datetime import timedelta
    
    session_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    expires_at = (datetime.now(timezone.utc) + timedelta(seconds=expires_in)).isoformat()
    
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO user_sessions (
                session_id, username, provider, access_token, 
                refresh_token, id_token, expires_at, created_at, last_activity
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, username, provider, access_token, 
             refresh_token, id_token, expires_at, created_at, created_at)
        )
        conn.commit()
        
        log_audit_event(
            "session_created",
            username=username,
            details=f"New session created with provider: {provider}"
        )
        
        return session_id
    finally:
        conn.close()


def get_user_sessions(username: str) -> List[Dict[str, Any]]:
    """Get all active sessions for a user."""
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            SELECT session_id, provider, expires_at, created_at, last_activity
            FROM user_sessions 
            WHERE username = ? AND expires_at > ?
            ORDER BY last_activity DESC
            """,
            (username, datetime.now(timezone.utc).isoformat())
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def revoke_session(session_id: str) -> bool:
    """Revoke a user session."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "DELETE FROM user_sessions WHERE session_id = ?",
            (session_id,)
        )
        conn.commit()
        
        if cur.rowcount > 0:
            log_audit_event(
                "session_revoked",
                details=f"Session {session_id} revoked"
            )
            return True
        return False
    finally:
        conn.close()


def cleanup_expired_sessions() -> int:
    """Clean up expired sessions. Returns number of sessions cleaned."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "DELETE FROM user_sessions WHERE expires_at <= ?",
            (datetime.now(timezone.utc).isoformat(),)
        )
        conn.commit()
        
        if cur.rowcount > 0:
            log_audit_event(
                "session_cleanup",
                details=f"Cleaned up {cur.rowcount} expired sessions"
            )
        
        return cur.rowcount
    finally:
        conn.close()


def get_user_permissions(username: str) -> List[str]:
    """Get all permissions for a user based on their role."""
    import json
    
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            SELECT r.permissions 
            FROM auth_user u
            JOIN roles r ON u.role = r.name
            WHERE u.username = ?
            """,
            (username,)
        )
        row = cur.fetchone()
        if not row:
            return []
        
        return json.loads(row["permissions"])
    finally:
        conn.close()


def create_policy(
    name: str,
    effect: str,
    actions: List[str],
    resources: List[str],
    conditions: Optional[Dict[str, Any]] = None,
    priority: int = 0,
    description: Optional[str] = None
) -> None:
    """Create a new policy for fine-grained access control."""
    import json
    
    created_at = datetime.now(timezone.utc).isoformat()
    actions_str = json.dumps(actions)
    resources_str = json.dumps(resources)
    conditions_str = json.dumps(conditions) if conditions else None
    
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO policies (name, description, effect, actions, resources, conditions, priority, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (name, description, effect, actions_str, resources_str, conditions_str, priority, created_at)
        )
        conn.commit()
        
        log_audit_event(
            "policy_created",
            details=f"Policy '{name}' created with effect: {effect}"
        )
    finally:
        conn.close()


def evaluate_policies(username: str, action: str, resource: str, context: Optional[Dict[str, Any]] = None) -> bool:
    """Evaluate all policies for a user action on a resource."""
    import json
    
    conn = get_connection()
    try:
        # Get user's role
        cur = conn.execute(
            "SELECT role FROM auth_user WHERE username = ?",
            (username,)
        )
        row = cur.fetchone()
        if not row:
            return False
        
        role = row["role"]
        
        # Get all applicable policies ordered by priority
        cur = conn.execute(
            """
            SELECT * FROM policies 
            ORDER BY priority DESC
            """
        )
        
        for policy in cur.fetchall():
            policy_actions = json.loads(policy["actions"])
            policy_resources = json.loads(policy["resources"])
            
            # Check if policy applies
            action_match = "*" in policy_actions or action in policy_actions
            resource_match = "*" in policy_resources or resource in policy_resources
            
            if action_match and resource_match:
                # Check conditions if any
                if policy["conditions"]:
                    conditions = json.loads(policy["conditions"])
                    if not _evaluate_conditions(conditions, context or {}):
                        continue
                
                # Apply policy effect
                if policy["effect"] == "allow":
                    return True
                elif policy["effect"] == "deny":
                    return False
        
        # Default deny if no policy matches
        return False
    finally:
        conn.close()


def _evaluate_conditions(conditions: Dict[str, Any], context: Dict[str, Any]) -> bool:
    """Evaluate policy conditions against context."""
    for key, expected_value in conditions.items():
        if key not in context:
            return False
        if context[key] != expected_value:
            return False
    return True


def get_security_report(start_time: Optional[str] = None, end_time: Optional[str] = None) -> Dict[str, Any]:
    """Generate a security report with authentication statistics."""
    conn = get_connection()
    try:
        report = {
            "total_users": 0,
            "active_users": 0,
            "locked_accounts": 0,
            "failed_logins": 0,
            "successful_logins": 0,
            "role_distribution": {},
            "oauth_providers": {},
            "recent_events": []
        }
        
        # Total users
        cur = conn.execute("SELECT COUNT(*) as count FROM auth_user")
        report["total_users"] = cur.fetchone()["count"]
        
        # Active users (logged in last 30 days)
        thirty_days_ago = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        cur = conn.execute(
            "SELECT COUNT(*) as count FROM auth_user WHERE last_login >= ?",
            (thirty_days_ago,)
        )
        report["active_users"] = cur.fetchone()["count"]
        
        # Locked accounts
        cur = conn.execute(
            "SELECT COUNT(*) as count FROM auth_user WHERE locked_until > ?",
            (datetime.now(timezone.utc).isoformat(),)
        )
        report["locked_accounts"] = cur.fetchone()["count"]
        
        # Login statistics
        query = "SELECT event_type, success, COUNT(*) as count FROM audit_logs"
        params = []
        
        if start_time or end_time:
            query += " WHERE"
            conditions = []
            
            if start_time:
                conditions.append(" timestamp >= ?")
                params.append(start_time)
            
            if end_time:
                conditions.append(" timestamp <= ?")
                params.append(end_time)
            
            query += " AND".join(conditions)
        
        query += " GROUP BY event_type, success"
        
        cur = conn.execute(query, params)
        for row in cur.fetchall():
            if row["event_type"] == "login":
                if row["success"]:
                    report["successful_logins"] = row["count"]
                else:
                    report["failed_logins"] = row["count"]
        
        # Role distribution
        cur = conn.execute("SELECT role, COUNT(*) as count FROM auth_user GROUP BY role")
        for row in cur.fetchall():
            report["role_distribution"][row["role"]] = row["count"]
        
        # OAuth providers
        cur = conn.execute(
            "SELECT oauth_provider, COUNT(*) as count FROM auth_user WHERE oauth_provider IS NOT NULL GROUP BY oauth_provider"
        )
        for row in cur.fetchall():
            report["oauth_providers"][row["oauth_provider"]] = row["count"]
        
        # Recent security events
        cur = conn.execute(
            "SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT 10"
        )
        report["recent_events"] = [dict(row) for row in cur.fetchall()]
        
        return report
    finally:
        conn.close()