# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Password hashing utilities using Argon2id.
"""

import secrets
from typing import Tuple
from argon2 import PasswordHasher, Type
from argon2.exceptions import VerifyMismatchError, InvalidHashError

# Argon2id hasher with secure defaults
_ph = PasswordHasher(
    time_cost=3,          # Number of iterations
    memory_cost=65536,    # 64MB memory usage
    parallelism=4,        # Number of parallel threads
    hash_len=32,          # Length of the hash in bytes
    salt_len=16,          # Length of random salt in bytes
    type=Type.ID          # Argon2id variant (hybrid)
)


def hash_password(password: str, salt: str | None = None) -> Tuple[str, str]:
    """
    Hash a password using Argon2id.
    
    Returns (salt, encoded_hash) tuple.
    Note: Argon2id encodes salt and parameters in the hash string,
    but we return them separately for compatibility with existing systems.
    """
    # Generate a random salt if not provided
    if salt is None:
        salt = secrets.token_hex(16)
    
    # Argon2id expects salt as bytes
    salt_bytes = bytes.fromhex(salt) if len(salt) == 32 else salt.encode('utf-8')
    
    # Hash the password with Argon2id
    # Note: We use the low-level hash method to provide custom salt
    from argon2.low_level import hash_secret
    raw_hash = hash_secret(
        secret=password.encode('utf-8'),
        salt=salt_bytes,
        time_cost=3,
        memory_cost=65536,
        parallelism=4,
        hash_len=32,
        type=Type.ID
    )
    
    # Return salt and the encoded hash (which includes parameters)
    return salt, raw_hash.decode('utf-8')


def verify_password(password: str, salt: str, hashed: str) -> bool:
    """
    Verify a password against a stored salt and hash.
    
    Uses constant-time comparison via Argon2's verification.
    """
    try:
        # Argon2's verify method handles constant-time comparison
        return _ph.verify(hashed, password)
    except (VerifyMismatchError, InvalidHashError):
        return False
    except Exception:
        # Catch any other unexpected errors
        return False


def needs_rehash(hashed: str) -> bool:
    """
    Check if a hash needs to be rehashed due to updated parameters.
    
    Useful for secret rotation - verify if stored hash uses current parameters.
    """
    try:
        return _ph.check_needs_rehash(hashed)
    except InvalidHashError:
        # Invalid hash format, should be rehashed
        return True


def get_hash_parameters(hashed: str) -> dict:
    """
    Extract parameters from an Argon2id hash string.
    
    Returns dict with time_cost, memory_cost, parallelism, etc.
    Useful for audit logging and debugging.
    """
    try:
        from argon2.low_level import decode_raw
        decoded = decode_raw(hashed)
        return {
            'algorithm': 'argon2id',
            'version': decoded.version,
            'time_cost': decoded.time_cost,
            'memory_cost': decoded.memory_cost,
            'parallelism': decoded.parallelism,
            'hash_length': len(decoded.hash),
            'salt_length': len(decoded.salt)
        }
    except Exception:
        return {'algorithm': 'unknown', 'error': 'Failed to decode hash'}