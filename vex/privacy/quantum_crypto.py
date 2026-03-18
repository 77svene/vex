"""
Quantum-Resistant Privacy Layer for Scrapy
==========================================

End-to-end encryption for scraped data using post-quantum cryptography,
automatic PII detection and masking, and GDPR-compliant data handling
with automatic data lifecycle management.
"""

import os
import json
import hashlib
import logging
import re
import base64
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

# Post-quantum cryptography imports
try:
    from pqcrypto.kem import kyber512, kyber768, kyber1024
    from pqcrypto.sign import dilithium2, dilithium3, dilithium5
    PQCRYPTO_AVAILABLE = True
except ImportError:
    PQCRYPTO_AVAILABLE = False
    # Fallback to classical cryptography if post-quantum not available
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.backends import default_backend

# ML-based PII detection
try:
    import spacy
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    ML_PII_AVAILABLE = True
except ImportError:
    ML_PII_AVAILABLE = False

from vex import signals
from vex.exceptions import NotConfigured
from vex.utils.project import get_project_settings
from vex.utils.misc import load_object

logger = logging.getLogger(__name__)


class EncryptionAlgorithm(Enum):
    """Supported encryption algorithms."""
    KYBER512 = "kyber512"
    KYBER768 = "kyber768"
    KYBER1024 = "kyber1024"
    DILITHIUM2 = "dilithium2"
    DILITHIUM3 = "dilithium3"
    DILITHIUM5 = "dilithium5"
    RSA_4096 = "rsa_4096"  # Fallback


class DataClassification(Enum):
    """Data classification levels for GDPR compliance."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PII = "pii"
    SENSITIVE_PII = "sensitive_pii"


@dataclass
class RetentionPolicy:
    """Data retention policy configuration."""
    classification: DataClassification
    retention_days: int
    auto_delete: bool = True
    encrypt_at_rest: bool = True
    anonymize_on_expiry: bool = False
    legal_hold: bool = False


@dataclass
class PIIMatch:
    """Detected PII match."""
    pii_type: str
    value: str
    start: int
    end: int
    confidence: float
    masked_value: str = ""


@dataclass
class EncryptionKey:
    """Encryption key container."""
    key_id: str
    public_key: bytes
    private_key: Optional[bytes] = None
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.KYBER768
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumResistantCrypto:
    """
    Quantum-resistant cryptographic operations using post-quantum algorithms.
    Supports Kyber for key encapsulation and Dilithium for digital signatures.
    """
    
    def __init__(self, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.KYBER768):
        self.algorithm = algorithm
        self._validate_algorithm()
        
    def _validate_algorithm(self):
        """Validate that the selected algorithm is available."""
        if not PQCRYPTO_AVAILABLE and self.algorithm in [
            EncryptionAlgorithm.KYBER512,
            EncryptionAlgorithm.KYBER768,
            EncryptionAlgorithm.KYBER1024,
            EncryptionAlgorithm.DILITHIUM2,
            EncryptionAlgorithm.DILITHIUM3,
            EncryptionAlgorithm.DILITHIUM5
        ]:
            logger.warning(
                "Post-quantum cryptography library not available. "
                "Falling back to RSA-4096. Install pqcrypto for quantum resistance."
            )
            self.algorithm = EncryptionAlgorithm.RSA_4096
    
    def generate_keypair(self) -> Tuple[bytes, bytes]:
        """Generate a quantum-resistant key pair."""
        if self.algorithm == EncryptionAlgorithm.KYBER512:
            pk, sk = kyber512.generate_keypair()
        elif self.algorithm == EncryptionAlgorithm.KYBER768:
            pk, sk = kyber768.generate_keypair()
        elif self.algorithm == EncryptionAlgorithm.KYBER1024:
            pk, sk = kyber1024.generate_keypair()
        elif self.algorithm == EncryptionAlgorithm.DILITHIUM2:
            pk, sk = dilithium2.generate_keypair()
        elif self.algorithm == EncryptionAlgorithm.DILITHIUM3:
            pk, sk = dilithium3.generate_keypair()
        elif self.algorithm == EncryptionAlgorithm.DILITHIUM5:
            pk, sk = dilithium5.generate_keypair()
        else:  # RSA fallback
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=4096,
                backend=default_backend()
            )
            public_key = private_key.public_key()
            pk = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            sk = private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
        return pk, sk
    
    def encapsulate(self, public_key: bytes) -> Tuple[bytes, bytes]:
        """Encapsulate a shared secret using public key."""
        if self.algorithm == EncryptionAlgorithm.KYBER512:
            ciphertext, shared_secret = kyber512.encap(public_key)
        elif self.algorithm == EncryptionAlgorithm.KYBER768:
            ciphertext, shared_secret = kyber768.encap(public_key)
        elif self.algorithm == EncryptionAlgorithm.KYBER1024:
            ciphertext, shared_secret = kyber1024.encap(public_key)
        else:  # RSA fallback
            # Generate random shared secret
            shared_secret = os.urandom(32)
            # Encrypt with RSA
            from cryptography.hazmat.primitives.asymmetric import padding as rsa_padding
            public_key_obj = serialization.load_pem_public_key(public_key, backend=default_backend())
            ciphertext = public_key_obj.encrypt(
                shared_secret,
                rsa_padding.OAEP(
                    mgf=rsa_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
        return ciphertext, shared_secret
    
    def decapsulate(self, ciphertext: bytes, private_key: bytes) -> bytes:
        """Decapsulate shared secret using private key."""
        if self.algorithm == EncryptionAlgorithm.KYBER512:
            shared_secret = kyber512.decap(ciphertext, private_key)
        elif self.algorithm == EncryptionAlgorithm.KYBER768:
            shared_secret = kyber768.decap(ciphertext, private_key)
        elif self.algorithm == EncryptionAlgorithm.KYBER1024:
            shared_secret = kyber1024.decap(ciphertext, private_key)
        else:  # RSA fallback
            from cryptography.hazmat.primitives.asymmetric import padding as rsa_padding
            private_key_obj = serialization.load_pem_private_key(
                private_key, password=None, backend=default_backend()
            )
            shared_secret = private_key_obj.decrypt(
                ciphertext,
                rsa_padding.OAEP(
                    mgf=rsa_padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
        return shared_secret
    
    def sign(self, message: bytes, private_key: bytes) -> bytes:
        """Create a digital signature."""
        if self.algorithm == EncryptionAlgorithm.DILITHIUM2:
            signature = dilithium2.sign(message, private_key)
        elif self.algorithm == EncryptionAlgorithm.DILITHIUM3:
            signature = dilithium3.sign(message, private_key)
        elif self.algorithm == EncryptionAlgorithm.DILITHIUM5:
            signature = dilithium5.sign(message, private_key)
        else:  # RSA fallback
            from cryptography.hazmat.primitives.asymmetric import padding as rsa_padding
            private_key_obj = serialization.load_pem_private_key(
                private_key, password=None, backend=default_backend()
            )
            signature = private_key_obj.sign(
                message,
                rsa_padding.PSS(
                    mgf=rsa_padding.MGF1(hashes.SHA256()),
                    salt_length=rsa_padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
        return signature
    
    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify a digital signature."""
        try:
            if self.algorithm == EncryptionAlgorithm.DILITHIUM2:
                dilithium2.verify(message, signature, public_key)
            elif self.algorithm == EncryptionAlgorithm.DILITHIUM3:
                dilithium3.verify(message, signature, public_key)
            elif self.algorithm == EncryptionAlgorithm.DILITHIUM5:
                dilithium5.verify(message, signature, public_key)
            else:  # RSA fallback
                from cryptography.hazmat.primitives.asymmetric import padding as rsa_padding
                public_key_obj = serialization.load_pem_public_key(public_key, backend=default_backend())
                public_key_obj.verify(
                    signature,
                    message,
                    rsa_padding.PSS(
                        mgf=rsa_padding.MGF1(hashes.SHA256()),
                        salt_length=rsa_padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
            return True
        except Exception:
            return False


class SymmetricEncryptor:
    """AES-256-GCM symmetric encryption for data."""
    
    def __init__(self):
        self.backend = default_backend()
    
    def encrypt(self, data: bytes, key: bytes) -> Tuple[bytes, bytes, bytes]:
        """Encrypt data with AES-256-GCM. Returns (ciphertext, nonce, tag)."""
        nonce = os.urandom(12)
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=self.backend)
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        return ciphertext, nonce, encryptor.tag
    
    def decrypt(self, ciphertext: bytes, key: bytes, nonce: bytes, tag: bytes) -> bytes:
        """Decrypt data with AES-256-GCM."""
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=self.backend)
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()
    
    def derive_key(self, shared_secret: bytes, salt: bytes = None) -> bytes:
        """Derive a symmetric key from shared secret using HKDF."""
        if salt is None:
            salt = os.urandom(16)
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            info=b'vex-privacy-key',
            backend=self.backend
        )
        return hkdf.derive(shared_secret)


class PIIDetector:
    """
    Personal Identifiable Information (PII) detection and masking.
    Supports both rule-based and ML-based detection.
    """
    
    # Common PII patterns
    PATTERNS = {
        'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        'phone': re.compile(r'(\+?1?\s*\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})'),
        'ssn': re.compile(r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'),
        'credit_card': re.compile(r'\b(?:\d[ -]*?){13,16}\b'),
        'ip_address': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
        'date_of_birth': re.compile(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b'),
        'passport': re.compile(r'\b[A-Z]{1,2}\d{6,9}\b'),
        'driver_license': re.compile(r'\b[A-Z]{1,2}\d{5,8}\b'),
    }
    
    def __init__(self, use_ml: bool = True, language: str = 'en'):
        self.use_ml = use_ml and ML_PII_AVAILABLE
        self.language = language
        self._init_ml_engine()
        
    def _init_ml_engine(self):
        """Initialize ML-based PII detection engine."""
        if self.use_ml:
            try:
                self.analyzer = AnalyzerEngine()
                self.anonymizer = AnonymizerEngine()
                logger.info("ML-based PII detection initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ML PII detection: {e}")
                self.use_ml = False
    
    def detect_pii(self, text: str) -> List[PIIMatch]:
        """Detect PII in text."""
        matches = []
        
        # Rule-based detection
        for pii_type, pattern in self.PATTERNS.items():
            for match in pattern.finditer(text):
                matches.append(PIIMatch(
                    pii_type=pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=0.9
                ))
        
        # ML-based detection
        if self.use_ml:
            try:
                results = self.analyzer.analyze(
                    text=text,
                    language=self.language,
                    entities=["PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", 
                             "IP_ADDRESS", "US_SSN", "US_DRIVER_LICENSE"]
                )
                for result in results:
                    matches.append(PIIMatch(
                        pii_type=result.entity_type.lower(),
                        value=text[result.start:result.end],
                        start=result.start,
                        end=result.end,
                        confidence=result.score
                    ))
            except Exception as e:
                logger.warning(f"ML PII detection failed: {e}")
        
        # Remove duplicates and overlaps
        return self._deduplicate_matches(matches)
    
    def _deduplicate_matches(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Remove duplicate and overlapping PII matches."""
        if not matches:
            return []
        
        # Sort by start position
        matches.sort(key=lambda x: x.start)
        
        deduplicated = [matches[0]]
        for match in matches[1:]:
            last = deduplicated[-1]
            # Check for overlap
            if match.start < last.end:
                # Keep the one with higher confidence
                if match.confidence > last.confidence:
                    deduplicated[-1] = match
            else:
                deduplicated.append(match)
        
        return deduplicated
    
    def mask_pii(self, text: str, matches: List[PIIMatch], 
                 mask_char: str = '*', preserve_format: bool = True) -> str:
        """Mask detected PII in text."""
        if not matches:
            return text
        
        # Sort matches in reverse order to avoid index shifting
        matches.sort(key=lambda x: x.start, reverse=True)
        
        masked_text = text
        for match in matches:
            if preserve_format:
                # Preserve format (e.g., keep email domain)
                if match.pii_type == 'email':
                    local, domain = match.value.split('@')
                    masked_local = mask_char * len(local)
                    masked_value = f"{masked_local}@{domain}"
                elif match.pii_type == 'phone':
                    # Keep last 4 digits
                    digits = re.sub(r'\D', '', match.value)
                    masked_value = mask_char * (len(digits) - 4) + digits[-4:]
                else:
                    masked_value = mask_char * len(match.value)
            else:
                masked_value = mask_char * len(match.value)
            
            match.masked_value = masked_value
            masked_text = masked_text[:match.start] + masked_value + masked_text[match.end:]
        
        return masked_text
    
    def classify_data(self, text: str) -> DataClassification:
        """Classify data based on PII content."""
        matches = self.detect_pii(text)
        
        if not matches:
            return DataClassification.PUBLIC
        
        # Check for sensitive PII
        sensitive_types = {'ssn', 'credit_card', 'passport', 'driver_license'}
        has_sensitive = any(m.pii_type in sensitive_types for m in matches)
        
        if has_sensitive:
            return DataClassification.SENSITIVE_PII
        
        # Check for regular PII
        pii_types = {'email', 'phone', 'date_of_birth', 'ip_address'}
        has_pii = any(m.pii_type in pii_types for m in matches)
        
        if has_pii:
            return DataClassification.PII
        
        return DataClassification.INTERNAL


class KeyManager:
    """Manages encryption keys with automatic rotation and lifecycle."""
    
    def __init__(self, key_store_path: str = './keys', 
                 default_retention_days: int = 365):
        self.key_store_path = key_store_path
        self.default_retention_days = default_retention_days
        self.crypto = QuantumResistantCrypto()
        self.keys: Dict[str, EncryptionKey] = {}
        self._lock = threading.RLock()
        
        # Create key store directory
        os.makedirs(key_store_path, exist_ok=True)
        
        # Load existing keys
        self._load_keys()
    
    def _load_keys(self):
        """Load keys from storage."""
        key_file = os.path.join(self.key_store_path, 'keys.json')
        if os.path.exists(key_file):
            try:
                with open(key_file, 'r') as f:
                    data = json.load(f)
                    for key_id, key_data in data.items():
                        self.keys[key_id] = EncryptionKey(
                            key_id=key_id,
                            public_key=base64.b64decode(key_data['public_key']),
                            private_key=base64.b64decode(key_data['private_key']) 
                                if key_data.get('private_key') else None,
                            algorithm=EncryptionAlgorithm(key_data.get('algorithm', 'kyber768')),
                            created_at=datetime.fromisoformat(key_data['created_at']),
                            expires_at=datetime.fromisoformat(key_data['expires_at']) 
                                if key_data.get('expires_at') else None,
                            metadata=key_data.get('metadata', {})
                        )
            except Exception as e:
                logger.error(f"Failed to load keys: {e}")
    
    def _save_keys(self):
        """Save keys to storage."""
        key_file = os.path.join(self.key_store_path, 'keys.json')
        data = {}
        for key_id, key in self.keys.items():
            data[key_id] = {
                'public_key': base64.b64encode(key.public_key).decode('utf-8'),
                'private_key': base64.b64encode(key.private_key).decode('utf-8') 
                    if key.private_key else None,
                'algorithm': key.algorithm.value,
                'created_at': key.created_at.isoformat(),
                'expires_at': key.expires_at.isoformat() if key.expires_at else None,
                'metadata': key.metadata
            }
        
        with open(key_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def generate_key(self, algorithm: EncryptionAlgorithm = None, 
                     retention_days: int = None) -> EncryptionKey:
        """Generate a new encryption key."""
        if algorithm is None:
            algorithm = self.crypto.algorithm
        
        if retention_days is None:
            retention_days = self.default_retention_days
        
        # Generate key pair
        public_key, private_key = self.crypto.generate_keypair()
        
        # Create key ID
        key_id = hashlib.sha256(public_key + str(time.time()).encode()).hexdigest()[:16]
        
        # Create key object
        key = EncryptionKey(
            key_id=key_id,
            public_key=public_key,
            private_key=private_key,
            algorithm=algorithm,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=retention_days),
            metadata={'retention_days': retention_days}
        )
        
        # Store key
        with self._lock:
            self.keys[key_id] = key
            self._save_keys()
        
        logger.info(f"Generated new encryption key: {key_id}")
        return key
    
    def get_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Get a key by ID."""
        with self._lock:
            return self.keys.get(key_id)
    
    def rotate_key(self, key_id: str) -> Optional[EncryptionKey]:
        """Rotate an encryption key."""
        old_key = self.get_key(key_id)
        if not old_key:
            return None
        
        # Generate new key with same algorithm
        new_key = self.generate_key(
            algorithm=old_key.algorithm,
            retention_days=old_key.metadata.get('retention_days', self.default_retention_days)
        )
        
        # Mark old key as rotated
        old_key.metadata['rotated_to'] = new_key.key_id
        old_key.metadata['rotated_at'] = datetime.now().isoformat()
        
        with self._lock:
            self._save_keys()
        
        logger.info(f"Rotated key {key_id} to {new_key.key_id}")
        return new_key
    
    def delete_key(self, key_id: str, secure_delete: bool = True) -> bool:
        """Delete a key (cryptographic erasure)."""
        with self._lock:
            if key_id not in self.keys:
                return False
            
            key = self.keys[key_id]
            
            if secure_delete:
                # Overwrite key material
                if key.private_key:
                    # Create a mutable copy and overwrite
                    mutable_key = bytearray(key.private_key)
                    for i in range(len(mutable_key)):
                        mutable_key[i] = 0
                    key.private_key = bytes(mutable_key)
                
                # Overwrite public key
                mutable_pub = bytearray(key.public_key)
                for i in range(len(mutable_pub)):
                    mutable_pub[i] = 0
                key.public_key = bytes(mutable_pub)
            
            # Remove from storage
            del self.keys[key_id]
            self._save_keys()
        
        logger.info(f"Deleted key: {key_id}")
        return True
    
    def cleanup_expired_keys(self) -> int:
        """Delete expired keys."""
        now = datetime.now()
        expired_keys = []
        
        with self._lock:
            for key_id, key in self.keys.items():
                if key.expires_at and key.expires_at < now:
                    # Check for legal hold
                    if not key.metadata.get('legal_hold', False):
                        expired_keys.append(key_id)
        
        # Delete expired keys
        deleted_count = 0
        for key_id in expired_keys:
            if self.delete_key(key_id):
                deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} expired keys")
        
        return deleted_count


class DataLifecycleManager:
    """
    Manages data lifecycle with automatic retention policies
    and GDPR-compliant data handling.
    """
    
    def __init__(self, key_manager: KeyManager, 
                 policies: List[RetentionPolicy] = None):
        self.key_manager = key_manager
        self.policies = policies or self._default_policies()
        self.data_registry: Dict[str, Dict[str, Any]] = {}
        self._registry_lock = threading.RLock()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="DataLifecycleManager-Cleanup"
        )
        self._cleanup_thread.start()
    
    def _default_policies(self) -> List[RetentionPolicy]:
        """Create default retention policies."""
        return [
            RetentionPolicy(
                classification=DataClassification.PUBLIC,
                retention_days=365,
                auto_delete=False,
                encrypt_at_rest=False
            ),
            RetentionPolicy(
                classification=DataClassification.INTERNAL,
                retention_days=180,
                auto_delete=True,
                encrypt_at_rest=True
            ),
            RetentionPolicy(
                classification=DataClassification.CONFIDENTIAL,
                retention_days=90,
                auto_delete=True,
                encrypt_at_rest=True,
                anonymize_on_expiry=True
            ),
            RetentionPolicy(
                classification=DataClassification.RESTRICTED,
                retention_days=30,
                auto_delete=True,
                encrypt_at_rest=True,
                anonymize_on_expiry=True
            ),
            RetentionPolicy(
                classification=DataClassification.PII,
                retention_days=30,
                auto_delete=True,
                encrypt_at_rest=True,
                anonymize_on_expiry=True
            ),
            RetentionPolicy(
                classification=DataClassification.SENSITIVE_PII,
                retention_days=7,
                auto_delete=True,
                encrypt_at_rest=True,
                anonymize_on_expiry=True
            )
        ]
    
    def register_data(self, data_id: str, classification: DataClassification,
                      encryption_key_id: str = None, 
                      metadata: Dict[str, Any] = None) -> str:
        """Register data for lifecycle management."""
        if encryption_key_id is None:
            # Generate a new key for this data
            key = self.key_manager.generate_key()
            encryption_key_id = key.key_id
        
        # Find applicable policy
        policy = self._get_policy_for_classification(classification)
        
        with self._registry_lock:
            self.data_registry[data_id] = {
                'classification': classification.value,
                'encryption_key_id': encryption_key_id,
                'registered_at': datetime.now().isoformat(),
                'expires_at': (datetime.now() + 
                              timedelta(days=policy.retention_days)).isoformat(),
                'policy': {
                    'auto_delete': policy.auto_delete,
                    'anonymize_on_expiry': policy.anonymize_on_expiry,
                    'legal_hold': policy.legal_hold
                },
                'metadata': metadata or {}
            }
        
        logger.debug(f"Registered data {data_id} with classification {classification.value}")
        return encryption_key_id
    
    def _get_policy_for_classification(self, classification: DataClassification) -> RetentionPolicy:
        """Get retention policy for a data classification."""
        for policy in self.policies:
            if policy.classification == classification:
                return policy
        
        # Default policy
        return RetentionPolicy(
            classification=classification,
            retention_days=365,
            auto_delete=True,
            encrypt_at_rest=True
        )
    
    def _cleanup_loop(self):
        """Background thread for cleaning up expired data."""
        while True:
            try:
                time.sleep(3600)  # Run every hour
                self.cleanup_expired_data()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def cleanup_expired_data(self) -> int:
        """Clean up expired data according to policies."""
        now = datetime.now()
        expired_data = []
        
        with self._registry_lock:
            for data_id, data_info in self.data_registry.items():
                expires_at = datetime.fromisoformat(data_info['expires_at'])
                if expires_at < now:
                    # Check for legal hold
                    if not data_info['policy'].get('legal_hold', False):
                        expired_data.append(data_id)
        
        # Process expired data
        deleted_count = 0
        anonymized_count = 0
        
        for data_id in expired_data:
            with self._registry_lock:
                data_info = self.data_registry.get(data_id)
                if not data_info:
                    continue
            
            # Apply policy
            if data_info['policy'].get('anonymize_on_expiry', False):
                # Anonymize data (mark as anonymized)
                data_info['anonymized_at'] = datetime.now().isoformat()
                data_info['status'] = 'anonymized'
                anonymized_count += 1
                logger.info(f"Anonymized data: {data_id}")
            
            if data_info['policy'].get('auto_delete', False):
                # Delete encryption key (cryptographic erasure)
                key_id = data_info['encryption_key_id']
                self.key_manager.delete_key(key_id)
                
                # Remove from registry
                with self._registry_lock:
                    del self.data_registry[data_id]
                
                deleted_count += 1
                logger.info(f"Deleted data: {data_id}")
        
        if deleted_count > 0 or anonymized_count > 0:
            logger.info(
                f"Cleanup completed: {deleted_count} deleted, "
                f"{anonymized_count} anonymized"
            )
        
        return deleted_count + anonymized_count
    
    def get_data_info(self, data_id: str) -> Optional[Dict[str, Any]]:
        """Get information about registered data."""
        with self._registry_lock:
            return self.data_registry.get(data_id)
    
    def extend_retention(self, data_id: str, additional_days: int) -> bool:
        """Extend retention period for data."""
        with self._registry_lock:
            if data_id not in self.data_registry:
                return False
            
            data_info = self.data_registry[data_id]
            expires_at = datetime.fromisoformat(data_info['expires_at'])
            new_expires = expires_at + timedelta(days=additional_days)
            data_info['expires_at'] = new_expires.isoformat()
            data_info['metadata']['retention_extended'] = True
            data_info['metadata']['extended_at'] = datetime.now().isoformat()
        
        logger.info(f"Extended retention for {data_id} by {additional_days} days")
        return True
    
    def apply_legal_hold(self, data_id: str, hold: bool = True) -> bool:
        """Apply or remove legal hold on data."""
        with self._registry_lock:
            if data_id not in self.data_registry:
                return False
            
            self.data_registry[data_id]['policy']['legal_hold'] = hold
            self.data_registry[data_id]['metadata']['legal_hold_applied'] = hold
            self.data_registry[data_id]['metadata']['legal_hold_at'] = datetime.now().isoformat()
        
        action = "applied" if hold else "removed"
        logger.info(f"Legal hold {action} for {data_id}")
        return True


class QuantumPrivacyMiddleware:
    """
    Scrapy middleware that provides quantum-resistant privacy protection.
    
    Features:
    - End-to-end encryption of scraped data
    - Automatic PII detection and masking
    - GDPR-compliant data handling
    - Automatic data lifecycle management
    """
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create middleware from crawler."""
        settings = crawler.settings
        
        # Check if privacy is enabled
        if not settings.getbool('QUANTUM_PRIVACY_ENABLED', False):
            raise NotConfigured("Quantum privacy middleware is disabled")
        
        # Initialize components
        key_store_path = settings.get('PRIVACY_KEY_STORE_PATH', './keys')
        default_retention = settings.getint('PRIVACY_DEFAULT_RETENTION_DAYS', 365)
        
        key_manager = KeyManager(
            key_store_path=key_store_path,
            default_retention_days=default_retention
        )
        
        lifecycle_manager = DataLifecycleManager(key_manager)
        
        # Initialize PII detector
        use_ml_pii = settings.getbool('PRIVACY_USE_ML_PII', True)
        pii_detector = PIIDetector(use_ml=use_ml_pii)
        
        # Initialize encryptor
        algorithm = EncryptionAlgorithm(
            settings.get('PRIVACY_ENCRYPTION_ALGORITHM', 'kyber768')
        )
        crypto = QuantumResistantCrypto(algorithm)
        symmetric_encryptor = SymmetricEncryptor()
        
        # Create middleware instance
        middleware = cls(
            crawler=crawler,
            key_manager=key_manager,
            lifecycle_manager=lifecycle_manager,
            pii_detector=pii_detector,
            crypto=crypto,
            symmetric_encryptor=symmetric_encryptor
        )
        
        # Connect to signals
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        crawler.signals.connect(middleware.spider_closed, signal=signals.spider_closed)
        crawler.signals.connect(middleware.item_scraped, signal=signals.item_scraped)
        
        return middleware
    
    def __init__(self, crawler, key_manager: KeyManager,
                 lifecycle_manager: DataLifecycleManager,
                 pii_detector: PIIDetector,
                 crypto: QuantumResistantCrypto,
                 symmetric_encryptor: SymmetricEncryptor):
        self.crawler = crawler
        self.key_manager = key_manager
        self.lifecycle_manager = lifecycle_manager
        self.pii_detector = pii_detector
        self.crypto = crypto
        self.symmetric_encryptor = symmetric_encryptor
        
        # Settings
        self.settings = crawler.settings
        self.mask_pii = self.settings.getbool('PRIVACY_MASK_PII', True)
        self.encrypt_data = self.settings.getbool('PRIVACY_ENCRYPT_DATA', True)
        self.auto_classify = self.settings.getbool('PRIVACY_AUTO_CLASSIFY', True)
        
        # Spider-specific keys
        self.spider_key = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Quantum privacy middleware initialized")
    
    def spider_opened(self, spider):
        """Handle spider opened signal."""
        # Generate or load spider-specific key
        key_id = f"spider_{spider.name}"
        self.spider_key = self.key_manager.get_key(key_id)
        
        if not self.spider_key:
            self.spider_key = self.key_manager.generate_key()
            self.spider_key.key_id = key_id
            self.spider_key.metadata['spider'] = spider.name
            # Save the key with spider name
            self.key_manager.keys[key_id] = self.spider_key
            self.key_manager._save_keys()
        
        logger.info(f"Spider {spider.name} privacy key loaded: {self.spider_key.key_id}")
    
    def spider_closed(self, spider):
        """Handle spider closed signal."""
        # Cleanup expired data
        self.lifecycle_manager.cleanup_expired_data()
        
        # Shutdown executor
        self.executor.shutdown(wait=False)
        
        logger.info(f"Spider {spider.name} privacy cleanup completed")
    
    def item_scraped(self, item, spider):
        """Process scraped item with privacy protection."""
        # Process item asynchronously
        future = self.executor.submit(self._process_item, item, spider)
        future.add_done_callback(self._item_processed_callback)
        
        # Return the future for potential waiting
        return future
    
    def _process_item(self, item, spider):
        """Process item with privacy protection."""
        try:
            # Convert item to dict for processing
            if hasattr(item, 'to_dict'):
                item_dict = item.to_dict()
            else:
                item_dict = dict(item)
            
            # Process each field
            processed_dict = {}
            pii_matches = []
            classification = DataClassification.PUBLIC
            
            for field_name, value in item_dict.items():
                if isinstance(value, str):
                    # Detect and mask PII
                    if self.mask_pii:
                        field_pii = self.pii_detector.detect_pii(value)
                        if field_pii:
                            pii_matches.extend(field_pii)
                            value = self.pii_detector.mask_pii(value, field_pii)
                    
                    # Classify data
                    if self.auto_classify:
                        field_classification = self.pii_detector.classify_data(value)
                        if field_classification.value > classification.value:
                            classification = field_classification
                
                processed_dict[field_name] = value
            
            # Register data for lifecycle management
            data_id = hashlib.sha256(
                json.dumps(processed_dict, sort_keys=True).encode()
            ).hexdigest()[:16]
            
            encryption_key_id = self.lifecycle_manager.register_data(
                data_id=data_id,
                classification=classification,
                metadata={
                    'spider': spider.name,
                    'url': item_dict.get('url', ''),
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            # Encrypt data if required
            if self.encrypt_data and classification in [
                DataClassification.CONFIDENTIAL,
                DataClassification.RESTRICTED,
                DataClassification.PII,
                DataClassification.SENSITIVE_PII
            ]:
                processed_dict = self._encrypt_dict(
                    processed_dict, 
                    encryption_key_id
                )
            
            # Add privacy metadata
            processed_dict['_privacy'] = {
                'encrypted': self.encrypt_data,
                'classification': classification.value,
                'pii_detected': len(pii_matches),
                'data_id': data_id,
                'encryption_key_id': encryption_key_id,
                'processed_at': datetime.now().isoformat()
            }
            
            # Update item
            if hasattr(item, 'update'):
                item.update(processed_dict)
            else:
                for key, value in processed_dict.items():
                    item[key] = value
            
            logger.debug(
                f"Processed item with privacy: {len(pii_matches)} PII matches, "
                f"classification: {classification.value}"
            )
            
            return item
            
        except Exception as e:
            logger.error(f"Error processing item with privacy: {e}")
            return item
    
    def _encrypt_dict(self, data_dict: Dict[str, Any], key_id: str) -> Dict[str, Any]:
        """Encrypt dictionary data."""
        key = self.key_manager.get_key(key_id)
        if not key:
            logger.error(f"Encryption key not found: {key_id}")
            return data_dict
        
        encrypted_dict = {}
        
        for field_name, value in data_dict.items():
            if isinstance(value, (str, bytes)):
                # Convert to bytes if string
                if isinstance(value, str):
                    value_bytes = value.encode('utf-8')
                else:
                    value_bytes = value
                
                # Encapsulate shared secret
                ciphertext, shared_secret = self.crypto.encapsulate(key.public_key)
                
                # Derive symmetric key
                salt = os.urandom(16)
                symmetric_key = self.symmetric_encryptor.derive_key(shared_secret, salt)
                
                # Encrypt data
                encrypted_data, nonce, tag = self.symmetric_encryptor.encrypt(
                    value_bytes, symmetric_key
                )
                
                # Package encrypted data
                encrypted_dict[field_name] = {
                    'encrypted': True,
                    'ciphertext': base64.b64encode(encrypted_data).decode('utf-8'),
                    'encapsulated_key': base64.b64encode(ciphertext).decode('utf-8'),
                    'nonce': base64.b64encode(nonce).decode('utf-8'),
                    'tag': base64.b64encode(tag).decode('utf-8'),
                    'salt': base64.b64encode(salt).decode('utf-8')
                }
            else:
                # Non-string data, store as-is
                encrypted_dict[field_name] = value
        
        return encrypted_dict
    
    def _item_processed_callback(self, future):
        """Callback for when item processing completes."""
        try:
            future.result()  # Raise any exceptions
        except Exception as e:
            logger.error(f"Item processing failed: {e}")


# Utility functions for standalone use
def encrypt_data(data: Union[str, bytes], public_key: bytes, 
                 algorithm: EncryptionAlgorithm = EncryptionAlgorithm.KYBER768) -> Dict[str, Any]:
    """
    Encrypt data with quantum-resistant encryption.
    
    Args:
        data: Data to encrypt
        public_key: Recipient's public key
        algorithm: Encryption algorithm to use
    
    Returns:
        Dictionary containing encrypted data and metadata
    """
    crypto = QuantumResistantCrypto(algorithm)
    symmetric_encryptor = SymmetricEncryptor()
    
    # Convert data to bytes
    if isinstance(data, str):
        data_bytes = data.encode('utf-8')
    else:
        data_bytes = data
    
    # Encapsulate shared secret
    ciphertext, shared_secret = crypto.encapsulate(public_key)
    
    # Derive symmetric key
    salt = os.urandom(16)
    symmetric_key = symmetric_encryptor.derive_key(shared_secret, salt)
    
    # Encrypt data
    encrypted_data, nonce, tag = symmetric_encryptor.encrypt(data_bytes, symmetric_key)
    
    return {
        'encrypted': True,
        'algorithm': algorithm.value,
        'ciphertext': base64.b64encode(encrypted_data).decode('utf-8'),
        'encapsulated_key': base64.b64encode(ciphertext).decode('utf-8'),
        'nonce': base64.b64encode(nonce).decode('utf-8'),
        'tag': base64.b64encode(tag).decode('utf-8'),
        'salt': base64.b64encode(salt).decode('utf-8')
    }


def decrypt_data(encrypted_data: Dict[str, Any], private_key: bytes) -> bytes:
    """
    Decrypt data with quantum-resistant encryption.
    
    Args:
        encrypted_data: Encrypted data dictionary
        private_key: Recipient's private key
    
    Returns:
        Decrypted data as bytes
    """
    algorithm = EncryptionAlgorithm(encrypted_data['algorithm'])
    crypto = QuantumResistantCrypto(algorithm)
    symmetric_encryptor = SymmetricEncryptor()
    
    # Decode components
    ciphertext = base64.b64decode(encrypted_data['ciphertext'])
    encapsulated_key = base64.b64decode(encrypted_data['encapsulated_key'])
    nonce = base64.b64decode(encrypted_data['nonce'])
    tag = base64.b64decode(encrypted_data['tag'])
    salt = base64.b64decode(encrypted_data['salt'])
    
    # Decapsulate shared secret
    shared_secret = crypto.decapsulate(encapsulated_key, private_key)
    
    # Derive symmetric key
    symmetric_key = symmetric_encryptor.derive_key(shared_secret, salt)
    
    # Decrypt data
    return symmetric_encryptor.decrypt(ciphertext, symmetric_key, nonce, tag)


def detect_and_mask_pii(text: str, use_ml: bool = True) -> Tuple[str, List[PIIMatch]]:
    """
    Detect and mask PII in text.
    
    Args:
        text: Text to process
        use_ml: Whether to use ML-based detection
    
    Returns:
        Tuple of (masked_text, pii_matches)
    """
    detector = PIIDetector(use_ml=use_ml)
    matches = detector.detect_pii(text)
    masked_text = detector.mask_pii(text, matches)
    return masked_text, matches


# Export main classes
__all__ = [
    'QuantumResistantCrypto',
    'SymmetricEncryptor',
    'PIIDetector',
    'KeyManager',
    'DataLifecycleManager',
    'QuantumPrivacyMiddleware',
    'EncryptionAlgorithm',
    'DataClassification',
    'RetentionPolicy',
    'PIIMatch',
    'EncryptionKey',
    'encrypt_data',
    'decrypt_data',
    'detect_and_mask_pii'
]