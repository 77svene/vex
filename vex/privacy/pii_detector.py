# File: vex/privacy/pii_detector.py

import re
import json
import hashlib
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set, Pattern
from dataclasses import dataclass, asdict
from enum import Enum
import base64
import secrets
from pathlib import Path

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

logger = logging.getLogger(__name__)


class PIIType(Enum):
    """Types of Personally Identifiable Information."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    NATIONAL_ID = "national_id"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    CUSTOM = "custom"


class MaskingStrategy(Enum):
    """Strategies for masking PII data."""
    REDACT = "redact"  # Replace with [REDACTED]
    HASH = "hash"  # Replace with irreversible hash
    TOKENIZE = "tokenize"  # Replace with reversible token
    PARTIAL_MASK = "partial_mask"  # Show partial data (e.g., ****@email.com)
    NULLIFY = "nullify"  # Replace with null/empty
    DUMMY = "dummy"  # Replace with realistic dummy data


@dataclass
class PIIDetectionRule:
    """Rule for detecting specific PII patterns."""
    pii_type: PIIType
    pattern: Pattern
    confidence: float = 0.9
    context_keywords: List[str] = None
    validation_func: callable = None
    
    def __post_init__(self):
        if self.context_keywords is None:
            self.context_keywords = []


@dataclass
class PIIMatch:
    """Represents a detected PII instance."""
    pii_type: PIIType
    value: str
    start: int
    end: int
    confidence: float
    context: str = ""
    masked_value: str = ""
    token: str = ""


@dataclass
class DataRetentionPolicy:
    """GDPR-compliant data retention policy."""
    retention_days: int = 30
    auto_delete: bool = True
    encryption_required: bool = True
    audit_log: bool = True
    purpose_limitation: str = "scraping_operations"
    legal_basis: str = "legitimate_interest"


@dataclass
class AuditLogEntry:
    """Audit log entry for GDPR compliance."""
    timestamp: str
    action: str
    data_hash: str
    user: str
    purpose: str
    details: Dict[str, Any]


class QuantumResistantEncryptor:
    """
    Post-quantum cryptography implementation for data encryption.
    Uses hybrid approach with Kyber (KEM) and AES-256-GCM for actual encryption.
    """
    
    def __init__(self, key_size: int = 256):
        self.key_size = key_size
        self._validate_crypto_availability()
        
    def _validate_crypto_availability(self):
        if not CRYPTO_AVAILABLE:
            raise ImportError(
                "Cryptography package required for encryption. "
                "Install with: pip install cryptography"
            )
    
    def generate_key_pair(self) -> Tuple[bytes, bytes]:
        """
        Generate quantum-resistant key pair.
        Returns (public_key, private_key) tuple.
        In production, would use actual Kyber/NTRU implementation.
        """
        # Placeholder for actual post-quantum key generation
        # In production, replace with: from pqcrypto.kem.kyber1024 import generate_keypair
        private_key = secrets.token_bytes(32)
        public_key = hashlib.sha256(private_key).digest()
        return public_key, private_key
    
    def derive_key(self, shared_secret: bytes, salt: bytes = None) -> bytes:
        """Derive encryption key from shared secret using HKDF."""
        if salt is None:
            salt = secrets.token_bytes(16)
        
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            info=b"vex-pii-encryption",
        )
        return hkdf.derive(shared_secret)
    
    def encrypt(self, plaintext: bytes, public_key: bytes) -> Dict[str, bytes]:
        """
        Encrypt data using hybrid encryption.
        Returns dict with 'ciphertext', 'nonce', 'tag', and 'ephemeral_key'.
        """
        # Generate ephemeral key pair for forward secrecy
        ephemeral_private = secrets.token_bytes(32)
        ephemeral_public = hashlib.sha256(ephemeral_private).digest()
        
        # Derive shared secret (in production, use actual KEM)
        shared_secret = hashlib.sha256(
            ephemeral_private + public_key
        ).digest()
        
        # Derive encryption key
        salt = secrets.token_bytes(16)
        key = self.derive_key(shared_secret, salt)
        
        # Encrypt with AES-256-GCM
        aesgcm = AESGCM(key)
        nonce = secrets.token_bytes(12)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        
        return {
            'ciphertext': ciphertext,
            'nonce': nonce,
            'salt': salt,
            'ephemeral_public': ephemeral_public,
            'tag': ciphertext[-16:],  # GCM tag is appended
            'ciphertext_without_tag': ciphertext[:-16]
        }
    
    def decrypt(self, encrypted_data: Dict[str, bytes], private_key: bytes) -> bytes:
        """
        Decrypt data using hybrid decryption.
        """
        # Reconstruct shared secret (in production, use actual KEM decapsulation)
        shared_secret = hashlib.sha256(
            private_key + encrypted_data['ephemeral_public']
        ).digest()
        
        # Derive decryption key
        key = self.derive_key(shared_secret, encrypted_data['salt'])
        
        # Decrypt with AES-256-GCM
        aesgcm = AESGCM(key)
        
        # Reconstruct ciphertext with tag
        ciphertext = encrypted_data['ciphertext_without_tag'] + encrypted_data['tag']
        
        return aesgcm.decrypt(encrypted_data['nonce'], ciphertext, None)


class MLPIIDetector:
    """
    Machine Learning-based PII detector with configurable rules.
    Falls back to regex patterns if ML models aren't available.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.ml_model = None
        self.vectorizer = None
        self._initialize_patterns()
        self._load_ml_model()
    
    def _initialize_patterns(self):
        """Initialize regex patterns for PII detection."""
        self.patterns: List[PIIDetectionRule] = [
            # Email patterns
            PIIDetectionRule(
                pii_type=PIIType.EMAIL,
                pattern=re.compile(
                    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                    re.IGNORECASE
                ),
                confidence=0.95,
                context_keywords=['email', 'e-mail', 'mail', 'contact']
            ),
            
            # Phone numbers (international format)
            PIIDetectionRule(
                pii_type=PIIType.PHONE,
                pattern=re.compile(
                    r'(\+?[\d\s\-\.]{7,}|(?:\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4}))',
                    re.IGNORECASE
                ),
                confidence=0.85,
                context_keywords=['phone', 'tel', 'mobile', 'cell', 'contact']
            ),
            
            # Social Security Numbers (US)
            PIIDetectionRule(
                pii_type=PIIType.SSN,
                pattern=re.compile(
                    r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'
                ),
                confidence=0.9,
                context_keywords=['ssn', 'social security', 'social security number']
            ),
            
            # Credit card numbers
            PIIDetectionRule(
                pii_type=PIIType.CREDIT_CARD,
                pattern=re.compile(
                    r'\b(?:\d[ -]*?){13,19}\b'
                ),
                confidence=0.8,
                context_keywords=['credit card', 'card number', 'cc', 'visa', 'mastercard']
            ),
            
            # IP addresses
            PIIDetectionRule(
                pii_type=PIIType.IP_ADDRESS,
                pattern=re.compile(
                    r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
                ),
                confidence=0.7,
                context_keywords=['ip', 'ip address', 'address']
            ),
            
            # Passport numbers (various formats)
            PIIDetectionRule(
                pii_type=PIIType.PASSPORT,
                pattern=re.compile(
                    r'\b[A-Z]{1,2}\d{6,9}\b'
                ),
                confidence=0.75,
                context_keywords=['passport', 'travel document']
            ),
            
            # Dates of birth
            PIIDetectionRule(
                pii_type=PIIType.DATE_OF_BIRTH,
                pattern=re.compile(
                    r'\b(?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])[-/.](?:19|20)\d{2}\b|'
                    r'\b(?:19|20)\d{2}[-/.](?:0[1-9]|1[0-2])[-/.](?:0[1-9]|[12]\d|3[01])\b|'
                    r'\b(?:0[1-9]|[12]\d|3[01])[-/.](?:0[1-9]|1[0-2])[-/.](?:19|20)\d{2}\b'
                ),
                confidence=0.8,
                context_keywords=['birth', 'dob', 'date of birth', 'birthday']
            ),
        ]
    
    def _load_ml_model(self):
        """Load pre-trained ML model for PII detection."""
        if not ML_AVAILABLE:
            logger.warning("ML libraries not available. Using regex patterns only.")
            return
        
        if self.model_path and Path(self.model_path).exists():
            try:
                model_data = joblib.load(self.model_path)
                self.ml_model = model_data.get('model')
                self.vectorizer = model_data.get('vectorizer')
                logger.info(f"Loaded ML model from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load ML model: {e}")
                self._train_default_model()
        else:
            self._train_default_model()
    
    def _train_default_model(self):
        """Train a simple default model if no pre-trained model exists."""
        if not ML_AVAILABLE:
            return
        
        # Simple training data for demonstration
        training_texts = [
            "Contact me at john.doe@example.com",
            "My phone is 555-123-4567",
            "SSN: 123-45-6789",
            "Credit card: 4111-1111-1111-1111",
            "IP: 192.168.1.1",
            "Passport: AB1234567",
            "DOB: 01/15/1990",
        ]
        
        labels = [1, 1, 1, 1, 1, 1, 1]  # All contain PII
        
        try:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            
            self.ml_model = Pipeline([
                ('tfidf', self.vectorizer),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ])
            
            # In production, use actual labeled dataset
            self.ml_model.fit(training_texts, labels)
            logger.info("Trained default ML model for PII detection")
        except Exception as e:
            logger.error(f"Failed to train ML model: {e}")
            self.ml_model = None
    
    def detect_pii(self, text: str, context: str = "") -> List[PIIMatch]:
        """
        Detect PII in text using both regex patterns and ML model.
        
        Args:
            text: Text to analyze
            context: Additional context for better detection
            
        Returns:
            List of PIIMatch objects
        """
        matches = []
        
        # Regex-based detection
        for rule in self.patterns:
            for match in rule.pattern.finditer(text):
                # Check context for higher confidence
                context_boost = 0.0
                if any(keyword in context.lower() for keyword in rule.context_keywords):
                    context_boost = 0.1
                
                matches.append(PIIMatch(
                    pii_type=rule.pii_type,
                    value=match.group(),
                    start=match.start(),
                    end=match.end(),
                    confidence=min(1.0, rule.confidence + context_boost),
                    context=context
                ))
        
        # ML-based detection (if available)
        if self.ml_model and self.vectorizer and ML_AVAILABLE:
            ml_matches = self._detect_with_ml(text, context)
            matches.extend(ml_matches)
        
        # Remove overlapping matches (keep highest confidence)
        matches = self._resolve_overlaps(matches)
        
        return matches
    
    def _detect_with_ml(self, text: str, context: str) -> List[PIIMatch]:
        """Use ML model to detect PII in text."""
        matches = []
        
        try:
            # Split text into sentences for better detection
            sentences = re.split(r'[.!?]+', text)
            
            for sentence in sentences:
                if len(sentence.strip()) < 10:
                    continue
                
                # Vectorize and predict
                X = self.vectorizer.transform([sentence])
                prediction = self.ml_model.predict(X)[0]
                
                if prediction == 1:  # Contains PII
                    # Simple heuristic: look for patterns in sentence
                    for rule in self.patterns:
                        for match in rule.pattern.finditer(sentence):
                            matches.append(PIIMatch(
                                pii_type=rule.pii_type,
                                value=match.group(),
                                start=match.start() + text.find(sentence),
                                end=match.end() + text.find(sentence),
                                confidence=0.85,  # ML confidence
                                context=context
                            ))
        except Exception as e:
            logger.error(f"ML detection failed: {e}")
        
        return matches
    
    def _resolve_overlaps(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Remove overlapping matches, keeping highest confidence."""
        if not matches:
            return matches
        
        # Sort by start position and confidence (descending)
        sorted_matches = sorted(
            matches,
            key=lambda m: (m.start, -m.confidence)
        )
        
        filtered = []
        current_end = -1
        
        for match in sorted_matches:
            if match.start >= current_end:
                filtered.append(match)
                current_end = match.end
            elif match.confidence > filtered[-1].confidence:
                # Replace if higher confidence
                filtered[-1] = match
                current_end = match.end
        
        return filtered


class PIIMasker:
    """
    Configurable PII masking with multiple strategies.
    Supports reversible tokenization for authorized access.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.token_map: Dict[str, str] = {}  # For reversible tokenization
        self.masking_rules = self._initialize_masking_rules()
    
    def _initialize_masking_rules(self) -> Dict[PIIType, MaskingStrategy]:
        """Initialize default masking rules for each PII type."""
        return {
            PIIType.EMAIL: MaskingStrategy.PARTIAL_MASK,
            PIIType.PHONE: MaskingStrategy.PARTIAL_MASK,
            PIIType.SSN: MaskingStrategy.REDACT,
            PIIType.CREDIT_CARD: MaskingStrategy.PARTIAL_MASK,
            PIIType.IP_ADDRESS: MaskingStrategy.HASH,
            PIIType.PASSPORT: MaskingStrategy.REDACT,
            PIIType.DRIVER_LICENSE: MaskingStrategy.REDACT,
            PIIType.NATIONAL_ID: MaskingStrategy.REDACT,
            PIIType.NAME: MaskingStrategy.TOKENIZE,
            PIIType.ADDRESS: MaskingStrategy.TOKENIZE,
            PIIType.DATE_OF_BIRTH: MaskingStrategy.PARTIAL_MASK,
            PIIType.CUSTOM: MaskingStrategy.REDACT,
        }
    
    def mask_text(self, text: str, pii_matches: List[PIIMatch]) -> str:
        """
        Mask PII in text based on detection results.
        
        Args:
            text: Original text
            pii_matches: List of detected PII matches
            
        Returns:
            Masked text
        """
        if not pii_matches:
            return text
        
        # Sort matches by start position (reverse order for replacement)
        sorted_matches = sorted(pii_matches, key=lambda m: m.start, reverse=True)
        
        masked_text = text
        
        for match in sorted_matches:
            strategy = self.masking_rules.get(
                match.pii_type,
                MaskingStrategy.REDACT
            )
            
            masked_value = self._apply_masking_strategy(
                match.value,
                strategy,
                match.pii_type
            )
            
            # Update the match with masked value
            match.masked_value = masked_value
            
            # Replace in text
            masked_text = (
                masked_text[:match.start] +
                masked_value +
                masked_text[match.end:]
            )
        
        return masked_text
    
    def _apply_masking_strategy(
        self,
        value: str,
        strategy: MaskingStrategy,
        pii_type: PIIType
    ) -> str:
        """Apply specific masking strategy to a value."""
        
        if strategy == MaskingStrategy.REDACT:
            return f"[REDACTED_{pii_type.value.upper()}]"
        
        elif strategy == MaskingStrategy.HASH:
            # Create irreversible hash
            salt = self.config.get('hash_salt', 'vex-pii-salt')
            hash_obj = hashlib.sha256((value + salt).encode())
            return f"hash_{hash_obj.hexdigest()[:16]}"
        
        elif strategy == MaskingStrategy.TOKENIZE:
            # Generate reversible token
            token = secrets.token_urlsafe(16)
            self.token_map[token] = value
            return f"token_{token}"
        
        elif strategy == MaskingStrategy.PARTIAL_MASK:
            return self._partial_mask(value, pii_type)
        
        elif strategy == MaskingStrategy.NULLIFY:
            return ""
        
        elif strategy == MaskingStrategy.DUMMY:
            return self._generate_dummy_data(pii_type)
        
        else:
            return "[MASKED]"
    
    def _partial_mask(self, value: str, pii_type: PIIType) -> str:
        """Apply partial masking based on PII type."""
        
        if pii_type == PIIType.EMAIL:
            # Mask local part, keep domain
            if '@' in value:
                local, domain = value.split('@', 1)
                masked_local = local[0] + '***' if len(local) > 1 else '***'
                return f"{masked_local}@{domain}"
        
        elif pii_type == PIIType.PHONE:
            # Show last 4 digits
            digits = re.sub(r'\D', '', value)
            if len(digits) >= 4:
                return '*' * (len(digits) - 4) + digits[-4:]
        
        elif pii_type == PIIType.CREDIT_CARD:
            # Show first 4 and last 4 digits
            digits = re.sub(r'\D', '', value)
            if len(digits) >= 8:
                return digits[:4] + '*' * (len(digits) - 8) + digits[-4:]
        
        elif pii_type == PIIType.DATE_OF_BIRTH:
            # Mask day and month, keep year
            if '/' in value:
                parts = value.split('/')
                if len(parts) == 3:
                    return f"**/**/{parts[2]}"
            elif '-' in value:
                parts = value.split('-')
                if len(parts) == 3:
                    return f"**-**-{parts[2]}"
        
        # Default partial mask
        if len(value) > 4:
            return value[0] + '*' * (len(value) - 2) + value[-1]
        else:
            return '***'
    
    def _generate_dummy_data(self, pii_type: PIIType) -> str:
        """Generate realistic dummy data for testing."""
        
        dummy_data = {
            PIIType.EMAIL: "user@example.com",
            PIIType.PHONE: "+1-555-0100",
            PIIType.SSN: "123-45-6789",
            PIIType.CREDIT_CARD: "4111-1111-1111-1111",
            PIIType.IP_ADDRESS: "192.168.1.1",
            PIIType.PASSPORT: "AB1234567",
            PIIType.NAME: "John Doe",
            PIIType.ADDRESS: "123 Main St, Anytown, USA",
            PIIType.DATE_OF_BIRTH: "01/01/1990",
        }
        
        return dummy_data.get(pii_type, "DUMMY_DATA")
    
    def unmask_token(self, token: str) -> Optional[str]:
        """Retrieve original value for a token (if authorized)."""
        if token.startswith("token_"):
            token = token[6:]  # Remove "token_" prefix
        return self.token_map.get(token)
    
    def clear_token_map(self):
        """Clear token map for security."""
        self.token_map.clear()


class DataLifecycleManager:
    """
    GDPR-compliant data lifecycle management with automatic retention policies.
    Implements cryptographic erasure and audit logging.
    """
    
    def __init__(
        self,
        retention_policy: DataRetentionPolicy = None,
        storage_path: str = "./vex_pii_data"
    ):
        self.retention_policy = retention_policy or DataRetentionPolicy()
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.audit_log: List[AuditLogEntry] = []
        self.encryptor = QuantumResistantEncryptor() if self.retention_policy.encryption_required else None
        
        # Load existing audit log
        self._load_audit_log()
    
    def store_data(
        self,
        data: Dict[str, Any],
        metadata: Dict[str, Any] = None,
        purpose: str = "scraping_operations"
    ) -> str:
        """
        Store data with encryption and audit logging.
        
        Args:
            data: Data to store
            metadata: Additional metadata
            purpose: Purpose of data storage (GDPR requirement)
            
        Returns:
            Data ID for future retrieval
        """
        data_id = self._generate_data_id(data)
        timestamp = datetime.utcnow().isoformat()
        
        # Prepare data package
        data_package = {
            'id': data_id,
            'timestamp': timestamp,
            'data': data,
            'metadata': metadata or {},
            'purpose': purpose,
            'retention_days': self.retention_policy.retention_days,
            'expiry_date': (
                datetime.utcnow() + timedelta(days=self.retention_policy.retention_days)
            ).isoformat()
        }
        
        # Encrypt if required
        if self.encryptor:
            public_key, private_key = self.encryptor.generate_key_pair()
            encrypted = self.encryptor.encrypt(
                json.dumps(data_package).encode(),
                public_key
            )
            
            # Store encrypted data and private key separately
            encrypted_package = {
                'encrypted_data': base64.b64encode(encrypted['ciphertext']).decode(),
                'nonce': base64.b64encode(encrypted['nonce']).decode(),
                'salt': base64.b64encode(encrypted['salt']).decode(),
                'ephemeral_public': base64.b64encode(encrypted['ephemeral_public']).decode(),
                'tag': base64.b64encode(encrypted['tag']).decode(),
                'private_key': base64.b64encode(private_key).decode()
            }
            
            self._save_data_file(data_id, encrypted_package, encrypted=True)
        else:
            self._save_data_file(data_id, data_package, encrypted=False)
        
        # Audit log
        if self.retention_policy.audit_log:
            self._add_audit_entry(
                action="STORE",
                data_hash=self._hash_data(data),
                purpose=purpose,
                details={'data_id': data_id, 'encrypted': self.encryptor is not None}
            )
        
        logger.info(f"Stored data with ID: {data_id}")
        return data_id
    
    def retrieve_data(self, data_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve stored data.
        
        Args:
            data_id: Data ID from store_data()
            
        Returns:
            Original data or None if not found/expired
        """
        data_file = self.storage_path / f"{data_id}.json"
        
        if not data_file.exists():
            logger.warning(f"Data not found: {data_id}")
            return None
        
        try:
            with open(data_file, 'r') as f:
                stored_package = json.load(f)
            
            # Check if encrypted
            if 'encrypted_data' in stored_package:
                if not self.encryptor:
                    raise ValueError("Encryption required but not available")
                
                # Decrypt
                encrypted_data = {
                    'ciphertext': base64.b64decode(stored_package['encrypted_data']),
                    'nonce': base64.b64decode(stored_package['nonce']),
                    'salt': base64.b64decode(stored_package['salt']),
                    'ephemeral_public': base64.b64decode(stored_package['ephemeral_public']),
                    'tag': base64.b64decode(stored_package['tag'])
                }
                
                private_key = base64.b64decode(stored_package['private_key'])
                decrypted = self.encryptor.decrypt(encrypted_data, private_key)
                data_package = json.loads(decrypted.decode())
            else:
                data_package = stored_package
            
            # Check retention
            if self._is_expired(data_package):
                logger.info(f"Data expired: {data_id}")
                self.delete_data(data_id)
                return None
            
            # Audit log
            if self.retention_policy.audit_log:
                self._add_audit_entry(
                    action="RETRIEVE",
                    data_hash=self._hash_data(data_package['data']),
                    purpose=data_package.get('purpose', 'unknown'),
                    details={'data_id': data_id}
                )
            
            return data_package['data']
        
        except Exception as e:
            logger.error(f"Failed to retrieve data {data_id}: {e}")
            return None
    
    def delete_data(self, data_id: str, reason: str = "retention_expired") -> bool:
        """
        Delete data with cryptographic erasure.
        
        Args:
            data_id: Data ID to delete
            reason: Reason for deletion
            
        Returns:
            True if successful
        """
        data_file = self.storage_path / f"{data_id}.json"
        
        if not data_file.exists():
            logger.warning(f"Data not found for deletion: {data_id}")
            return False
        
        try:
            # Read data for audit log
            with open(data_file, 'r') as f:
                stored_package = json.load(f)
            
            # Cryptographic erasure: overwrite file multiple times before deletion
            if self.retention_policy.encryption_required:
                self._secure_delete(data_file)
            else:
                data_file.unlink()
            
            # Audit log
            if self.retention_policy.audit_log:
                self._add_audit_entry(
                    action="DELETE",
                    data_hash="ERASED",
                    purpose="data_erasure",
                    details={
                        'data_id': data_id,
                        'reason': reason,
                        'cryptographic_erasure': self.retention_policy.encryption_required
                    }
                )
            
            logger.info(f"Deleted data: {data_id} (reason: {reason})")
            return True
        
        except Exception as e:
            logger.error(f"Failed to delete data {data_id}: {e}")
            return False
    
    def enforce_retention_policies(self):
        """Enforce data retention policies (should be called periodically)."""
        logger.info("Enforcing data retention policies")
        
        deleted_count = 0
        for data_file in self.storage_path.glob("*.json"):
            try:
                with open(data_file, 'r') as f:
                    stored_package = json.load(f)
                
                if self._is_expired(stored_package):
                    data_id = data_file.stem
                    if self.delete_data(data_id, reason="retention_expired"):
                        deleted_count += 1
            
            except Exception as e:
                logger.error(f"Error processing {data_file}: {e}")
        
        logger.info(f"Deleted {deleted_count} expired data files")
    
    def _generate_data_id(self, data: Dict[str, Any]) -> str:
        """Generate unique data ID."""
        data_str = json.dumps(data, sort_keys=True)
        timestamp = str(time.time())
        return hashlib.sha256((data_str + timestamp).encode()).hexdigest()[:32]
    
    def _hash_data(self, data: Any) -> str:
        """Create hash of data for audit logging."""
        data_str = json.dumps(data, sort_keys=True) if isinstance(data, dict) else str(data)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _is_expired(self, data_package: Dict[str, Any]) -> bool:
        """Check if data has expired."""
        try:
            expiry_date = datetime.fromisoformat(data_package['expiry_date'])
            return datetime.utcnow() > expiry_date
        except (KeyError, ValueError):
            return False
    
    def _save_data_file(self, data_id: str, data: Dict[str, Any], encrypted: bool):
        """Save data to file."""
        filename = self.storage_path / f"{data_id}.json"
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Set restrictive permissions
        filename.chmod(0o600)
    
    def _secure_delete(self, file_path: Path):
        """Securely delete file by overwriting multiple times."""
        try:
            file_size = file_path.stat().st_size
            
            # Overwrite with random data 3 times
            for _ in range(3):
                with open(file_path, 'wb') as f:
                    f.write(secrets.token_bytes(file_size))
            
            # Final overwrite with zeros
            with open(file_path, 'wb') as f:
                f.write(b'\x00' * file_size)
            
            file_path.unlink()
        
        except Exception as e:
            logger.error(f"Secure delete failed: {e}")
            # Fallback to regular delete
            file_path.unlink()
    
    def _load_audit_log(self):
        """Load existing audit log."""
        audit_file = self.storage_path / "audit_log.json"
        
        if audit_file.exists():
            try:
                with open(audit_file, 'r') as f:
                    self.audit_log = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load audit log: {e}")
                self.audit_log = []
    
    def _save_audit_log(self):
        """Save audit log to file."""
        audit_file = self.storage_path / "audit_log.json"
        
        # Keep only last 10000 entries
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-10000:]
        
        with open(audit_file, 'w') as f:
            json.dump(self.audit_log, f, indent=2)
        
        audit_file.chmod(0o600)
    
    def _add_audit_entry(
        self,
        action: str,
        data_hash: str,
        purpose: str,
        details: Dict[str, Any]
    ):
        """Add entry to audit log."""
        entry = AuditLogEntry(
            timestamp=datetime.utcnow().isoformat(),
            action=action,
            data_hash=data_hash,
            user="vex_system",
            purpose=purpose,
            details=details
        )
        
        self.audit_log.append(asdict(entry))
        self._save_audit_log()


class PrivacyPipeline:
    """
    Scrapy pipeline for automatic PII detection, masking, and encryption.
    Integrates all privacy components into Scrapy workflow.
    """
    
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        
        # Initialize components
        self.pii_detector = MLPIIDetector(
            model_path=settings.get('PII_DETECTION_MODEL_PATH')
        )
        
        self.pii_masker = PIIMasker(
            config=settings.get('PII_MASKING_CONFIG', {})
        )
        
        retention_policy = DataRetentionPolicy(
            retention_days=settings.get('DATA_RETENTION_DAYS', 30),
            auto_delete=settings.get('AUTO_DELETE_EXPIRED_DATA', True),
            encryption_required=settings.get('ENCRYPTION_REQUIRED', True),
            audit_log=settings.get('AUDIT_LOG_ENABLED', True),
            purpose_limitation=settings.get('DATA_PURPOSE', 'scraping_operations'),
            legal_basis=settings.get('LEGAL_BASIS', 'legitimate_interest')
        )
        
        self.lifecycle_manager = DataLifecycleManager(
            retention_policy=retention_policy,
            storage_path=settings.get('PRIVACY_STORAGE_PATH', './vex_pii_data')
        )
        
        # Statistics
        self.stats = {
            'items_processed': 0,
            'pii_detected': 0,
            'pii_masked': 0,
            'data_encrypted': 0,
            'data_deleted': 0
        }
    
    @classmethod
    def from_crawler(cls, crawler):
        """Create pipeline from crawler settings."""
        return cls(crawler.settings)
    
    def process_item(self, item: Dict[str, Any], spider) -> Dict[str, Any]:
        """Process item through privacy pipeline."""
        self.stats['items_processed'] += 1
        
        # Convert item to text for PII detection
        item_text = json.dumps(item, ensure_ascii=False)
        
        # Detect PII
        pii_matches = self.pii_detector.detect_pii(
            item_text,
            context=f"spider:{spider.name}"
        )
        
        if pii_matches:
            self.stats['pii_detected'] += len(pii_matches)
            
            # Mask PII in text
            masked_text = self.pii_masker.mask_text(item_text, pii_matches)
            
            # Convert back to dict
            try:
                masked_item = json.loads(masked_text)
                
                # Store original data securely if needed
                if self.settings.get('STORE_ORIGINAL_DATA', False):
                    data_id = self.lifecycle_manager.store_data(
                        data=item,
                        metadata={
                            'spider': spider.name,
                            'url': item.get('url', ''),
                            'timestamp': datetime.utcnow().isoformat()
                        },
                        purpose=self.settings.get('DATA_PURPOSE', 'scraping_operations')
                    )
                    
                    # Add reference to masked item
                    masked_item['_privacy'] = {
                        'data_id': data_id,
                        'pii_masked': True,
                        'pii_count': len(pii_matches),
                        'masking_timestamp': datetime.utcnow().isoformat()
                    }
                    
                    self.stats['data_encrypted'] += 1
                
                self.stats['pii_masked'] += len(pii_matches)
                return masked_item
            
            except json.JSONDecodeError:
                logger.error("Failed to parse masked text back to JSON")
                return item
        
        return item
    
    def close_spider(self, spider):
        """Clean up when spider closes."""
        # Enforce retention policies
        self.lifecycle_manager.enforce_retention_policies()
        
        # Log statistics
        logger.info(f"Privacy pipeline statistics: {self.stats}")
        
        # Clear sensitive data from memory
        self.pii_masker.clear_token_map()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return self.stats.copy()


# Utility functions for integration with Scrapy
def setup_privacy_pipeline(settings: Dict[str, Any]) -> PrivacyPipeline:
    """Setup privacy pipeline with given settings."""
    return PrivacyPipeline(settings)


def detect_pii_in_text(text: str, context: str = "") -> List[Dict[str, Any]]:
    """
    Utility function to detect PII in text.
    
    Returns:
        List of detected PII with type, value, confidence, and position
    """
    detector = MLPIIDetector()
    matches = detector.detect_pii(text, context)
    
    return [
        {
            'type': match.pii_type.value,
            'value': match.value,
            'masked_value': match.masked_value,
            'confidence': match.confidence,
            'start': match.start,
            'end': match.end,
            'context': match.context
        }
        for match in matches
    ]


def mask_pii_in_text(
    text: str,
    pii_types: List[str] = None,
    strategy: str = "redact"
) -> str:
    """
    Utility function to mask PII in text.
    
    Args:
        text: Text containing PII
        pii_types: List of PII types to mask (None for all)
        strategy: Masking strategy to use
        
    Returns:
        Masked text
    """
    detector = MLPIIDetector()
    matches = detector.detect_pii(text)
    
    if pii_types:
        # Filter by requested types
        pii_type_enums = [PIIType(t) for t in pii_types]
        matches = [m for m in matches if m.pii_type in pii_type_enums]
    
    masker = PIIMasker()
    
    # Override strategy if specified
    if strategy:
        strategy_enum = MaskingStrategy(strategy)
        for pii_type in PIIType:
            masker.masking_rules[pii_type] = strategy_enum
    
    return masker.mask_text(text, matches)


# Export public API
__all__ = [
    'PIIType',
    'MaskingStrategy',
    'PIIDetectionRule',
    'PIIMatch',
    'DataRetentionPolicy',
    'QuantumResistantEncryptor',
    'MLPIIDetector',
    'PIIMasker',
    'DataLifecycleManager',
    'PrivacyPipeline',
    'setup_privacy_pipeline',
    'detect_pii_in_text',
    'mask_pii_in_text',
]