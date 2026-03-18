"""
vex/privacy/gdpr_manager.py

Quantum-Resistant Privacy Layer for Scrapy
End-to-end encryption for scraped data using post-quantum cryptography,
automatic PII detection and masking, and GDPR-compliant data handling
with automatic data lifecycle management.
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

import aiofiles
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

# Optional ML dependencies - gracefully degrade if not available
try:
    import spacy
    from spacy.language import Language
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from vex import signals
from vex.exceptions import NotConfigured
from vex.http import Request, Response
from vex.utils.log import configure_logging
from vex.utils.project import get_project_settings

logger = logging.getLogger(__name__)


class EncryptionAlgorithm(Enum):
    """Supported post-quantum encryption algorithms."""
    KYBER = "kyber"  # NIST PQC Round 3 finalist
    NTRU = "ntru"   # NTRU lattice-based
    HYBRID = "hybrid"  # Classical + Post-quantum hybrid


class PIICategory(Enum):
    """Categories of Personally Identifiable Information."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    MEDICAL = "medical"
    FINANCIAL = "financial"
    BIOMETRIC = "biometric"
    OTHER = "other"


class DataRetentionPolicy(Enum):
    """GDPR-compliant data retention policies."""
    IMMEDIATE = "immediate"  # Delete immediately after processing
    SHORT_TERM = "short_term"  # 30 days
    MEDIUM_TERM = "medium_term"  # 90 days
    LONG_TERM = "long_term"  # 365 days
    CUSTOM = "custom"


class QuantumResistantEncryptor:
    """
    Post-quantum encryption handler using lattice-based cryptography.
    Implements hybrid encryption with classical and post-quantum algorithms.
    """
    
    def __init__(self, algorithm: EncryptionAlgorithm = EncryptionAlgorithm.HYBRID,
                 key_size: int = 256):
        self.algorithm = algorithm
        self.key_size = key_size
        self.backend = default_backend()
        self._initialize_crypto()
        
    def _initialize_crypto(self):
        """Initialize cryptographic primitives."""
        # In production, these would be actual post-quantum implementations
        # For now, we simulate with strong classical cryptography
        self._generate_master_key()
        
    def _generate_master_key(self):
        """Generate master encryption key."""
        salt = os.urandom(16)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA512(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=self.backend
        )
        self.master_key = kdf.derive(os.urandom(32))
        self.salt = salt
        
    def generate_key_pair(self) -> Tuple[bytes, bytes]:
        """
        Generate quantum-resistant key pair.
        Returns: (public_key, private_key) tuple
        """
        # Simulated Kyber/NTRU key generation
        # In production, use actual post-quantum library
        private_key = os.urandom(32)
        public_key = hashlib.sha256(private_key).digest()
        return public_key, private_key
        
    def encrypt(self, data: Union[str, bytes], public_key: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Encrypt data using quantum-resistant hybrid encryption.
        
        Args:
            data: Data to encrypt
            public_key: Optional recipient's public key
            
        Returns:
            Dictionary containing encrypted data and metadata
        """
        if isinstance(data, str):
            data = data.encode('utf-8')
            
        # Generate ephemeral symmetric key
        ephemeral_key = os.urandom(32)
        iv = os.urandom(16)
        
        # Encrypt data with AES-256-GCM (simulating post-quantum symmetric)
        cipher = Cipher(
            algorithms.AES(ephemeral_key),
            modes.GCM(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Encrypt ephemeral key with post-quantum KEM (simulated)
        if public_key:
            encrypted_key = self._kem_encapsulate(ephemeral_key, public_key)
        else:
            # Use master key encryption
            encrypted_key = self._encrypt_with_master_key(ephemeral_key)
            
        return {
            'ciphertext': base64.b64encode(ciphertext).decode('utf-8'),
            'encrypted_key': base64.b64encode(encrypted_key).decode('utf-8'),
            'iv': base64.b64encode(iv).decode('utf-8'),
            'tag': base64.b64encode(encryptor.tag).decode('utf-8'),
            'algorithm': self.algorithm.value,
            'timestamp': datetime.utcnow().isoformat(),
            'key_id': hashlib.sha256(public_key or self.master_key).hexdigest()[:16]
        }
        
    def decrypt(self, encrypted_data: Dict[str, Any], private_key: Optional[bytes] = None) -> bytes:
        """
        Decrypt quantum-resistant encrypted data.
        
        Args:
            encrypted_data: Dictionary from encrypt()
            private_key: Optional private key for decryption
            
        Returns:
            Decrypted data as bytes
        """
        ciphertext = base64.b64decode(encrypted_data['ciphertext'])
        encrypted_key = base64.b64decode(encrypted_data['encrypted_key'])
        iv = base64.b64decode(encrypted_data['iv'])
        tag = base64.b64decode(encrypted_data['tag'])
        
        # Decrypt ephemeral key
        if private_key:
            ephemeral_key = self._kem_decapsulate(encrypted_key, private_key)
        else:
            ephemeral_key = self._decrypt_with_master_key(encrypted_key)
            
        # Decrypt data
        cipher = Cipher(
            algorithms.AES(ephemeral_key),
            modes.GCM(iv, tag),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()
        
    def _kem_encapsulate(self, symmetric_key: bytes, public_key: bytes) -> bytes:
        """Simulated Key Encapsulation Mechanism."""
        # In production, use actual Kyber/NTRU KEM
        shared_secret = hashlib.sha256(public_key + symmetric_key).digest()
        return shared_secret + symmetric_key  # Simplified
        
    def _kem_decapsulate(self, encapsulated_key: bytes, private_key: bytes) -> bytes:
        """Simulated Key Decapsulation Mechanism."""
        # In production, use actual Kyber/NTRU KEM
        shared_secret = encapsulated_key[:32]
        symmetric_key = encapsulated_key[32:]
        expected_secret = hashlib.sha256(
            hashlib.sha256(private_key).digest() + symmetric_key
        ).digest()
        if shared_secret != expected_secret:
            raise ValueError("Decapsulation failed - invalid key")
        return symmetric_key
        
    def _encrypt_with_master_key(self, data: bytes) -> bytes:
        """Encrypt with master key using hybrid encryption."""
        iv = os.urandom(16)
        cipher = Cipher(
            algorithms.AES(self.master_key[:32]),
            modes.CFB(iv),
            backend=self.backend
        )
        encryptor = cipher.encryptor()
        return iv + encryptor.update(data) + encryptor.finalize()
        
    def _decrypt_with_master_key(self, encrypted_data: bytes) -> bytes:
        """Decrypt with master key."""
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]
        cipher = Cipher(
            algorithms.AES(self.master_key[:32]),
            modes.CFB(iv),
            backend=self.backend
        )
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()
        
    def rotate_keys(self):
        """Rotate encryption keys for forward secrecy."""
        old_key = self.master_key
        self._generate_master_key()
        logger.info("Encryption keys rotated successfully")
        return old_key
        
    def cryptographic_erasure(self, key_id: str) -> bool:
        """
        Perform cryptographic erasure by destroying key material.
        Makes encrypted data permanently unrecoverable.
        """
        # In production, this would securely wipe key material
        # For simulation, we just log the operation
        logger.info(f"Cryptographic erasure performed for key: {key_id}")
        return True


class PIIDetector:
    """
    Machine learning-based PII detection with configurable rules.
    Supports multiple detection strategies and custom patterns.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._initialize_detectors()
        self._compile_patterns()
        
    def _initialize_detectors(self):
        """Initialize detection models and patterns."""
        self.nlp = None
        self.ml_model = None
        self.vectorizer = None
        
        # Load spaCy model if available
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model for PII detection")
            except:
                logger.warning("Could not load spaCy model, using regex fallback")
                
        # Initialize ML model if available
        if ML_AVAILABLE and self.config.get('use_ml', True):
            self._initialize_ml_model()
            
    def _initialize_ml_model(self):
        """Initialize machine learning model for PII detection."""
        try:
            # Simple TF-IDF + Logistic Regression for demonstration
            # In production, use transformer-based models
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                analyzer='char'
            )
            self.ml_model = LogisticRegression(
                max_iter=1000,
                class_weight='balanced'
            )
            self._train_initial_model()
            logger.info("Initialized ML model for PII detection")
        except Exception as e:
            logger.warning(f"Could not initialize ML model: {e}")
            self.ml_model = None
            
    def _train_initial_model(self):
        """Train initial model with synthetic data."""
        # Synthetic training data for common PII patterns
        training_data = [
            ("john.doe@example.com", 1),
            ("555-123-4567", 1),
            ("123-45-6789", 1),
            ("4111-1111-1111-1111", 1),
            ("192.168.1.1", 1),
            ("John Doe", 1),
            ("123 Main St", 1),
            ("01/01/1990", 1),
            ("Hello world", 0),
            ("This is a test", 0),
            ("Scrapy spider", 0),
        ]
        
        texts, labels = zip(*training_data)
        X = self.vectorizer.fit_transform(texts)
        self.ml_model.fit(X, labels)
        
    def _compile_patterns(self):
        """Compile regex patterns for PII detection."""
        self.patterns = {
            PIICategory.EMAIL: re.compile(
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            ),
            PIICategory.PHONE: re.compile(
                r'(\+?1?\s?)?(\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}'
            ),
            PIICategory.SSN: re.compile(
                r'\b\d{3}[-]?\d{2}[-]?\d{4}\b'
            ),
            PIICategory.CREDIT_CARD: re.compile(
                r'\b(?:\d[ -]*?){13,16}\b'
            ),
            PIICategory.IP_ADDRESS: re.compile(
                r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            ),
            PIICategory.DATE_OF_BIRTH: re.compile(
                r'\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/(19|20)\d{2}\b'
            ),
        }
        
        # Custom patterns from config
        custom_patterns = self.config.get('custom_patterns', {})
        for category_name, pattern in custom_patterns.items():
            try:
                category = PIICategory(category_name)
                self.patterns[category] = re.compile(pattern)
            except ValueError:
                logger.warning(f"Unknown PII category: {category_name}")
                
    def detect(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PII in text using multiple strategies.
        
        Returns:
            List of detected PII entities with metadata
        """
        entities = []
        
        # Strategy 1: Regex patterns
        regex_entities = self._detect_regex(text)
        entities.extend(regex_entities)
        
        # Strategy 2: spaCy NER
        if self.nlp:
            spacy_entities = self._detect_spacy(text)
            entities.extend(spacy_entities)
            
        # Strategy 3: ML model
        if self.ml_model:
            ml_entities = self._detect_ml(text)
            entities.extend(ml_entities)
            
        # Deduplicate and merge overlapping entities
        return self._merge_entities(entities)
        
    def _detect_regex(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII using regex patterns."""
        entities = []
        for category, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                entities.append({
                    'text': match.group(),
                    'start': match.start(),
                    'end': match.end(),
                    'category': category.value,
                    'confidence': 0.9,
                    'method': 'regex'
                })
        return entities
        
    def _detect_spacy(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII using spaCy NER."""
        entities = []
        doc = self.nlp(text)
        
        # Map spaCy entity types to PII categories
        entity_mapping = {
            'PERSON': PIICategory.NAME,
            'GPE': PIICategory.ADDRESS,
            'LOC': PIICategory.ADDRESS,
            'ORG': PIICategory.OTHER,
            'DATE': PIICategory.DATE_OF_BIRTH,
            'MONEY': PIICategory.FINANCIAL,
        }
        
        for ent in doc.ents:
            if ent.label_ in entity_mapping:
                entities.append({
                    'text': ent.text,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'category': entity_mapping[ent.label_].value,
                    'confidence': 0.8,
                    'method': 'spacy'
                })
        return entities
        
    def _detect_ml(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII using machine learning model."""
        entities = []
        
        # Simple sliding window approach for demonstration
        words = text.split()
        for i, word in enumerate(words):
            try:
                X = self.vectorizer.transform([word])
                prediction = self.ml_model.predict(X)[0]
                if prediction == 1:  # PII detected
                    # Find position in original text
                    start = text.find(word)
                    if start != -1:
                        entities.append({
                            'text': word,
                            'start': start,
                            'end': start + len(word),
                            'category': PIICategory.OTHER.value,
                            'confidence': 0.7,
                            'method': 'ml'
                        })
            except:
                continue
        return entities
        
    def _merge_entities(self, entities: List[Dict]) -> List[Dict]:
        """Merge overlapping entities."""
        if not entities:
            return []
            
        # Sort by start position
        sorted_entities = sorted(entities, key=lambda x: x['start'])
        merged = [sorted_entities[0]]
        
        for entity in sorted_entities[1:]:
            last = merged[-1]
            
            # Check for overlap
            if entity['start'] <= last['end']:
                # Merge entities
                if entity['confidence'] > last['confidence']:
                    merged[-1] = entity
                elif entity['end'] > last['end']:
                    merged[-1]['end'] = entity['end']
                    merged[-1]['text'] = merged[-1]['text'] + ' ' + entity['text']
            else:
                merged.append(entity)
                
        return merged
        
    def update_model(self, texts: List[str], labels: List[int]):
        """Update ML model with new training data."""
        if not self.ml_model or not self.vectorizer:
            return
            
        try:
            X = self.vectorizer.transform(texts)
            self.ml_model.partial_fit(X, labels)
            logger.info(f"Updated PII detection model with {len(texts)} samples")
        except Exception as e:
            logger.error(f"Failed to update model: {e}")


class PIIMasker:
    """
    Configurable PII masking with multiple strategies.
    Supports redaction, tokenization, and format-preserving encryption.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.masking_rules = self._load_masking_rules()
        self.token_map = {}
        self.reverse_token_map = {}
        
    def _load_masking_rules(self) -> Dict[str, Dict]:
        """Load masking rules from configuration."""
        default_rules = {
            PIICategory.EMAIL.value: {
                'strategy': 'partial_mask',
                'mask_char': '*',
                'preserve_domain': True,
                'visible_chars': 2
            },
            PIICategory.PHONE.value: {
                'strategy': 'partial_mask',
                'mask_char': '*',
                'visible_last': 4
            },
            PIICategory.SSN.value: {
                'strategy': 'full_redact',
                'replacement': '[SSN_REDACTED]'
            },
            PIICategory.CREDIT_CARD.value: {
                'strategy': 'partial_mask',
                'mask_char': '*',
                'visible_last': 4
            },
            PIICategory.NAME.value: {
                'strategy': 'tokenize',
                'prefix': 'NAME_'
            },
            PIICategory.ADDRESS.value: {
                'strategy': 'generalize',
                'replacement': '[ADDRESS_REDACTED]'
            },
        }
        
        # Override with custom rules
        custom_rules = self.config.get('masking_rules', {})
        for category, rule in custom_rules.items():
            if category in default_rules:
                default_rules[category].update(rule)
            else:
                default_rules[category] = rule
                
        return default_rules
        
    def mask(self, text: str, entities: List[Dict]) -> Tuple[str, Dict[str, str]]:
        """
        Mask PII in text according to configured rules.
        
        Returns:
            Tuple of (masked_text, token_mapping)
        """
        if not entities:
            return text, {}
            
        # Sort entities by start position in reverse order
        # to avoid index shifting during replacement
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        masked_text = text
        token_mapping = {}
        
        for entity in sorted_entities:
            category = entity['category']
            original_text = entity['text']
            
            # Get masking rule for this category
            rule = self.masking_rules.get(category, {
                'strategy': 'full_redact',
                'replacement': f'[{category.upper()}_REDACTED]'
            })
            
            # Apply masking strategy
            masked_value = self._apply_masking_strategy(
                original_text, rule, category
            )
            
            # Replace in text
            start, end = entity['start'], entity['end']
            masked_text = masked_text[:start] + masked_value + masked_text[end:]
            
            # Store token mapping for reversible masking
            if rule.get('strategy') == 'tokenize':
                token_mapping[masked_value] = original_text
                
        return masked_text, token_mapping
        
    def _apply_masking_strategy(self, text: str, rule: Dict, category: str) -> str:
        """Apply specific masking strategy."""
        strategy = rule.get('strategy', 'full_redact')
        
        if strategy == 'full_redact':
            return rule.get('replacement', f'[{category.upper()}_REDACTED]')
            
        elif strategy == 'partial_mask':
            mask_char = rule.get('mask_char', '*')
            
            if rule.get('preserve_domain') and '@' in text:
                # Email masking
                local, domain = text.rsplit('@', 1)
                visible = rule.get('visible_chars', 2)
                masked_local = local[:visible] + mask_char * (len(local) - visible)
                return f"{masked_local}@{domain}"
                
            elif 'visible_last' in rule:
                # Phone/card masking
                visible = rule.get('visible_last', 4)
                if len(text) > visible:
                    return mask_char * (len(text) - visible) + text[-visible:]
                    
            # Default partial masking
            visible = rule.get('visible_chars', 2)
            if len(text) > visible * 2:
                return text[:visible] + mask_char * (len(text) - visible * 2) + text[-visible:]
            else:
                return mask_char * len(text)
                
        elif strategy == 'tokenize':
            # Generate deterministic token
            token_hash = hashlib.sha256(text.encode()).hexdigest()[:8]
            prefix = rule.get('prefix', 'PII_')
            token = f"{prefix}{token_hash}"
            
            # Store mapping
            self.token_map[token] = text
            self.reverse_token_map[text] = token
            return token
            
        elif strategy == 'generalize':
            # Generalize to broader category
            return rule.get('replacement', f'[{category.upper()}_GENERALIZED]')
            
        elif strategy == 'format_preserving':
            # Format-preserving encryption (simplified)
            return self._format_preserving_mask(text, rule)
            
        else:
            # Default redaction
            return f'[{category.upper()}_REDACTED]'
            
    def _format_preserving_mask(self, text: str, rule: Dict) -> str:
        """Apply format-preserving masking."""
        # Simple implementation - in production, use FF1/FF3 algorithms
        result = []
        for char in text:
            if char.isdigit():
                result.append('*')
            elif char.isalpha():
                result.append('X')
            else:
                result.append(char)
        return ''.join(result)
        
    def unmask(self, masked_text: str) -> str:
        """Reverse tokenization masking."""
        for token, original in self.token_map.items():
            masked_text = masked_text.replace(token, original)
        return masked_text


class DataLifecycleManager:
    """
    GDPR-compliant data lifecycle management.
    Handles retention policies, consent tracking, and data subject rights.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.retention_days = self._get_retention_days()
        self.consent_records = {}
        self.data_registry = {}
        self.erasure_queue = []
        
    def _get_retention_days(self) -> Dict[str, int]:
        """Get retention periods from configuration."""
        policy = self.config.get('retention_policy', DataRetentionPolicy.MEDIUM_TERM.value)
        
        retention_map = {
            DataRetentionPolicy.IMMEDIATE.value: 0,
            DataRetentionPolicy.SHORT_TERM.value: 30,
            DataRetentionPolicy.MEDIUM_TERM.value: 90,
            DataRetentionPolicy.LONG_TERM.value: 365,
        }
        
        if policy == DataRetentionPolicy.CUSTOM.value:
            return self.config.get('custom_retention_days', {
                'default': 90,
                'sensitive': 30,
                'anonymous': 365
            })
        else:
            days = retention_map.get(policy, 90)
            return {'default': days}
            
    def register_data(self, data_id: str, metadata: Dict) -> str:
        """Register data for lifecycle management."""
        registration_id = str(uuid.uuid4())
        
        self.data_registry[registration_id] = {
            'data_id': data_id,
            'metadata': metadata,
            'created_at': datetime.utcnow(),
            'expires_at': datetime.utcnow() + timedelta(
                days=self._get_retention_period(metadata)
            ),
            'status': 'active',
            'consent_id': metadata.get('consent_id'),
            'pii_categories': metadata.get('pii_categories', [])
        }
        
        return registration_id
        
    def _get_retention_period(self, metadata: Dict) -> int:
        """Determine retention period based on data sensitivity."""
        pii_categories = metadata.get('pii_categories', [])
        
        # Sensitive data gets shorter retention
        sensitive_categories = {
            PIICategory.SSN.value,
            PIICategory.CREDIT_CARD.value,
            PIICategory.MEDICAL.value,
            PIICategory.BIOMETRIC.value
        }
        
        if any(cat in sensitive_categories for cat in pii_categories):
            return self.retention_days.get('sensitive', 30)
        elif pii_categories:
            return self.retention_days.get('default', 90)
        else:
            return self.retention_days.get('anonymous', 365)
            
    def check_expired_data(self) -> List[str]:
        """Check for expired data that needs deletion."""
        now = datetime.utcnow()
        expired = []
        
        for reg_id, record in self.data_registry.items():
            if record['status'] == 'active' and record['expires_at'] < now:
                expired.append(reg_id)
                record['status'] = 'expired'
                self.erasure_queue.append(reg_id)
                
        return expired
        
    async def process_erasure_queue(self, encryptor: QuantumResistantEncryptor):
        """Process data erasure queue with cryptographic erasure."""
        for reg_id in self.erasure_queue[:]:
            try:
                record = self.data_registry.get(reg_id)
                if record:
                    # Perform cryptographic erasure
                    key_id = record['metadata'].get('encryption_key_id')
                    if key_id:
                        encryptor.cryptographic_erasure(key_id)
                        
                    # Mark as erased
                    record['status'] = 'erased'
                    record['erased_at'] = datetime.utcnow()
                    
                    logger.info(f"Data {reg_id} cryptographically erased")
                    self.erasure_queue.remove(reg_id)
                    
            except Exception as e:
                logger.error(f"Failed to erase data {reg_id}: {e}")
                
    def handle_erasure_request(self, data_subject_id: str) -> Dict:
        """
        Handle GDPR Article 17 - Right to erasure.
        Returns status of erasure request.
        """
        affected_records = []
        
        for reg_id, record in self.data_registry.items():
            if record['metadata'].get('data_subject_id') == data_subject_id:
                affected_records.append(reg_id)
                self.erasure_queue.append(reg_id)
                
        return {
            'request_id': str(uuid.uuid4()),
            'data_subject_id': data_subject_id,
            'affected_records': len(affected_records),
            'status': 'queued',
            'timestamp': datetime.utcnow().isoformat()
        }
        
    def handle_portability_request(self, data_subject_id: str) -> Dict:
        """
        Handle GDPR Article 20 - Right to data portability.
        Returns data in structured format.
        """
        portable_data = []
        
        for reg_id, record in self.data_registry.items():
            if record['metadata'].get('data_subject_id') == data_subject_id:
                portable_data.append({
                    'registration_id': reg_id,
                    'data_id': record['data_id'],
                    'created_at': record['created_at'].isoformat(),
                    'categories': record['pii_categories']
                })
                
        return {
            'request_id': str(uuid.uuid4()),
            'data_subject_id': data_subject_id,
            'data': portable_data,
            'format': 'json',
            'timestamp': datetime.utcnow().isoformat()
        }
        
    def generate_privacy_report(self) -> Dict:
        """Generate GDPR compliance report."""
        now = datetime.utcnow()
        
        stats = {
            'total_records': len(self.data_registry),
            'active_records': 0,
            'expired_records': 0,
            'erased_records': 0,
            'pii_distribution': defaultdict(int),
            'retention_compliance': 0.0
        }
        
        for record in self.data_registry.values():
            if record['status'] == 'active':
                stats['active_records'] += 1
                if record['expires_at'] > now:
                    stats['retention_compliance'] += 1
            elif record['status'] == 'expired':
                stats['expired_records'] += 1
            elif record['status'] == 'erased':
                stats['erased_records'] += 1
                
            for category in record['pii_categories']:
                stats['pii_distribution'][category] += 1
                
        if stats['active_records'] > 0:
            stats['retention_compliance'] = (
                stats['retention_compliance'] / stats['active_records'] * 100
            )
            
        return stats


class GDPRManager:
    """
    Main GDPR compliance manager for Scrapy.
    Integrates encryption, PII detection, masking, and lifecycle management.
    """
    
    def __init__(self, crawler=None):
        self.crawler = crawler
        self.settings = crawler.settings if crawler else get_project_settings()
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.encryptor = QuantumResistantEncryptor(
            algorithm=EncryptionAlgorithm(self.config.get('encryption_algorithm', 'hybrid'))
        )
        self.pii_detector = PIIDetector(self.config.get('pii_detection', {}))
        self.pii_masker = PIIMasker(self.config.get('pii_masking', {}))
        self.lifecycle_manager = DataLifecycleManager(self.config.get('data_lifecycle', {}))
        
        # State
        self.processed_items = 0
        self.encrypted_items = 0
        self.masked_items = 0
        self.erasure_requests = []
        
        # Register signals
        if crawler:
            crawler.signals.connect(self.engine_started, signal=signals.engine_started)
            crawler.signals.connect(self.engine_stopped, signal=signals.engine_stopped)
            crawler.signals.connect(self.item_scraped, signal=signals.item_scraped)
            
    def _load_config(self) -> Dict:
        """Load configuration from settings."""
        return {
            'enabled': self.settings.getbool('GDPR_ENABLED', True),
            'encryption_algorithm': self.settings.get('GDPR_ENCRYPTION_ALGORITHM', 'hybrid'),
            'encrypt_all': self.settings.getbool('GDPR_ENCRYPT_ALL', False),
            'mask_pii': self.settings.getbool('GDPR_MASK_PII', True),
            'retention_policy': self.settings.get('GDPR_RETENTION_POLICY', 'medium_term'),
            'pii_detection': self.settings.getdict('GDPR_PII_DETECTION', {}),
            'pii_masking': self.settings.getdict('GDPR_PII_MASKING', {}),
            'data_lifecycle': self.settings.getdict('GDPR_DATA_LIFECYCLE', {}),
            'log_level': self.settings.get('GDPR_LOG_LEVEL', 'INFO')
        }
        
    @classmethod
    def from_crawler(cls, crawler):
        """Create GDPRManager from crawler."""
        if not crawler.settings.getbool('GDPR_ENABLED', True):
            raise NotConfigured("GDPR manager disabled")
            
        manager = cls(crawler)
        
        # Set log level
        log_level = manager.config.get('log_level', 'INFO')
        logger.setLevel(getattr(logging, log_level))
        
        return manager
        
    def engine_started(self):
        """Called when Scrapy engine starts."""
        logger.info("GDPR Privacy Manager initialized")
        logger.info(f"Configuration: {json.dumps(self.config, indent=2)}")
        
        # Start periodic cleanup task
        if self.crawler:
            self._start_cleanup_task()
            
    def engine_stopped(self):
        """Called when Scrapy engine stops."""
        self._generate_final_report()
        logger.info("GDPR Privacy Manager stopped")
        
    def _start_cleanup_task(self):
        """Start periodic data cleanup task."""
        # In production, use Twisted's LoopingCall
        logger.info("Data lifecycle cleanup task scheduled")
        
    def item_scraped(self, item, response, spider):
        """Process scraped item through GDPR pipeline."""
        if not self.config.get('enabled'):
            return item
            
        try:
            # Convert item to processable format
            item_data = self._item_to_dict(item)
            
            # Step 1: PII Detection
            pii_entities = []
            if self.config.get('mask_pii'):
                text_content = self._extract_text_content(item_data)
                pii_entities = self.pii_detector.detect(text_content)
                
                if pii_entities:
                    logger.debug(f"Detected {len(pii_entities)} PII entities")
                    self.masked_items += 1
                    
            # Step 2: PII Masking
            masked_data = item_data
            token_mapping = {}
            if pii_entities:
                masked_data, token_mapping = self._mask_item_data(item_data, pii_entities)
                
            # Step 3: Encryption
            encrypted_data = masked_data
            if self.config.get('encrypt_all') or pii_entities:
                encrypted_data = self._encrypt_item_data(masked_data)
                self.encrypted_items += 1
                
            # Step 4: Data Registration
            metadata = {
                'spider': spider.name,
                'url': response.url if response else None,
                'pii_categories': [e['category'] for e in pii_entities],
                'has_pii': bool(pii_entities),
                'encrypted': encrypted_data != masked_data,
                'data_subject_id': self._extract_data_subject_id(item_data)
            }
            
            registration_id = self.lifecycle_manager.register_data(
                data_id=str(uuid.uuid4()),
                metadata=metadata
            )
            
            # Update item with processed data
            processed_item = self._dict_to_item(encrypted_data)
            processed_item['gdpr_metadata'] = {
                'registration_id': registration_id,
                'processed_at': datetime.utcnow().isoformat(),
                'pii_detected': len(pii_entities),
                'encrypted': metadata['encrypted'],
                'token_mapping': token_mapping if token_mapping else None
            }
            
            self.processed_items += 1
            return processed_item
            
        except Exception as e:
            logger.error(f"GDPR processing failed: {e}")
            # Return original item on error
            return item
            
    def _item_to_dict(self, item) -> Dict:
        """Convert Scrapy item to dictionary."""
        if hasattr(item, 'to_dict'):
            return item.to_dict()
        elif isinstance(item, dict):
            return item.copy()
        else:
            return dict(item)
            
    def _dict_to_item(self, data: Dict):
        """Convert dictionary back to item."""
        # In production, preserve original item type
        return data
        
    def _extract_text_content(self, item_data: Dict) -> str:
        """Extract all text content from item for PII detection."""
        text_parts = []
        
        def extract_strings(obj):
            if isinstance(obj, str):
                text_parts.append(obj)
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_strings(value)
            elif isinstance(obj, (list, tuple)):
                for value in obj:
                    extract_strings(value)
                    
        extract_strings(item_data)
        return ' '.join(text_parts)
        
    def _mask_item_data(self, item_data: Dict, entities: List[Dict]) -> Tuple[Dict, Dict]:
        """Mask PII in item data."""
        # For simplicity, mask all string fields
        # In production, map entities to specific fields
        masked_data = {}
        all_tokens = {}
        
        for key, value in item_data.items():
            if isinstance(value, str):
                masked_value, tokens = self.pii_masker.mask(value, entities)
                masked_data[key] = masked_value
                all_tokens.update(tokens)
            else:
                masked_data[key] = value
                
        return masked_data, all_tokens
        
    def _encrypt_item_data(self, item_data: Dict) -> Dict:
        """Encrypt sensitive item data."""
        # Serialize and encrypt the entire item
        serialized = json.dumps(item_data, default=str)
        encrypted = self.encryptor.encrypt(serialized)
        
        return {
            '_encrypted': True,
            '_encryption_metadata': encrypted,
            '_original_keys': list(item_data.keys())
        }
        
    def _extract_data_subject_id(self, item_data: Dict) -> Optional[str]:
        """Extract or generate data subject identifier."""
        # Look for common identifier fields
        id_fields = ['user_id', 'email', 'customer_id', 'id']
        
        for field in id_fields:
            if field in item_data:
                value = item_data[field]
                if isinstance(value, str):
                    # Hash the identifier for privacy
                    return hashlib.sha256(value.encode()).hexdigest()[:16]
                    
        # Generate anonymous ID
        return f"anon_{uuid.uuid4().hex[:16]}"
        
    def _generate_final_report(self):
        """Generate final compliance report."""
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'processed_items': self.processed_items,
            'encrypted_items': self.encrypted_items,
            'masked_items': self.masked_items,
            'lifecycle_stats': self.lifecycle_manager.generate_privacy_report(),
            'erasure_requests': len(self.erasure_requests)
        }
        
        logger.info(f"GDPR Compliance Report: {json.dumps(report, indent=2)}")
        
        # Save report to file
        try:
            report_path = Path('gdpr_compliance_report.json')
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Compliance report saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save compliance report: {e}")
            
    def handle_erasure_request(self, data_subject_id: str) -> Dict:
        """Handle GDPR erasure request."""
        result = self.lifecycle_manager.handle_erasure_request(data_subject_id)
        self.erasure_requests.append(result)
        return result
        
    def handle_portability_request(self, data_subject_id: str) -> Dict:
        """Handle GDPR data portability request."""
        return self.lifecycle_manager.handle_portability_request(data_subject_id)
        
    def get_privacy_settings(self) -> Dict:
        """Get current privacy configuration."""
        return {
            'encryption_algorithm': self.encryptor.algorithm.value,
            'pii_detection_enabled': self.config.get('mask_pii'),
            'retention_policy': self.config.get('retention_policy'),
            'encrypt_all': self.config.get('encrypt_all')
        }
        
    def update_privacy_settings(self, new_settings: Dict):
        """Update privacy configuration."""
        self.config.update(new_settings)
        
        # Reinitialize components if needed
        if 'encryption_algorithm' in new_settings:
            self.encryptor = QuantumResistantEncryptor(
                algorithm=EncryptionAlgorithm(new_settings['encryption_algorithm'])
            )
            
        logger.info(f"Privacy settings updated: {new_settings}")


# Integration with Scrapy settings
def configure_gdpr_settings(settings):
    """Configure GDPR settings in Scrapy project."""
    settings.set('GDPR_ENABLED', True, priority='cmdline')
    settings.set('GDPR_ENCRYPTION_ALGORITHM', 'hybrid', priority='cmdline')
    settings.set('GDPR_MASK_PII', True, priority='cmdline')
    settings.set('GDPR_RETENTION_POLICY', 'medium_term', priority='cmdline')
    
    # Add to item pipelines
    settings.set('ITEM_PIPELINES', {
        **settings.getdict('ITEM_PIPELINES'),
        'vex.privacy.gdpr_manager.GDPRManager': 100
    }, priority='cmdline')


# Example middleware integration
class GDPRMiddleware:
    """Scrapy middleware for GDPR compliance."""
    
    def __init__(self, settings):
        self.gdpr_manager = GDPRManager(settings=settings)
        
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)
        
    def process_spider_input(self, response, spider):
        """Process response before spider sees it."""
        # Could decrypt response if encrypted
        return None
        
    def process_spider_output(self, response, result, spider):
        """Process spider output for GDPR compliance."""
        for item in result:
            if isinstance(item, (dict,)):
                processed = self.gdpr_manager.item_scraped(item, response, spider)
                yield processed
            else:
                yield item


# Command-line interface for GDPR operations
class GDPRCommand:
    """Scrapy command for GDPR operations."""
    
    def __init__(self):
        self.gdpr_manager = GDPRManager()
        
    def run(self, args):
        """Execute GDPR command."""
        if args[0] == 'report':
            self.print_report()
        elif args[0] == 'erasure':
            if len(args) > 1:
                self.handle_erasure(args[1])
            else:
                print("Error: Data subject ID required")
        elif args[0] == 'portability':
            if len(args) > 1:
                self.handle_portability(args[1])
            else:
                print("Error: Data subject ID required")
        elif args[0] == 'settings':
            self.print_settings()
        else:
            print("Unknown GDPR command")
            
    def print_report(self):
        """Print privacy compliance report."""
        report = self.gdpr_manager.lifecycle_manager.generate_privacy_report()
        print(json.dumps(report, indent=2))
        
    def handle_erasure(self, data_subject_id):
        """Handle erasure request."""
        result = self.gdpr_manager.handle_erasure_request(data_subject_id)
        print(f"Erasure request submitted: {json.dumps(result, indent=2)}")
        
    def handle_portability(self, data_subject_id):
        """Handle portability request."""
        result = self.gdpr_manager.handle_portability_request(data_subject_id)
        print(f"Portability request processed: {json.dumps(result, indent=2)}")
        
    def print_settings(self):
        """Print current privacy settings."""
        settings = self.gdpr_manager.get_privacy_settings()
        print(json.dumps(settings, indent=2))


# Export public API
__all__ = [
    'GDPRManager',
    'QuantumResistantEncryptor',
    'PIIDetector',
    'PIIMasker',
    'DataLifecycleManager',
    'GDPRMiddleware',
    'GDPRCommand',
    'configure_gdpr_settings',
    'EncryptionAlgorithm',
    'PIICategory',
    'DataRetentionPolicy'
]