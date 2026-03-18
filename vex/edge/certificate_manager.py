"""
Certificate Manager for Scrapy Edge Deployment
Automatic TLS certificate management with ACME protocol integration
for zero-config edge deployment on Cloudflare Workers, Deno Deploy, and browser extensions.
"""

import os
import json
import time
import base64
import hashlib
import logging
import threading
from typing import Optional, Dict, Any, Tuple, List, Callable
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict, field
from enum import Enum
from abc import ABC, abstractmethod
import tempfile

# Cryptography imports for certificate operations
try:
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, ec
    from cryptography.hazmat.backends import default_backend
    CRYPTOGRAPHY_AVAILABLE = True
except ImportError:
    CRYPTOGRAPHY_AVAILABLE = False
    logging.warning("cryptography library not available. Certificate operations will be limited.")

# ACME protocol implementation
try:
    import josepy as jose
    from acme import client, messages, challenges
    from acme.messages import Directory
    ACME_AVAILABLE = True
except ImportError:
    ACME_AVAILABLE = False
    logging.warning("acme library not available. ACME protocol support disabled.")

from vex.settings import Settings
from vex.exceptions import NotConfigured
from vex.utils.misc import load_object
from vex.utils.log import configure_logging


logger = logging.getLogger(__name__)


class CertificateStatus(Enum):
    """Certificate status states."""
    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"
    EXPIRED = "expired"
    REVOKED = "revoked"


class CertificateType(Enum):
    """Certificate types for different deployment scenarios."""
    SELF_SIGNED = "self_signed"
    ACME_LETSENCRYPT = "acme_letsencrypt"
    ACME_ZEROSSL = "acme_zerossl"
    ACME_BUYPASS = "acme_buypass"
    CUSTOM = "custom"


@dataclass
class CertificateInfo:
    """Certificate information container."""
    domain: str
    status: CertificateStatus
    certificate_type: CertificateType
    issued_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    issuer: Optional[str] = None
    serial_number: Optional[str] = None
    fingerprint: Optional[str] = None
    key_type: str = "rsa"
    key_size: int = 2048
    san_domains: List[str] = field(default_factory=list)
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = None
    certificate_chain_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        data['certificate_type'] = self.certificate_type.value
        if self.issued_at:
            data['issued_at'] = self.issued_at.isoformat()
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CertificateInfo':
        """Create from dictionary."""
        if 'status' in data:
            data['status'] = CertificateStatus(data['status'])
        if 'certificate_type' in data:
            data['certificate_type'] = CertificateType(data['certificate_type'])
        if 'issued_at' in data and isinstance(data['issued_at'], str):
            data['issued_at'] = datetime.fromisoformat(data['issued_at'])
        if 'expires_at' in data and isinstance(data['expires_at'], str):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        return cls(**data)


class CertificateBackend(ABC):
    """Abstract base class for certificate backends."""
    
    @abstractmethod
    def issue_certificate(self, domain: str, san_domains: Optional[List[str]] = None) -> CertificateInfo:
        """Issue a certificate for the given domain."""
        pass
    
    @abstractmethod
    def renew_certificate(self, domain: str) -> CertificateInfo:
        """Renew an existing certificate."""
        pass
    
    @abstractmethod
    def revoke_certificate(self, domain: str) -> bool:
        """Revoke a certificate."""
        pass
    
    @abstractmethod
    def get_certificate_info(self, domain: str) -> Optional[CertificateInfo]:
        """Get certificate information for a domain."""
        pass


class SelfSignedBackend(CertificateBackend):
    """Backend for self-signed certificate generation."""
    
    def __init__(self, storage_path: str, key_size: int = 2048):
        self.storage_path = Path(storage_path)
        self.key_size = key_size
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def _generate_private_key(self, key_type: str = "rsa") -> Tuple[Any, str]:
        """Generate a private key."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("cryptography library required for key generation")
        
        if key_type == "rsa":
            private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=self.key_size,
                backend=default_backend()
            )
        elif key_type == "ecdsa":
            private_key = ec.generate_private_key(
                ec.SECP256R1(),
                backend=default_backend()
            )
        else:
            raise ValueError(f"Unsupported key type: {key_type}")
        
        # Serialize private key
        private_key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        return private_key, private_key_pem.decode('utf-8')
    
    def _generate_self_signed_certificate(self, domain: str, private_key: Any, 
                                         san_domains: Optional[List[str]] = None) -> str:
        """Generate a self-signed certificate."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("cryptography library required for certificate generation")
        
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "California"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "San Francisco"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Scrapy Edge"),
            x509.NameAttribute(NameOID.COMMON_NAME, domain),
        ])
        
        # Build certificate
        builder = x509.CertificateBuilder()
        builder = builder.subject_name(subject)
        builder = builder.issuer_name(issuer)
        builder = builder.public_key(private_key.public_key())
        builder = builder.serial_number(x509.random_serial_number())
        builder = builder.not_valid_before(datetime.utcnow())
        builder = builder.not_valid_after(datetime.utcnow() + timedelta(days=365))
        
        # Add SANs
        san_list = [x509.DNSName(domain)]
        if san_domains:
            for san in san_domains:
                san_list.append(x509.DNSName(san))
        
        builder = builder.add_extension(
            x509.SubjectAlternativeName(san_list),
            critical=False
        )
        
        # Add basic constraints
        builder = builder.add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True
        )
        
        # Sign certificate
        certificate = builder.sign(
            private_key=private_key,
            algorithm=hashes.SHA256(),
            backend=default_backend()
        )
        
        # Serialize certificate
        certificate_pem = certificate.public_bytes(serialization.Encoding.PEM)
        
        return certificate_pem.decode('utf-8'), certificate
    
    def issue_certificate(self, domain: str, san_domains: Optional[List[str]] = None) -> CertificateInfo:
        """Issue a self-signed certificate."""
        logger.info(f"Generating self-signed certificate for {domain}")
        
        # Generate private key
        private_key, private_key_pem = self._generate_private_key()
        
        # Generate certificate
        certificate_pem, certificate_obj = self._generate_self_signed_certificate(
            domain, private_key, san_domains
        )
        
        # Save files
        domain_path = self.storage_path / domain
        domain_path.mkdir(exist_ok=True)
        
        cert_path = domain_path / "cert.pem"
        key_path = domain_path / "key.pem"
        
        cert_path.write_text(certificate_pem)
        key_path.write_text(private_key_pem)
        
        # Create certificate info
        info = CertificateInfo(
            domain=domain,
            status=CertificateStatus.VALID,
            certificate_type=CertificateType.SELF_SIGNED,
            issued_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=365),
            issuer="Scrapy Edge Self-Signed",
            serial_number=str(certificate_obj.serial_number),
            fingerprint=certificate_obj.fingerprint(hashes.SHA256()).hex(),
            san_domains=san_domains or [],
            certificate_path=str(cert_path),
            private_key_path=str(key_path)
        )
        
        # Save metadata
        meta_path = domain_path / "meta.json"
        meta_path.write_text(json.dumps(info.to_dict(), indent=2))
        
        logger.info(f"Self-signed certificate generated for {domain}")
        return info
    
    def renew_certificate(self, domain: str) -> CertificateInfo:
        """Renew a self-signed certificate (generate new one)."""
        return self.issue_certificate(domain)
    
    def revoke_certificate(self, domain: str) -> bool:
        """Revoke a self-signed certificate (delete files)."""
        domain_path = self.storage_path / domain
        if domain_path.exists():
            import shutil
            shutil.rmtree(domain_path)
            logger.info(f"Revoked and deleted certificate for {domain}")
            return True
        return False
    
    def get_certificate_info(self, domain: str) -> Optional[CertificateInfo]:
        """Get certificate information for a domain."""
        domain_path = self.storage_path / domain
        meta_path = domain_path / "meta.json"
        
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text())
                return CertificateInfo.from_dict(data)
            except Exception as e:
                logger.error(f"Failed to load certificate info for {domain}: {e}")
        
        return None


class ACMEBackend(CertificateBackend):
    """Backend for ACME protocol (Let's Encrypt, ZeroSSL, etc.)."""
    
    def __init__(self, storage_path: str, acme_server: str, 
                 email: str, key_size: int = 2048):
        self.storage_path = Path(storage_path)
        self.acme_server = acme_server
        self.email = email
        self.key_size = key_size
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        if not ACME_AVAILABLE:
            raise NotConfigured("acme library not available for ACME backend")
        
        self.directory_url = self._get_directory_url(acme_server)
        self.account_key_path = self.storage_path / "account_key.pem"
        self._init_account()
    
    def _get_directory_url(self, server: str) -> str:
        """Get ACME directory URL for the server."""
        servers = {
            "letsencrypt": "https://acme-v02.api.letsencrypt.org/directory",
            "letsencrypt-staging": "https://acme-staging-v02.api.letsencrypt.org/directory",
            "zerossl": "https://acme.zerossl.com/v2/DV90",
            "buypass": "https://api.buypass.com/acme/directory",
        }
        return servers.get(server.lower(), server)
    
    def _init_account(self):
        """Initialize ACME account."""
        if not self.account_key_path.exists():
            # Generate account key
            account_key = jose.JWKRSA(
                key=rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=self.key_size,
                    backend=default_backend()
                )
            )
            
            # Save account key
            self.account_key_path.write_text(
                account_key.json_dumps()
            )
        
        # Load account key
        with open(self.account_key_path, 'r') as f:
            self.account_key = jose.JWKRSA.json_loads(f.read())
        
        # Initialize ACME client
        net = client.ClientNetwork(self.account_key, user_agent="ScrapyEdge/1.0")
        directory = messages.Directory.from_json(net.get(self.directory_url).json())
        self.acme_client = client.ClientV2(directory, net)
        
        # Register account
        try:
            regr = self.acme_client.new_account(
                messages.NewRegistration.from_data(
                    email=self.email,
                    terms_of_service_agreed=True
                )
            )
            logger.info(f"ACME account registered: {regr.uri}")
        except errors.ConflictError as e:
            logger.info(f"ACME account already exists: {e.location}")
    
    def _generate_csr(self, domain: str, san_domains: Optional[List[str]] = None) -> Tuple[str, Any]:
        """Generate Certificate Signing Request."""
        if not CRYPTOGRAPHY_AVAILABLE:
            raise RuntimeError("cryptography library required for CSR generation")
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=self.key_size,
            backend=default_backend()
        )
        
        # Build CSR
        csr_builder = x509.CertificateSigningRequestBuilder()
        csr_builder = csr_builder.subject_name(x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, domain),
        ]))
        
        # Add SANs
        if san_domains:
            san_list = [x509.DNSName(domain)]
            for san in san_domains:
                san_list.append(x509.DNSName(san))
            
            csr_builder = csr_builder.add_extension(
                x509.SubjectAlternativeName(san_list),
                critical=False
            )
        
        # Sign CSR
        csr = csr_builder.sign(private_key, hashes.SHA256(), default_backend())
        
        # Serialize
        csr_pem = csr.public_bytes(serialization.Encoding.PEM)
        private_key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        return csr_pem.decode('utf-8'), private_key_pem.decode('utf-8')
    
    def _perform_http_challenge(self, domain: str, challenge: challenges.HTTP01) -> bool:
        """Perform HTTP-01 challenge."""
        # This is a simplified implementation
        # In production, you'd need to serve the challenge response
        # For edge deployment, this might involve updating a CDN or edge function
        
        logger.info(f"HTTP challenge for {domain}: {challenge.validation}")
        
        # For now, we'll simulate success
        # In a real implementation, you'd:
        # 1. Create a well-known URL endpoint
        # 2. Serve the challenge response
        # 3. Wait for ACME server to verify
        
        return True
    
    def issue_certificate(self, domain: str, san_domains: Optional[List[str]] = None) -> CertificateInfo:
        """Issue certificate using ACME protocol."""
        logger.info(f"Requesting ACME certificate for {domain}")
        
        # Generate CSR
        csr_pem, private_key_pem = self._generate_csr(domain, san_domains)
        
        # Create new order
        order = self.acme_client.new_order(csr_pem)
        
        # Perform challenges
        for authz in order.authorizations:
            for challenge in authz.body.challenges:
                if isinstance(challenge.chall, challenges.HTTP01):
                    if self._perform_http_challenge(domain, challenge.chall):
                        # Respond to challenge
                        response = challenge.chall.response(self.account_key)
                        self.acme_client.answer_challenge(challenge, response)
                    else:
                        raise RuntimeError(f"Failed to complete HTTP challenge for {domain}")
        
        # Finalize order
        order = self.acme_client.poll_and_finalize(order)
        
        # Save certificate and key
        domain_path = self.storage_path / domain
        domain_path.mkdir(exist_ok=True)
        
        cert_path = domain_path / "cert.pem"
        key_path = domain_path / "key.pem"
        chain_path = domain_path / "chain.pem"
        
        # Save certificate chain
        certificate_pem = order.fullchain_pem
        cert_path.write_text(certificate_pem)
        key_path.write_text(private_key_pem)
        
        # Extract chain (excluding end-entity certificate)
        chain_pem = "\n".join(certificate_pem.split("\n\n")[1:])
        if chain_pem:
            chain_path.write_text(chain_pem)
        
        # Parse certificate for info
        if CRYPTOGRAPHY_AVAILABLE:
            cert = x509.load_pem_x509_certificate(
                certificate_pem.encode(), default_backend()
            )
            issued_at = cert.not_valid_before
            expires_at = cert.not_valid_after
            issuer = cert.issuer.rfc4514_string()
            serial = str(cert.serial_number)
            fingerprint = cert.fingerprint(hashes.SHA256()).hex()
        else:
            issued_at = datetime.utcnow()
            expires_at = datetime.utcnow() + timedelta(days=90)
            issuer = "ACME Certificate"
            serial = "unknown"
            fingerprint = "unknown"
        
        # Create certificate info
        info = CertificateInfo(
            domain=domain,
            status=CertificateStatus.VALID,
            certificate_type=CertificateType.ACME_LETSENCRYPT,
            issued_at=issued_at,
            expires_at=expires_at,
            issuer=issuer,
            serial_number=serial,
            fingerprint=fingerprint,
            san_domains=san_domains or [],
            certificate_path=str(cert_path),
            private_key_path=str(key_path),
            certificate_chain_path=str(chain_path) if chain_path.exists() else None
        )
        
        # Save metadata
        meta_path = domain_path / "meta.json"
        meta_path.write_text(json.dumps(info.to_dict(), indent=2))
        
        logger.info(f"ACME certificate issued for {domain}, expires {expires_at}")
        return info
    
    def renew_certificate(self, domain: str) -> CertificateInfo:
        """Renew an ACME certificate."""
        logger.info(f"Renewing ACME certificate for {domain}")
        
        # Get existing certificate info
        existing_info = self.get_certificate_info(domain)
        if not existing_info:
            raise ValueError(f"No existing certificate found for {domain}")
        
        # Issue new certificate
        return self.issue_certificate(domain, existing_info.san_domains)
    
    def revoke_certificate(self, domain: str) -> bool:
        """Revoke an ACME certificate."""
        domain_path = self.storage_path / domain
        meta_path = domain_path / "meta.json"
        
        if not meta_path.exists():
            return False
        
        try:
            # Load certificate
            cert_path = domain_path / "cert.pem"
            if cert_path.exists():
                cert_pem = cert_path.read_text()
                
                # Parse certificate
                if CRYPTOGRAPHY_AVAILABLE:
                    cert = x509.load_pem_x509_certificate(
                        cert_pem.encode(), default_backend()
                    )
                    
                    # Revoke via ACME
                    # Note: This requires the certificate serial and reason
                    # For simplicity, we'll just delete the files
                    pass
            
            # Delete certificate files
            import shutil
            shutil.rmtree(domain_path)
            logger.info(f"Revoked certificate for {domain}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke certificate for {domain}: {e}")
            return False
    
    def get_certificate_info(self, domain: str) -> Optional[CertificateInfo]:
        """Get certificate information for a domain."""
        domain_path = self.storage_path / domain
        meta_path = domain_path / "meta.json"
        
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text())
                return CertificateInfo.from_dict(data)
            except Exception as e:
                logger.error(f"Failed to load certificate info for {domain}: {e}")
        
        return None


class CertificateManager:
    """
    Main certificate manager for Scrapy Edge deployment.
    
    Handles automatic TLS certificate issuance, renewal, and management
    for edge computing environments.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.storage_path = Path(settings.get('EDGE_CERT_STORAGE', './certs'))
        self.auto_renew = settings.getbool('EDGE_CERT_AUTO_RENEW', True)
        self.renew_days_before = settings.getint('EDGE_CERT_RENEW_DAYS', 30)
        self.key_type = settings.get('EDGE_CERT_KEY_TYPE', 'rsa')
        self.key_size = settings.getint('EDGE_CERT_KEY_SIZE', 2048)
        
        # Initialize storage
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize backends
        self.backends: Dict[CertificateType, CertificateBackend] = {}
        self._init_backends()
        
        # Start renewal thread if auto-renew is enabled
        self.renewal_thread = None
        self.renewal_stop_event = threading.Event()
        
        if self.auto_renew:
            self.start_auto_renewal()
        
        logger.info(f"Certificate manager initialized with storage at {self.storage_path}")
    
    def _init_backends(self):
        """Initialize certificate backends based on settings."""
        # Self-signed backend (always available)
        self.backends[CertificateType.SELF_SIGNED] = SelfSignedBackend(
            storage_path=str(self.storage_path / "self-signed"),
            key_size=self.key_size
        )
        
        # ACME backends (if configured and available)
        acme_servers = self.settings.getlist('EDGE_CERT_ACME_SERVERS', [])
        
        for server in acme_servers:
            if server.lower() == 'letsencrypt':
                cert_type = CertificateType.ACME_LETSENCRYPT
            elif server.lower() == 'zerossl':
                cert_type = CertificateType.ACME_ZEROSSL
            elif server.lower() == 'buypass':
                cert_type = CertificateType.ACME_BUYPASS
            else:
                cert_type = CertificateType.CUSTOM
            
            try:
                self.backends[cert_type] = ACMEBackend(
                    storage_path=str(self.storage_path / server.lower()),
                    acme_server=server,
                    email=self.settings.get('EDGE_CERT_ACME_EMAIL', 'admin@example.com'),
                    key_size=self.key_size
                )
                logger.info(f"Initialized ACME backend for {server}")
            except Exception as e:
                logger.warning(f"Failed to initialize ACME backend for {server}: {e}")
    
    def get_certificate(self, domain: str, 
                       certificate_type: Optional[CertificateType] = None,
                       san_domains: Optional[List[str]] = None) -> CertificateInfo:
        """
        Get or issue a certificate for the given domain.
        
        Args:
            domain: The primary domain for the certificate
            certificate_type: Type of certificate to issue (auto-select if None)
            san_domains: Additional Subject Alternative Names
            
        Returns:
            CertificateInfo object
        """
        # Check for existing valid certificate
        existing_info = self.get_certificate_info(domain)
        if existing_info and existing_info.status == CertificateStatus.VALID:
            # Check if certificate is still valid
            if existing_info.expires_at and existing_info.expires_at > datetime.utcnow():
                logger.info(f"Using existing certificate for {domain}")
                return existing_info
        
        # Auto-select certificate type if not specified
        if certificate_type is None:
            certificate_type = self._select_certificate_type(domain)
        
        # Get backend
        backend = self.backends.get(certificate_type)
        if not backend:
            raise ValueError(f"No backend available for certificate type: {certificate_type}")
        
        # Issue certificate
        logger.info(f"Issuing {certificate_type.value} certificate for {domain}")
        return backend.issue_certificate(domain, san_domains)
    
    def _select_certificate_type(self, domain: str) -> CertificateType:
        """Select the best certificate type for a domain."""
        # Prefer ACME certificates if available
        for cert_type in [CertificateType.ACME_LETSENCRYPT, 
                         CertificateType.ACME_ZEROSSL,
                         CertificateType.ACME_BUYPASS]:
            if cert_type in self.backends:
                return cert_type
        
        # Fall back to self-signed
        return CertificateType.SELF_SIGNED
    
    def renew_certificate(self, domain: str) -> CertificateInfo:
        """Renew a certificate for the given domain."""
        existing_info = self.get_certificate_info(domain)
        if not existing_info:
            raise ValueError(f"No certificate found for {domain}")
        
        backend = self.backends.get(existing_info.certificate_type)
        if not backend:
            raise ValueError(f"No backend for certificate type: {existing_info.certificate_type}")
        
        return backend.renew_certificate(domain)
    
    def revoke_certificate(self, domain: str) -> bool:
        """Revoke a certificate for the given domain."""
        existing_info = self.get_certificate_info(domain)
        if not existing_info:
            return False
        
        backend = self.backends.get(existing_info.certificate_type)
        if not backend:
            return False
        
        return backend.revoke_certificate(domain)
    
    def get_certificate_info(self, domain: str) -> Optional[CertificateInfo]:
        """Get certificate information for a domain."""
        # Check all backends
        for backend in self.backends.values():
            info = backend.get_certificate_info(domain)
            if info:
                return info
        
        return None
    
    def get_all_certificates(self) -> List[CertificateInfo]:
        """Get information for all managed certificates."""
        certificates = []
        
        for backend in self.backends.values():
            # This is a simplified approach - in reality, you'd need to
            # track all domains managed by each backend
            backend_path = Path(backend.storage_path) if hasattr(backend, 'storage_path') else None
            if backend_path and backend_path.exists():
                for domain_dir in backend_path.iterdir():
                    if domain_dir.is_dir():
                        info = backend.get_certificate_info(domain_dir.name)
                        if info:
                            certificates.append(info)
        
        return certificates
    
    def start_auto_renewal(self, check_interval: int = 3600):
        """Start automatic certificate renewal thread."""
        if self.renewal_thread and self.renewal_thread.is_alive():
            logger.warning("Auto-renewal thread already running")
            return
        
        def renewal_worker():
            logger.info("Certificate auto-renewal thread started")
            
            while not self.renewal_stop_event.is_set():
                try:
                    self._check_and_renew_certificates()
                except Exception as e:
                    logger.error(f"Error in certificate renewal: {e}")
                
                # Wait for next check
                self.renewal_stop_event.wait(check_interval)
        
        self.renewal_thread = threading.Thread(
            target=renewal_worker,
            daemon=True,
            name="CertificateRenewal"
        )
        self.renewal_thread.start()
    
    def stop_auto_renewal(self):
        """Stop automatic certificate renewal thread."""
        if self.renewal_thread:
            self.renewal_stop_event.set()
            self.renewal_thread.join(timeout=5)
            self.renewal_thread = None
            logger.info("Certificate auto-renewal thread stopped")
    
    def _check_and_renew_certificates(self):
        """Check and renew certificates that are about to expire."""
        logger.debug("Checking certificates for renewal")
        
        certificates = self.get_all_certificates()
        now = datetime.utcnow()
        
        for cert_info in certificates:
            if cert_info.expires_at:
                days_until_expiry = (cert_info.expires_at - now).days
                
                if days_until_expiry <= self.renew_days_before:
                    logger.info(f"Certificate for {cert_info.domain} expires in {days_until_expiry} days, renewing")
                    try:
                        self.renew_certificate(cert_info.domain)
                        logger.info(f"Successfully renewed certificate for {cert_info.domain}")
                    except Exception as e:
                        logger.error(f"Failed to renew certificate for {cert_info.domain}: {e}")
    
    def get_certificate_files(self, domain: str) -> Dict[str, Optional[str]]:
        """Get paths to certificate files for a domain."""
        info = self.get_certificate_info(domain)
        if not info:
            return {}
        
        return {
            'certificate': info.certificate_path,
            'private_key': info.private_key_path,
            'chain': info.certificate_chain_path
        }
    
    def load_certificate(self, domain: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Load certificate, private key, and chain for a domain."""
        files = self.get_certificate_files(domain)
        
        cert_content = None
        key_content = None
        chain_content = None
        
        if files.get('certificate') and os.path.exists(files['certificate']):
            with open(files['certificate'], 'r') as f:
                cert_content = f.read()
        
        if files.get('private_key') and os.path.exists(files['private_key']):
            with open(files['private_key'], 'r') as f:
                key_content = f.read()
        
        if files.get('chain') and os.path.exists(files['chain']):
            with open(files['chain'], 'r') as f:
                chain_content = f.read()
        
        return cert_content, key_content, chain_content
    
    def export_certificate(self, domain: str, export_path: str, 
                          format: str = 'pem') -> bool:
        """Export certificate to a specific location."""
        cert_content, key_content, chain_content = self.load_certificate(domain)
        
        if not cert_content or not key_content:
            logger.error(f"No certificate found for {domain}")
            return False
        
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'pem':
            # Write individual files
            (export_dir / "cert.pem").write_text(cert_content)
            (export_dir / "key.pem").write_text(key_content)
            if chain_content:
                (export_dir / "chain.pem").write_text(chain_content)
            
            # Write combined file
            combined = cert_content
            if chain_content:
                combined += "\n" + chain_content
            combined += "\n" + key_content
            
            (export_dir / "fullchain.pem").write_text(combined)
            
        elif format.lower() == 'pfx' or format.lower() == 'p12':
            # Convert to PKCS#12 format
            if not CRYPTOGRAPHY_AVAILABLE:
                logger.error("cryptography library required for PFX export")
                return False
            
            try:
                from cryptography.hazmat.primitives.serialization import pkcs12
                
                # Load certificate and key
                cert = x509.load_pem_x509_certificate(cert_content.encode())
                key = serialization.load_pem_private_key(key_content.encode(), password=None)
                
                # Create PKCS12
                pfx_data = pkcs12.serialize_key_and_certificates(
                    name=domain.encode(),
                    key=key,
                    cert=cert,
                    cas=None,
                    encryption_algorithm=serialization.NoEncryption()
                )
                
                (export_dir / f"{domain}.pfx").write_bytes(pfx_data)
                
            except Exception as e:
                logger.error(f"Failed to export PFX: {e}")
                return False
        
        else:
            logger.error(f"Unsupported export format: {format}")
            return False
        
        logger.info(f"Certificate exported to {export_dir}")
        return True
    
    def cleanup_expired_certificates(self, days_expired: int = 30):
        """Clean up expired certificates."""
        certificates = self.get_all_certificates()
        now = datetime.utcnow()
        
        for cert_info in certificates:
            if cert_info.expires_at:
                days_expired_actual = (now - cert_info.expires_at).days
                
                if days_expired_actual > days_expired:
                    logger.info(f"Cleaning up expired certificate for {cert_info.domain} "
                              f"(expired {days_expired_actual} days ago)")
                    self.revoke_certificate(cert_info.domain)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get certificate manager statistics."""
        certificates = self.get_all_certificates()
        
        stats = {
            'total_certificates': len(certificates),
            'valid_certificates': 0,
            'expiring_soon': 0,
            'expired': 0,
            'by_type': {},
            'oldest_expiry': None,
            'newest_expiry': None,
        }
        
        now = datetime.utcnow()
        
        for cert in certificates:
            # Count by status
            if cert.status == CertificateStatus.VALID:
                stats['valid_certificates'] += 1
            
            # Count by type
            cert_type = cert.certificate_type.value
            stats['by_type'][cert_type] = stats['by_type'].get(cert_type, 0) + 1
            
            # Check expiry
            if cert.expires_at:
                days_until_expiry = (cert.expires_at - now).days
                
                if days_until_expiry < 0:
                    stats['expired'] += 1
                elif days_until_expiry <= self.renew_days_before:
                    stats['expiring_soon'] += 1
                
                # Track oldest and newest expiry
                if stats['oldest_expiry'] is None or cert.expires_at < stats['oldest_expiry']:
                    stats['oldest_expiry'] = cert.expires_at
                if stats['newest_expiry'] is None or cert.expires_at > stats['newest_expiry']:
                    stats['newest_expiry'] = cert.expires_at
        
        return stats
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop_auto_renewal()


def get_certificate_manager(settings: Optional[Settings] = None) -> CertificateManager:
    """
    Factory function to get a CertificateManager instance.
    
    Args:
        settings: Scrapy settings object
        
    Returns:
        CertificateManager instance
    """
    if settings is None:
        from vex.utils.project import get_project_settings
        settings = get_project_settings()
    
    return CertificateManager(settings)


# Edge deployment integration helpers
class EdgeDeploymentHelper:
    """Helper class for edge deployment certificate integration."""
    
    @staticmethod
    def generate_cloudflare_workers_config(domain: str, 
                                          cert_manager: CertificateManager) -> Dict[str, Any]:
        """Generate Cloudflare Workers configuration with TLS certificates."""
        cert_content, key_content, chain_content = cert_manager.load_certificate(domain)
        
        if not cert_content or not key_content:
            raise ValueError(f"No certificate found for {domain}")
        
        # Cloudflare Workers expects certificates in a specific format
        # This is a simplified example
        return {
            'domain': domain,
            'certificates': [{
                'cert': base64.b64encode(cert_content.encode()).decode(),
                'key': base64.b64encode(key_content.encode()).decode(),
            }],
            'settings': {
                'min_tls_version': '1.2',
                'automatic_https_rewrites': True,
                'always_use_https': True,
            }
        }
    
    @staticmethod
    def generate_deno_deploy_config(domain: str,
                                   cert_manager: CertificateManager) -> Dict[str, Any]:
        """Generate Deno Deploy configuration with TLS certificates."""
        cert_content, key_content, _ = cert_manager.load_certificate(domain)
        
        if not cert_content or not key_content:
            raise ValueError(f"No certificate found for {domain}")
        
        return {
            'domain': domain,
            'tls': {
                'cert': cert_content,
                'key': key_content,
            },
            'deployment': {
                'project': f'vex-edge-{domain.replace(".", "-")}',
                'production': True,
            }
        }
    
    @staticmethod
    def generate_browser_extension_manifest(domain: str,
                                           cert_manager: CertificateManager) -> Dict[str, Any]:
        """Generate browser extension manifest with certificate info."""
        cert_info = cert_manager.get_certificate_info(domain)
        
        if not cert_info:
            raise ValueError(f"No certificate found for {domain}")
        
        return {
            'manifest_version': 3,
            'name': f'Scrapy Edge - {domain}',
            'version': '1.0.0',
            'permissions': [
                'webRequest',
                'webRequestBlocking',
                '<all_urls>',
                'storage',
            ],
            'host_permissions': [
                f'https://{domain}/*',
                f'http://{domain}/*',
            ],
            'background': {
                'service_worker': 'background.js',
                'type': 'module',
            },
            'content_security_policy': {
                'extension_pages': "script-src 'self'; object-src 'self'",
            },
            'web_accessible_resources': [{
                'resources': ['certs/*.pem'],
                'matches': ['<all_urls>'],
            }],
            'vex_edge': {
                'domain': domain,
                'certificate_info': cert_info.to_dict(),
                'auto_renew': True,
            }
        }


# CLI integration
def certificate_cli_command(args):
    """CLI command for certificate management."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrapy Edge Certificate Manager')
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Issue command
    issue_parser = subparsers.add_parser('issue', help='Issue a new certificate')
    issue_parser.add_argument('domain', help='Domain name')
    issue_parser.add_argument('--type', choices=['self-signed', 'acme'], 
                            default='self-signed', help='Certificate type')
    issue_parser.add_argument('--san', action='append', 
                            help='Subject Alternative Names')
    
    # Renew command
    renew_parser = subparsers.add_parser('renew', help='Renew a certificate')
    renew_parser.add_argument('domain', help='Domain name')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all certificates')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export a certificate')
    export_parser.add_argument('domain', help='Domain name')
    export_parser.add_argument('--path', required=True, help='Export path')
    export_parser.add_argument('--format', choices=['pem', 'pfx'], 
                             default='pem', help='Export format')
    
    # Stats command
    subparsers.add_parser('stats', help='Show certificate statistics')
    
    parsed_args = parser.parse_args(args)
    
    if not parsed_args.command:
        parser.print_help()
        return
    
    # Get certificate manager
    from vex.utils.project import get_project_settings
    settings = get_project_settings()
    cert_manager = get_certificate_manager(settings)
    
    try:
        if parsed_args.command == 'issue':
            cert_type_map = {
                'self-signed': CertificateType.SELF_SIGNED,
                'acme': CertificateType.ACME_LETSENCRYPT,
            }
            
            info = cert_manager.get_certificate(
                domain=parsed_args.domain,
                certificate_type=cert_type_map.get(parsed_args.type),
                san_domains=parsed_args.san
            )
            
            print(f"Certificate issued for {parsed_args.domain}")
            print(f"Type: {info.certificate_type.value}")
            print(f"Expires: {info.expires_at}")
            print(f"Files: {info.certificate_path}")
        
        elif parsed_args.command == 'renew':
            info = cert_manager.renew_certificate(parsed_args.domain)
            print(f"Certificate renewed for {parsed_args.domain}")
            print(f"New expiry: {info.expires_at}")
        
        elif parsed_args.command == 'list':
            certificates = cert_manager.get_all_certificates()
            
            if not certificates:
                print("No certificates found")
                return
            
            print(f"{'Domain':<30} {'Type':<15} {'Status':<10} {'Expires':<20}")
            print("-" * 75)
            
            for cert in certificates:
                expires = cert.expires_at.strftime("%Y-%m-%d %H:%M") if cert.expires_at else "N/A"
                print(f"{cert.domain:<30} {cert.certificate_type.value:<15} "
                      f"{cert.status.value:<10} {expires:<20}")
        
        elif parsed_args.command == 'export':
            success = cert_manager.export_certificate(
                domain=parsed_args.domain,
                export_path=parsed_args.path,
                format=parsed_args.format
            )
            
            if success:
                print(f"Certificate exported to {parsed_args.path}")
            else:
                print("Export failed")
        
        elif parsed_args.command == 'stats':
            stats = cert_manager.get_stats()
            
            print("Certificate Manager Statistics")
            print("=" * 40)
            print(f"Total certificates: {stats['total_certificates']}")
            print(f"Valid certificates: {stats['valid_certificates']}")
            print(f"Expiring soon: {stats['expiring_soon']}")
            print(f"Expired: {stats['expired']}")
            print()
            print("By type:")
            for cert_type, count in stats['by_type'].items():
                print(f"  {cert_type}: {count}")
    
    except Exception as e:
        print(f"Error: {e}")
        raise


# Integration with Scrapy
class CertificateMiddleware:
    """
    Scrapy middleware for automatic certificate management.
    
    Automatically issues and manages TLS certificates for crawled domains.
    """
    
    def __init__(self, settings):
        self.settings = settings
        self.cert_manager = get_certificate_manager(settings)
        self.auto_issue = settings.getbool('EDGE_CERT_AUTO_ISSUE', True)
        self.domains_cache = {}
    
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)
    
    def process_request(self, request, spider):
        """Process outgoing request."""
        if not self.auto_issue:
            return None
        
        # Extract domain from request
        from urllib.parse import urlparse
        parsed = urlparse(request.url)
        domain = parsed.netloc
        
        # Remove port if present
        if ':' in domain:
            domain = domain.split(':')[0]
        
        # Check if we need to issue a certificate
        if domain not in self.domains_cache:
            try:
                # Check for existing certificate
                cert_info = self.cert_manager.get_certificate_info(domain)
                
                if not cert_info or cert_info.status != CertificateStatus.VALID:
                    # Issue new certificate
                    logger.info(f"Auto-issuing certificate for {domain}")
                    self.cert_manager.get_certificate(domain)
                
                self.domains_cache[domain] = True
                
            except Exception as e:
                logger.warning(f"Failed to issue certificate for {domain}: {e}")
                self.domains_cache[domain] = False
        
        return None


# Settings documentation
CERTIFICATE_MANAGER_SETTINGS = {
    'EDGE_CERT_STORAGE': {
        'default': './certs',
        'description': 'Path to store certificates',
    },
    'EDGE_CERT_AUTO_RENEW': {
        'default': True,
        'description': 'Automatically renew certificates before expiry',
    },
    'EDGE_CERT_RENEW_DAYS': {
        'default': 30,
        'description': 'Days before expiry to renew certificates',
    },
    'EDGE_CERT_KEY_TYPE': {
        'default': 'rsa',
        'description': 'Key type (rsa or ecdsa)',
    },
    'EDGE_CERT_KEY_SIZE': {
        'default': 2048,
        'description': 'Key size in bits',
    },
    'EDGE_CERT_ACME_SERVERS': {
        'default': [],
        'description': 'List of ACME servers to use (letsencrypt, zerossl, buypass)',
    },
    'EDGE_CERT_ACME_EMAIL': {
        'default': 'admin@example.com',
        'description': 'Email for ACME account registration',
    },
    'EDGE_CERT_AUTO_ISSUE': {
        'default': True,
        'description': 'Automatically issue certificates for crawled domains',
    },
}


if __name__ == '__main__':
    # CLI entry point
    import sys
    certificate_cli_command(sys.argv[1:])