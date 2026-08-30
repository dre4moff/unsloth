from __future__ import annotations

import base64
import hashlib
import os
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID


class DesktopIdentity:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.key_path = root / "desktop-signing-key.pem"
        self.cert_path = root / "desktop-certificate.pem"
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        self._key = self._load_or_create_key()
        self._certificate = self._load_or_create_certificate()

    @property
    def public_key_base64(self) -> str:
        value = self._key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
        return base64.b64encode(value).decode("ascii")

    @property
    def certificate_sha256(self) -> str:
        return hashlib.sha256(self._certificate.public_bytes(serialization.Encoding.DER)).hexdigest()

    def sign_base64(self, data: bytes) -> str:
        return base64.b64encode(self._key.sign(data, ec.ECDSA(hashes.SHA256()))).decode("ascii")

    @staticmethod
    def verify_base64(public_key_base64: str, signature_base64: str, data: bytes) -> None:
        key = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), base64.b64decode(public_key_base64, validate=True))
        key.verify(base64.b64decode(signature_base64, validate=True), data, ec.ECDSA(hashes.SHA256()))

    @staticmethod
    def pairing_code(server_key: str, device_key: str, nonce: bytes, certificate_sha256: str) -> str:
        digest = hashlib.sha256(base64.b64decode(server_key) + base64.b64decode(device_key) + nonce + certificate_sha256.encode()).digest()
        return f"{int.from_bytes(digest[:4], 'big') % 1_000_000:06d}"

    def _load_or_create_key(self) -> ec.EllipticCurvePrivateKey:
        if self.key_path.exists():
            value = serialization.load_pem_private_key(self.key_path.read_bytes(), password=None)
            if not isinstance(value, ec.EllipticCurvePrivateKey) or not isinstance(value.curve, ec.SECP256R1):
                raise RuntimeError("Companion desktop key is not P-256")
            return value
        key = ec.generate_private_key(ec.SECP256R1())
        self._write_private(self.key_path, key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        return key

    def _load_or_create_certificate(self) -> x509.Certificate:
        if self.cert_path.exists():
            certificate = x509.load_pem_x509_certificate(self.cert_path.read_bytes())
            if certificate.not_valid_after_utc > datetime.now(timezone.utc) + timedelta(days=30):
                return certificate
        now = datetime.now(timezone.utc)
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Unsloth Companion Desktop")])
        certificate = (
            x509.CertificateBuilder()
            .subject_name(name).issuer_name(name).public_key(self._key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(days=3650))
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(socket.gethostname()), x509.DNSName("localhost")]), critical=False)
            .sign(self._key, hashes.SHA256())
        )
        self._write_private(self.cert_path, certificate.public_bytes(serialization.Encoding.PEM))
        return certificate

    @staticmethod
    def _write_private(path: Path, data: bytes) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        os.chmod(temporary, 0o600)
        temporary.replace(path)
