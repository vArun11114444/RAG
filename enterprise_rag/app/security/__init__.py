"""
app/security — Production-grade security layer for the Enterprise RAG pipeline.

Public API:

    from app.security import SecurityPipeline, SecurityRequest, SecurityResult
    from app.security.exceptions import SecurityException, SecurityEventType

Initialise once at startup:

    pipeline = SecurityPipeline()
    pipeline.initialise()   # builds Presidio engine

Per-request:

    result = await pipeline.run(SecurityRequest(query=..., uploaded_files=[...]))
    if result.blocked:
        raise HTTPException(400, result.block_reason)
    sanitized_query = result.sanitized_query
"""
from app.security.exceptions import (
    SecurityException,
    SecurityEventType,
    PIIDetectedException,
    PromptInjectionException,
    FileValidationException,
    MaliciousFileException,
)
from app.security.pii_detector import PIIDetector, PIIDetectionResult, PIIEntity
from app.security.data_masker import DataMasker, MaskingResult, MaskingStrategy
from app.security.prompt_injection import PromptInjectionDetector, InjectionDetectionResult
from app.security.file_validator import FileValidator, FileValidationResult, UploadedFile
from app.security.security_pipeline import SecurityPipeline, SecurityRequest, SecurityResult

__all__ = [
    # Pipeline
    "SecurityPipeline",
    "SecurityRequest",
    "SecurityResult",
    # Exceptions
    "SecurityException",
    "SecurityEventType",
    "PIIDetectedException",
    "PromptInjectionException",
    "FileValidationException",
    "MaliciousFileException",
    # Components
    "PIIDetector",
    "PIIDetectionResult",
    "PIIEntity",
    "DataMasker",
    "MaskingResult",
    "MaskingStrategy",
    "PromptInjectionDetector",
    "InjectionDetectionResult",
    "FileValidator",
    "FileValidationResult",
    "UploadedFile",
]
