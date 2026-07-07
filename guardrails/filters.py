"""
Programmatic guardrail filters that run inside the FastAPI request path,
in addition to the NeMo-Guardrails topical rails. These are fast,
deterministic checks that don't require an extra LLM call for every request.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

PII_REDACTION_ENABLED = os.getenv("PII_REDACTION_ENABLED", "true").lower() == "true"

_INJECTION_PATTERNS = [
    re.compile(r"ignore (all )?(previous|prior|above) instructions", re.IGNORECASE),
    re.compile(r"reveal (your |the )?system prompt", re.IGNORECASE),
    re.compile(r"you are now (DAN|jailbroken|unrestricted)", re.IGNORECASE),
    re.compile(r"disregard (your|all) (guidelines|rules|instructions)", re.IGNORECASE),
    re.compile(r"pretend (you|that you) (have no|are not) (restrictions|bound)", re.IGNORECASE),
]


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str = ""
    sanitized_text: str = ""


def check_prompt_injection(text: str) -> GuardrailResult:
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return GuardrailResult(allowed=False, reason="prompt_injection_detected", sanitized_text=text)
    return GuardrailResult(allowed=True, sanitized_text=text)


@lru_cache(maxsize=1)
def _get_presidio_engines():
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    # Use the pre-installed lightweight en_core_web_sm model to avoid downloading 400MB en_core_web_lg
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }
    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()

    return AnalyzerEngine(nlp_engine=nlp_engine), AnonymizerEngine()


def redact_pii(text: str) -> str:
    """Redact PII (emails, phone numbers, SSNs, credit cards) from text
    before it is logged or sent to a third-party LLM provider. Excludes PERSON
    and ORGANIZATION to preserve search utility."""
    if not PII_REDACTION_ENABLED:
        return text
    try:
        analyzer, anonymizer = _get_presidio_engines()
        # Keep PERSON out of entities list to allow searching for names/companies
        target_entities = [
            "CREDIT_CARD",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "IP_ADDRESS",
            "IN_AADHAAR",
            "IN_PAN",
            "US_SSN",
        ]
        results = analyzer.analyze(text=text, language="en", entities=target_entities)
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text
    except Exception:  # noqa: BLE001
        # Fail open on redaction errors but never fail the whole request over it.
        return text


def run_input_guardrails(user_query: str) -> GuardrailResult:
    injection_check = check_prompt_injection(user_query)
    if not injection_check.allowed:
        return injection_check

    sanitized = redact_pii(user_query)
    return GuardrailResult(allowed=True, sanitized_text=sanitized)


def run_output_guardrails(answer: str) -> GuardrailResult:
    """Redact any PII that may have leaked into the generated answer."""
    sanitized = redact_pii(answer)
    return GuardrailResult(allowed=True, sanitized_text=sanitized)
