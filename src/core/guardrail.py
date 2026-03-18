#!/usr/bin/env python3
"""
Guardrail Module for the CE RAG System
Provides input validation for safety, scope, academic integrity, and PII protection.
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict, Optional, Any, Tuple

from ..utils.config import config
from ..utils.logger import get_logger
from ..utils.performance_monitor import monitor_operation
from .llm_client import LLMClient, LLMProvider

logger = get_logger(__name__)

# Maps guardrail classifications to user-facing rejection messages
REJECTION_MESSAGES = {
    "injection": "Your query has been rejected because it appears to be an attempt to manipulate the system.",
    "abusive": "Your query has been rejected due to violation of usage policies (abusive or inappropriate content).",
    "academic_integrity": "I cannot assist with requests that may compromise academic integrity (e.g., exam leaks, cheating, or bypassing academic rules).",
    "pii_request": "I cannot provide personal or private information about students or staff that is not publicly available.",
    "malicious_comparison": "I cannot rank, demean, or compare professors/staff in a harmful or disrespectful manner.",
    "out_of_scope": "I am sorry, but I can only assist with Computer Engineering academic queries and topics related to KMITL.",
}

class GuardrailException(Exception):
    """Exception raised when a query is rejected by guardrails."""
    def __init__(self, message: str, reason: str = "abusive"):
        self.message = message
        self.reason = reason
        super().__init__(self.message)

class Guardrail:
    """Orchestrates input guardrails for safety, scope, and policy enforcement."""
    
    def __init__(self, ollama_url: str = None, model_name: str = None):
        self.llm_client = LLMClient(
            provider=LLMProvider.OLLAMA,
            ollama_url=ollama_url,
            model_name=model_name or config.models.ollama_model
        )
        self.available = self.llm_client.is_available()
        
        # All valid safety classifications from the guardrail prompt
        self._valid_labels = set(REJECTION_MESSAGES.keys()) | {"safe"}
        
        # Schema for guardrail response
        self.GUARDRAIL_SCHEMA = {
            "type": "object",
            "properties": {
                "safety": {
                    "type": "string",
                    "enum": ["safe", "injection", "abusive", "academic_integrity", "pii_request", "malicious_comparison", "out_of_scope"]
                },
                "reason": {"type": "string"}
            },
            "required": ["safety", "reason"]
        }

    @monitor_operation("input_guardrail")
    async def validate(self, query: str) -> bool:
        """
        Validate that the input query is safe, in-scope, and policy-compliant.
        Returns True if safe, raises GuardrailException if rejected.
        """
        if not self.available:
            logger.warning("Guardrail LLM not available, skipping validation")
            return True
            
        try:
            prompt_file = os.path.join(
                os.path.dirname(__file__),
                "..", "..", "prompt", "input_guardrail.md"
            )
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_template = f.read()

            prompt = prompt_template.format(query=query)
            
            async with aiohttp.ClientSession() as session:
                response_text = await self.llm_client.generate_async(
                    session, 
                    prompt, 
                    temperature=0.0, 
                    format=self.GUARDRAIL_SCHEMA,
                    num_predict=config.models.num_predict_short
                )
            
            if not response_text.strip():
                logger.warning("Empty response from Guardrail. Assuming safe.")
                return True

            response_json = self._parse_json(response_text)
            if not response_json:
                logger.warning(f"Failed to parse Guardrail response. Raw: {response_text[:200]}. Assuming safe.")
                return True
            
            safety = response_json.get("safety", "safe")
            reason = response_json.get("reason", "No reason provided")
            logger.debug(f"Guardrail LLM response: safety={safety}, reason={reason}")
            
            if safety == "safe":
                logger.info(f"Guardrail PASSED (safe): {reason}")
                return True
            
            # Any non-safe label triggers rejection
            if safety in REJECTION_MESSAGES:
                logger.warning(f"Guardrail REJECTED ({safety}): {reason}")
                raise GuardrailException(
                    REJECTION_MESSAGES[safety],
                    reason=safety
                )
            
            # Unknown label — log and let through
            logger.warning(f"Guardrail returned unknown label '{safety}': {reason}. Assuming safe.")
            return True
                
        except GuardrailException:
            raise
        except Exception as e:
            logger.error(f"Error in input guardrail validation: {e}")
            return True

    def _parse_json(self, text: str) -> Optional[Dict]:
        """Parse JSON response from LLM."""
        try:
            cleaned = text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            return json.loads(cleaned.strip())
        except Exception:
            return None
