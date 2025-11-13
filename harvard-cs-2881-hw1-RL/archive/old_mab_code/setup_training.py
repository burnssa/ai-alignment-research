
"""
Setup Training Module - Backward Compatibility

This module provides backward compatibility by re-exporting functions from the new modular structure.
For new code, prefer importing directly from policy.py and openai_client.py.
"""

# Re-export for backward compatibility
from policy import Policy, get_policy, moving_average
from openai_client import call_openai, persona_prompt, judge_response

