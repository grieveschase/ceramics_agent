"""
Compatibility re-export module.

Source-of-truth prompts live in `agents/<agent>/prompt.py`.
"""

from agents.coverage.prompt import COVERAGE_PROMPT
from agents.evidence.prompt import EVIDENCE_PROMPT
from agents.marks.prompt import MARKS_PROMPT
from agents.hypotheses.prompt import HYPOTHESES_PROMPT
from agents.resolver.prompt import RESOLVER_PROMPT
from agents.value.prompt import VALUE_PROMPT

__all__ = [
    "COVERAGE_PROMPT",
    "EVIDENCE_PROMPT",
    "MARKS_PROMPT",
    "HYPOTHESES_PROMPT",
    "RESOLVER_PROMPT",
    "VALUE_PROMPT",
]
