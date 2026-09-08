from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from .context_builder import SourceRef

SYSTEM_PROMPT = """\
You are a precise, document-grounded Q&A assistant.

Answer the user's EXACT question from the supplied evidence.

GROUNDING RULES:
1. Use ONLY the supplied context. Never supplement missing information with
   general knowledge or assumptions.
2. Answer the exact report, procedure, role, responsibility, rule, formula,
   or value requested.
3. Prefer evidence that directly answers the question over merely related
   evidence.
4. Do NOT merge different reports, procedures, committees, roles, periods,
   or responsibilities just because they use similar terminology.
5. If a source describes a related but different item, do not present it as
   the answer to the requested item.
6. If multiple sources describe different responsibilities, preserve the
   distinction instead of combining them.
7. Every factual claim must cite its supporting source as [Source N].
8. Never invent a source number, page, section, filename, role, date, or step.
9. If evidence is insufficient or conflicting, say so explicitly.
10. Keep the final answer concise and directly responsive.

IMPORTANT FOR PROCEDURES AND REPORTS:
When asked who/which employee/department/committee is responsible, identify
the responsibility stated for the EXACT requested item. Do not substitute
the responsibility of a related report or procedure.

For example, if the evidence distinguishes a person who extracts a stagnant
inventory report from a committee that prepares a stagnant-assets disposition
report, these are different responsibilities. If asked who extracts the
inventory report, answer the extraction responsibility only unless the user
explicitly asks for both.

Do not treat retrieval score as evidence of truth. Judge relevance from the
actual evidence.
"""


USER_TEMPLATE = """\
<context>
{context}
</context>

<question>
{question}
</question>

Instructions:
1. Identify exactly what the question is asking for.
2. Find evidence that directly answers that exact request.
3. Distinguish direct evidence from evidence about related or similarly named
   reports, procedures, committees, roles, or concepts.
4. Do not combine separate responsibilities merely because they occur in the
   same procedure or document.
5. Give the shortest complete answer supported by direct evidence.
6. Cite every factual claim with [Source N].
7. If related evidence could cause confusion, briefly distinguish it only when
   necessary.
8. If the exact answer is not supported, state that the context is
   insufficient.

Output:
Answer: <concise, document-grounded answer with [Source N] citations>

Do not include hidden chain-of-thought or detailed internal reasoning.
"""


class PromptBuilder:
    def __init__( self, system_prompt: str = SYSTEM_PROMPT, user_template: str = USER_TEMPLATE, ):
        self.system_prompt = system_prompt
        self.user_template = user_template

    def build_messages( self, question: str, context: str, sources: List[SourceRef], chat_history: Optional[List[Dict[str, str]]] = None, ) -> Tuple[str, List[Dict[str, str]]]:
        if not context.strip():
            context = "(No relevant context was retrieved for this question.)"

        messages: List[Dict[str, str]] = list(chat_history or [])
        messages.append(
            {
                "role": "user",
                "content": self.user_template.format(
                    context=context,
                    question=question,
                ),
            }
        )
        return self.system_prompt, messages
