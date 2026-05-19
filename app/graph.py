import os
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime
import requests
import urllib3
from dotenv import load_dotenv

from app.database import (
    find_related_pages,
    get_page_clarifications,
    find_similar_clarifications,
    save_clarification,
    session_save,
    session_get,
    session_list,
    session_delete,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
urllib3.disable_warnings()
load_dotenv(override=True)

BEDROCK_URL = "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-4-6/invoke"
PROXY_HOST = "proxy.azercell.com:8080"
PROXY_USER = "garayevh"


DEFAULT_ANALYSIS_PROMPT = """
You are a senior QA analyst. Analyze the following requirements from Confluence.

Your task:
1. Identify ALL gaps in the requirements:
   - Missing negative scenarios
   - Missing validation rules
   - Missing error handling
   - Missing edge cases
   - Ambiguous or contradictory statements

2. For each gap, generate ONE clear clarification question.

3. Calculate a Completeness Score (0-100) with this breakdown:
   PENALTIES:
   - Missing negative scenarios: -10 each
   - Missing validation rules: -8 each
   - Missing error handling: -6 each
   - Missing edge cases: -5 each
   BONUSES:
   - Complete happy path: +10
   - Good UI coverage: +15

Requirements to analyze:
{requirements_text}

Past clarifications already provided (apply these automatically):
{past_clarifications}

Respond ONLY in valid JSON, no markdown, no backticks, no explanation:
{{
  "gaps": [
    {{
      "id": "gap_1",
      "type": "missing_validation | missing_negative | missing_error | missing_edge | ambiguous",
      "description": "Clear description of the gap",
      "question": "Clarification question for the user",
      "penalty": -8
    }}
  ],
  "score": {{
    "total": 72,
    "breakdown": [
      {{"factor": "Missing validation rules (2)", "impact": -16, "type": "penalty"}},
      {{"factor": "Complete happy path", "impact": 10, "type": "bonus"}}
    ]
  }},
  "summary": "Brief summary of requirement quality"
}}
"""

DEFAULT_TC_PROMPT = """
You are a senior QA engineer. Generate comprehensive test cases in MANUAL STEPS format.

Requirements:
{requirements_text}

Identified gaps and clarifications:
{clarifications_text}

Generate test cases covering:
1. Positive scenarios (happy path)
2. Negative scenarios (invalid inputs, boundary violations)
3. Edge cases (limits, empty states, concurrency)
4. Risk-based scenarios (high-impact areas from gap analysis)

IMPORTANT: Use manual step-by-step format, NOT BDD.

Respond ONLY in valid JSON, no markdown, no backticks, no explanation:
{{
  "test_cases": [
    {{
      "id": "TC_001",
      "title": "Short descriptive title",
      "type": "positive | negative | edge | risk",
      "priority": "high | medium | low",
      "preconditions": "System state and prerequisites",
      "steps": [
        "Step 1: Navigate to...",
        "Step 2: Enter... in the field",
        "Step 3: Click..."
      ],
      "expected_result": "What should happen after all steps",
      "notes": "Optional context"
    }}
  ]
}}
"""

DEFAULT_BDD_PROMPT = """
You are a senior QA automation engineer. Convert these manual test cases into BDD format (Gherkin Given/When/Then).

Manual test cases:
{manual_test_cases}

Rules:
- Keep EXACTLY the same test cases — do NOT add or remove any
- Given: system state and preconditions
- When: the action performed
- Then: the expected result
- Use clear, automation-friendly language

Respond ONLY in valid JSON, no markdown, no backticks, no explanation:
{{
  "test_cases": [
    {{
      "id": "TC_001",
      "title": "Same title as manual",
      "type": "positive | negative | edge | risk",
      "priority": "high | medium | low",
      "given": "System state and preconditions",
      "when": "Action performed by the user or system",
      "then": "Expected result",
      "notes": "Optional notes"
    }}
  ]
}}
"""
def _parse_json(raw: str) -> dict:
    clean = raw.strip()
    if "```" in clean:
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    return json.loads(clean.strip())


def _build_requirements_text(chunks: List[Dict]) -> str:
    return "\n\n".join(f"## {c['breadcrumb']}\n{c['content']}" for c in chunks)


def _build_clarifications_text(gaps: List[Dict], gap_reviews: Dict) -> str:
    parts = []
    for gap in gaps:
        comment = gap_reviews.get(gap["id"], {}).get("comment", "")
        if not comment:
            similar = find_similar_clarifications(gap["description"], limit=1)
            if similar:
                comment = f"[From memory] {similar[0]['clarification']}"
        if comment:
            parts.append(f"Gap: {gap['description']}\nClarification: {comment}")
    return "\n\n".join(parts) or "No clarifications provided"


def _build_manual_tc_text(manual_test_cases: List[Dict]) -> str:
    parts = []
    for tc in manual_test_cases:
        steps_text = "\n".join(f"  {s}" for s in tc.get("steps", []))
        parts.append(
            f"ID: {tc['id']}\n"
            f"Title: {tc['title']}\n"
            f"Type: {tc['type']} | Priority: {tc['priority']}\n"
            f"Preconditions: {tc.get('preconditions', '')}\n"
            f"Steps:\n{steps_text}\n"
            f"Expected Result: {tc.get('expected_result', '')}\n"
            f"Notes: {tc.get('notes', '')}"
        )
    return "\n\n---\n\n".join(parts)


def call_ai(
    prompt: str,
    system: str = "",
    max_tokens: int = 8000,
    thinking_budget: int = 5000,
    max_retries: int = 3,
) -> str:
    load_dotenv(override=True)
    proxy_password = os.getenv("CORP_PROXY_PASSWORD", "")
    bedrock_api_key = os.getenv("BEDROCK_API_KEY", "")

    if not proxy_password:
        raise ValueError("CORP_PROXY_PASSWORD not set in .env")
    if not bedrock_api_key:
        raise ValueError("BEDROCK_API_KEY not set in .env")

    proxies = {
        "http":  f"http://{PROXY_USER}:{proxy_password}@{PROXY_HOST}",
        "https": f"http://{PROXY_USER}:{proxy_password}@{PROXY_HOST}",
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {bedrock_api_key}",
    }
    body: dict = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system

    logger.info(f"[call_ai] prompt_len={len(prompt)} max_tokens={max_tokens} thinking={thinking_budget}")

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                BEDROCK_URL, headers=headers, json=body,
                proxies=proxies, verify=False, timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()

            if "content" in data and isinstance(data["content"], list):
                for block in data["content"]:
                    if isinstance(block, dict) and block.get("type") == "text":
                        logger.info(f"[call_ai] Success on attempt {attempt}")
                        return block["text"]

            raise ValueError(f"Cannot extract text from Bedrock response. Keys: {list(data.keys())}")

        except Exception as e:
            logger.warning(f"[call_ai] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                wait = 10 * attempt  # 10s, 20s, 30s
                logger.info(f"[call_ai] Retrying in {wait}s...")
                import time
                time.sleep(wait)
            else:
                logger.error(f"[call_ai] All {max_retries} attempts failed")
                raise
def start_analysis(
    session_id: str,
    page_ids: List[str],
    page_titles: Dict[str, str],
    chunks: List[Dict],
    analysis_prompt: str = None,
    tc_prompt: str = None,
    bdd_prompt: str = None,
) -> Dict:
    related = find_related_pages(page_ids, limit=5)
    all_past = []
    for pid in page_ids:
        all_past.extend(get_page_clarifications(pid))

    past_text = "\n".join(
        f"- Gap: {c['gap']} → Clarification: {c['clarification']}" for c in all_past
    ) or "None"

    prompt = (analysis_prompt or DEFAULT_ANALYSIS_PROMPT).format(
        requirements_text=_build_requirements_text(chunks),
        past_clarifications=past_text,
    )
    raw = call_ai(prompt, max_tokens=8000, thinking_budget=5000)

    try:
        result = _parse_json(raw)
    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"[start_analysis] Parse error: {e} | raw[:300]={raw[:300]}")
        result = {"gaps": [], "score": {"total": 0, "breakdown": []}, "summary": "Parse error"}

    page_title_display = next(iter(page_titles.values()), "") if page_titles else ""

    session = {
        "session_id": session_id,
        "page_ids": page_ids,
        "page_titles": page_titles,
        "page_title_display": page_title_display,
        "chunks": chunks,
        "related_pages": related,
        "past_clarifications": all_past,
        "analysis_prompt": analysis_prompt,
        "tc_prompt": tc_prompt,
        "bdd_prompt": bdd_prompt,
        "gaps": result.get("gaps", []),
        "score": result.get("score", {}),
        "summary": result.get("summary", ""),
        "gap_reviews": {},
        "review_1_approved": False,
        "manual_test_cases": [],
        "bdd_test_cases": [],
        "review_2_approved": False,
        "review_3_approved": False,
        "export_ready": False,
        "exported_at": None,
        "current_stage": "human_review_1",
        "error": None,
    }
    session_save(session)

    return {
        "session_id": session_id,
        "current_stage": "human_review_1",
        "gaps": session["gaps"],
        "score": session["score"],
        "summary": session["summary"],
        "related_pages": related,
    }


def submit_gap_review(
    session_id: str, gap_id: str, action: str, comment: str = ""
) -> Dict:
    session = session_get(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")

    gap_reviews = {**session.get("gap_reviews", {}), gap_id: {"action": action, "comment": comment}}
    gaps = session.get("gaps", [])
    score_total = session.get("score", {}).get("total", 0)
    breakdown = list(session.get("score", {}).get("breakdown", []))

    if action == "comment" and comment:
        gap = next((g for g in gaps if g["id"] == gap_id), None)
        if gap:
            penalty = gap.get("penalty", 0)
            score_total = min(100, max(0, score_total - penalty))
            breakdown.append({
                "factor": f"Clarification: {gap['description'][:50]}...",
                "impact": abs(penalty),
                "type": "recovery",
            })
            for pid in session.get("page_ids", []):
                save_clarification(
                    page_id=pid,
                    page_title=session.get("page_titles", {}).get(pid, ""),
                    gap_description=gap["description"],
                    clarification_text=comment,
                )

    updated_score = {**session.get("score", {}), "total": score_total, "breakdown": breakdown}
    all_reviewed = all(g["id"] in gap_reviews for g in gaps)

    session_save({
        "session_id": session_id,
        "gap_reviews": gap_reviews,
        "score": updated_score,
        "review_1_approved": all_reviewed,
        "current_stage": "tc_generation" if all_reviewed else "human_review_1",
    })

    if all_reviewed:
        manual_tcs = _generate_manual_test_cases(session_get(session_id))
        session_save({
            "session_id": session_id,
            "manual_test_cases": manual_tcs,
            "current_stage": "human_review_2",
        })
        return {
            "session_id": session_id,
            "current_stage": "human_review_2",
            "score": updated_score,
            "review_1_approved": True,
            "manual_test_cases": manual_tcs,
        }

    return {
        "session_id": session_id,
        "current_stage": "human_review_1",
        "score": updated_score,
        "review_1_approved": False,
    }
def _generate_manual_test_cases(session: Dict) -> List[Dict]:
    prompt = (session.get("tc_prompt") or DEFAULT_TC_PROMPT).format(
        requirements_text=_build_requirements_text(session.get("chunks", [])),
        clarifications_text=_build_clarifications_text(
            session.get("gaps", []), session.get("gap_reviews", {})
        ),
    )
    raw = call_ai(prompt, max_tokens=32000, thinking_budget=4000)

    try:
        test_cases = _parse_json(raw).get("test_cases", [])
    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"[_generate_manual_test_cases] Parse error: {e} | raw[:300]={raw[:300]}")
        test_cases = []

    now = datetime.utcnow().isoformat()
    for tc in test_cases:
        tc["generated_at"] = now
        tc["source_pages"] = session.get("page_ids", [])
        tc["manually_edited"] = False
    return test_cases


def _generate_bdd_test_cases(session: Dict) -> List[Dict]:
    manual_tcs = session.get("manual_test_cases", [])
    prompt = (session.get("bdd_prompt") or DEFAULT_BDD_PROMPT).format(
        manual_test_cases=_build_manual_tc_text(manual_tcs),
    )
    raw = call_ai(prompt, max_tokens=32000, thinking_budget=4000)

    try:
        test_cases = _parse_json(raw).get("test_cases", [])
    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"[_generate_bdd_test_cases] Parse error: {e} | raw[:300]={raw[:300]}")
        test_cases = []

    now = datetime.utcnow().isoformat()
    for tc in test_cases:
        tc["generated_at"] = now
        tc["source_pages"] = session.get("page_ids", [])
        tc["manually_edited"] = False
    return test_cases


def submit_tc_review(
    session_id: str, test_cases: List[Dict], approved: bool
) -> Dict:
    session = session_get(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")

    if test_cases:
        for tc in test_cases:
            tc["manually_edited"] = True
    else:
        test_cases = session.get("manual_test_cases", [])

    if not approved:
        session_save({
            "session_id": session_id,
            "manual_test_cases": test_cases,
            "current_stage": "human_review_2",
        })
        return {
            "session_id": session_id,
            "current_stage": "human_review_2",
            "review_2_approved": False,
            "manual_test_cases": test_cases,
        }

    session_save({
        "session_id": session_id,
        "manual_test_cases": test_cases,
        "review_2_approved": True,
        "current_stage": "bdd_generation",
    })

    updated_session = session_get(session_id)
    bdd_tcs = _generate_bdd_test_cases(updated_session)

    session_save({
        "session_id": session_id,
        "bdd_test_cases": bdd_tcs,
        "current_stage": "human_review_3",
    })

    return {
        "session_id": session_id,
        "current_stage": "human_review_3",
        "review_2_approved": True,
        "bdd_test_cases": bdd_tcs,
    }


def submit_bdd_review(
    session_id: str, test_cases: List[Dict], approved: bool
) -> Dict:
    session = session_get(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")

    if test_cases:
        for tc in test_cases:
            tc["manually_edited"] = True
    else:
        test_cases = session.get("bdd_test_cases", [])

    if not approved:
        session_save({
            "session_id": session_id,
            "bdd_test_cases": test_cases,
            "current_stage": "human_review_3",
        })
        return {
            "session_id": session_id,
            "current_stage": "human_review_3",
            "review_3_approved": False,
            "bdd_test_cases": test_cases,
        }

    exported = []
    for tc in test_cases:
        exported.append({
            "id": tc["id"],
            "title": tc["title"],
            "type": tc["type"],
            "priority": tc["priority"],
            "bdd": f"Scenario: {tc['title']}\n  Given {tc['given']}\n  When {tc['when']}\n  Then {tc['then']}",
            "raw": tc,
            "source_pages": tc.get("source_pages", []),
            "manually_edited": tc.get("manually_edited", False),
        })

    session_save({
        "session_id": session_id,
        "bdd_test_cases": exported,
        "review_3_approved": True,
        "export_ready": True,
        "exported_at": datetime.utcnow().isoformat(),
        "current_stage": "completed",
    })

    return {
        "session_id": session_id,
        "current_stage": "completed",
        "review_3_approved": True,
        "export_ready": True,
        "bdd_test_cases": exported,
    }


def get_session_state(session_id: str) -> Optional[Dict]:
    return session_get(session_id)


def list_sessions() -> list:
    return session_list()


def delete_session(session_id: str) -> bool:
    return session_delete(session_id)