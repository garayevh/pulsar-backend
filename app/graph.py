import os
import json
import logging
from typing import List, Dict, Optional
from datetime import datetime
import requests
import urllib3
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.database import (
    find_related_pages,
    get_page_clarifications,
    find_similar_clarifications,
    save_clarification,
)

# ── Setup ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
urllib3.disable_warnings()
load_dotenv(override=True)

# ── Bedrock config ────────────────────────────────────────────────────────
BEDROCK_URL = "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-sonnet-4-6/invoke"
PROXY_HOST = "proxy.azercell.com:8080"
PROXY_USER = "garayevh"


# ── Default prompts ───────────────────────────────────────────────────────
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
You are a senior QA engineer. Generate comprehensive test cases in BDD format (Given/When/Then).

Requirements:
{requirements_text}

Identified gaps and clarifications:
{clarifications_text}

Generate test cases covering:
1. Positive scenarios (happy path)
2. Negative scenarios (invalid inputs, boundary violations)
3. Edge cases (limits, empty states, concurrency)
4. Risk-based scenarios (high-impact areas from gap analysis)

Respond ONLY in valid JSON, no markdown, no backticks, no explanation:
{{
  "test_cases": [
    {{
      "id": "TC_001",
      "title": "Short descriptive title",
      "type": "positive | negative | edge | risk",
      "priority": "high | medium | low",
      "given": "System state and preconditions",
      "when": "Action performed by the user or system",
      "then": "Expected result",
      "notes": "Optional: additional context or risk notes"
    }}
  ]
}}
"""


# ── State ─────────────────────────────────────────────────────────────────
class PipelineState(TypedDict):
    session_id: str
    page_ids: List[str]
    page_titles: Dict[str, str]
    chunks: List[Dict]
    related_pages: List[Dict]
    past_clarifications: List[Dict]
    analysis_prompt: Optional[str]
    tc_prompt: Optional[str]
    gaps: List[Dict]
    score: Dict
    summary: str
    gap_reviews: Dict[str, Dict]
    review_1_approved: bool
    test_cases: List[Dict]
    review_2_approved: bool
    export_ready: bool
    exported_at: Optional[str]
    current_stage: str
    error: Optional[str]


# ── Helpers ───────────────────────────────────────────────────────────────
def _parse_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON from AI response."""
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


# ── AI caller ─────────────────────────────────────────────────────────────
def call_ai(
    prompt: str,
    system: str = "",
    max_tokens: int = 8000,
    thinking_budget: int = 5000,
) -> str:
    """
    Call Bedrock API via corporate proxy with extended thinking.
    Analysis:      max_tokens=8000,  thinking_budget=5000
    TC generation: max_tokens=32000, thinking_budget=4000
    """
    proxy_password = os.getenv("CORP_PROXY_PASSWORD", "")
    if not proxy_password:
        raise ValueError("CORP_PROXY_PASSWORD not set in .env")

    proxies = {
        "http":  f"http://{PROXY_USER}:{proxy_password}@{PROXY_HOST}",
        "https": f"http://{PROXY_USER}:{proxy_password}@{PROXY_HOST}",
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BEDROCK_API_KEY}",
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

    resp = requests.post(
        BEDROCK_URL, headers=headers, json=body,
        proxies=proxies, verify=False, timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()

    if "content" in data and isinstance(data["content"], list):
        for block in data["content"]:
            if isinstance(block, dict) and block.get("type") == "text":
                logger.info("[call_ai] Success")
                return block["text"]

    raise ValueError(f"Cannot extract text from Bedrock response. Keys: {list(data.keys())}")


# ── LangGraph nodes ───────────────────────────────────────────────────────
def node_load_context(state: PipelineState) -> PipelineState:
    related = find_related_pages(state["page_ids"], limit=5)
    all_past = []
    for pid in state["page_ids"]:
        all_past.extend(get_page_clarifications(pid))
    return {**state, "related_pages": related, "past_clarifications": all_past, "current_stage": "analysis"}


def node_analyze(state: PipelineState) -> PipelineState:
    logger.info(f"[node_analyze] chunks={len(state['chunks'])}")
    past_text = "\n".join(
        f"- Gap: {c['gap']} → Clarification: {c['clarification']}"
        for c in state.get("past_clarifications", [])
    ) or "None"

    prompt = (state.get("analysis_prompt") or DEFAULT_ANALYSIS_PROMPT).format(
        requirements_text=_build_requirements_text(state["chunks"]),
        past_clarifications=past_text,
    )
    raw = call_ai(prompt, max_tokens=8000, thinking_budget=5000)

    try:
        result = _parse_json(raw)
    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"[node_analyze] Parse error: {e} | raw[:300]={raw[:300]}")
        result = {"gaps": [], "score": {"total": 0, "breakdown": []}, "summary": "Parse error"}

    return {**state, "gaps": result.get("gaps", []), "score": result.get("score", {}),
            "summary": result.get("summary", ""), "current_stage": "human_review_1"}


def node_human_review_1(state: PipelineState) -> PipelineState:
    return {**state, "current_stage": "human_review_1", "review_1_approved": False}


def node_process_gap_review(state: PipelineState) -> PipelineState:
    gap_reviews = state.get("gap_reviews", {})
    score_total = state.get("score", {}).get("total", 0)
    breakdown = list(state.get("score", {}).get("breakdown", []))

    for gap_id, review in gap_reviews.items():
        action, comment = review.get("action"), review.get("comment", "")
        if action == "comment" and comment:
            gap = next((g for g in state["gaps"] if g["id"] == gap_id), None)
            if gap:
                for pid in state["page_ids"]:
                    save_clarification(
                        page_id=pid,
                        page_title=state["page_titles"].get(pid, ""),
                        gap_description=gap["description"],
                        clarification_text=comment,
                    )
                penalty = gap.get("penalty", 0)
                score_total = min(100, max(0, score_total - penalty))
                breakdown.append({"factor": f"Clarification: {gap['description'][:50]}...",
                                   "impact": abs(penalty), "type": "recovery"})

    all_reviewed = all(g["id"] in gap_reviews for g in state["gaps"])
    return {
        **state,
        "score": {**state.get("score", {}), "total": score_total, "breakdown": breakdown},
        "review_1_approved": all_reviewed,
        "current_stage": "test_case_generation" if all_reviewed else "human_review_1",
    }


def node_generate_test_cases(state: PipelineState) -> PipelineState:
    logger.info("[node_generate_test_cases] Generating")
    prompt = (state.get("tc_prompt") or DEFAULT_TC_PROMPT).format(
        requirements_text=_build_requirements_text(state["chunks"]),
        clarifications_text=_build_clarifications_text(
            state.get("gaps", []), state.get("gap_reviews", {})
        ),
    )
    raw = call_ai(prompt, max_tokens=32000, thinking_budget=4000)

    try:
        test_cases = _parse_json(raw).get("test_cases", [])
    except (json.JSONDecodeError, IndexError) as e:
        logger.error(f"[node_generate_test_cases] Parse error: {e} | raw[:300]={raw[:300]}")
        test_cases = []

    now = datetime.utcnow().isoformat()
    for tc in test_cases:
        tc["generated_at"] = now
        tc["source_pages"] = state["page_ids"]
        tc["manually_edited"] = False

    return {**state, "test_cases": test_cases, "current_stage": "human_review_2"}


def node_human_review_2(state: PipelineState) -> PipelineState:
    return {**state, "current_stage": "human_review_2", "review_2_approved": False}


def node_export(state: PipelineState) -> PipelineState:
    bdd = []
    for tc in state.get("test_cases", []):
        bdd.append({
            "id": tc["id"], "title": tc["title"],
            "type": tc["type"], "priority": tc["priority"],
            "bdd": f"Scenario: {tc['title']}\n  Given {tc['given']}\n  When {tc['when']}\n  Then {tc['then']}",
            "raw": tc, "source_pages": tc.get("source_pages", []),
            "manually_edited": tc.get("manually_edited", False),
        })
    return {**state, "export_ready": True, "exported_at": datetime.utcnow().isoformat(),
            "test_cases": bdd, "current_stage": "completed"}


# ── Conditional edges ─────────────────────────────────────────────────────
def should_continue_review_1(state: PipelineState) -> str:
    return "generate_test_cases" if state.get("review_1_approved") else "human_review_1"

def should_continue_review_2(state: PipelineState) -> str:
    return "export" if state.get("review_2_approved") else "human_review_2"


# ── Pipeline assembly ─────────────────────────────────────────────────────
def build_pipeline() -> StateGraph:
    wf = StateGraph(PipelineState)
    for name, fn in [
        ("load_context", node_load_context), ("analyze", node_analyze),
        ("human_review_1", node_human_review_1), ("process_gap_review", node_process_gap_review),
        ("generate_test_cases", node_generate_test_cases), ("human_review_2", node_human_review_2),
        ("export", node_export),
    ]:
        wf.add_node(name, fn)

    wf.set_entry_point("load_context")
    wf.add_edge("load_context", "analyze")
    wf.add_edge("analyze", "human_review_1")
    wf.add_edge("human_review_1", "process_gap_review")
    wf.add_edge("generate_test_cases", "human_review_2")
    wf.add_edge("export", END)
    wf.add_conditional_edges("process_gap_review", should_continue_review_1,
                              {"human_review_1": END, "generate_test_cases": "generate_test_cases"})
    wf.add_conditional_edges("human_review_2", should_continue_review_2,
                              {"human_review_2": END, "export": "export"})
    return wf.compile(checkpointer=MemorySaver())

pipeline = build_pipeline()


# ── Session store ─────────────────────────────────────────────────────────
_sessions: Dict[str, Dict] = {}


def start_analysis(
    session_id: str, page_ids: List[str], page_titles: Dict[str, str],
    chunks: List[Dict], analysis_prompt: str = None, tc_prompt: str = None,
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

    session = {
        "session_id": session_id, "page_ids": page_ids, "page_titles": page_titles,
        "chunks": chunks, "related_pages": related, "past_clarifications": all_past,
        "analysis_prompt": analysis_prompt, "tc_prompt": tc_prompt,
        "gaps": result.get("gaps", []), "score": result.get("score", {}),
        "summary": result.get("summary", ""), "gap_reviews": {},
        "review_1_approved": False, "test_cases": [], "review_2_approved": False,
        "export_ready": False, "exported_at": None, "current_stage": "human_review_1",
    }
    _sessions[session_id] = session
    return {"session_id": session_id, "stage": "human_review_1",
            "gaps": session["gaps"], "score": session["score"],
            "summary": session["summary"], "related_pages": related}


def submit_gap_review(session_id: str, gap_id: str, action: str, comment: str = "") -> Dict:
    session = _sessions.get(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")

    gap_reviews = {**session.get("gap_reviews", {}), gap_id: {"action": action, "comment": comment}}
    session["gap_reviews"] = gap_reviews

    gaps = session.get("gaps", [])
    score_total = session.get("score", {}).get("total", 0)
    breakdown = list(session.get("score", {}).get("breakdown", []))

    if action == "comment" and comment:
        gap = next((g for g in gaps if g["id"] == gap_id), None)
        if gap:
            penalty = gap.get("penalty", 0)
            score_total = min(100, max(0, score_total - penalty))
            breakdown.append({"factor": f"Clarification: {gap['description'][:50]}...",
                               "impact": abs(penalty), "type": "recovery"})
            for pid in session.get("page_ids", []):
                save_clarification(
                    page_id=pid,
                    page_title=session.get("page_titles", {}).get(pid, ""),
                    gap_description=gap["description"],
                    clarification_text=comment,
                )

    updated_score = {**session.get("score", {}), "total": score_total, "breakdown": breakdown}
    session["score"] = updated_score
    all_reviewed = all(g["id"] in gap_reviews for g in gaps)
    session["review_1_approved"] = all_reviewed
    _sessions[session_id] = session

    if all_reviewed:
        session["current_stage"] = "test_case_generation"
        test_cases = _generate_test_cases(session)
        session["test_cases"] = test_cases
        session["current_stage"] = "human_review_2"
        _sessions[session_id] = session
        return {"session_id": session_id, "stage": "human_review_2",
                "score": updated_score, "review_1_approved": True, "test_cases": test_cases}

    session["current_stage"] = "human_review_1"
    return {"session_id": session_id, "stage": "human_review_1",
            "score": updated_score, "review_1_approved": False}


def _generate_test_cases(session: Dict) -> List[Dict]:
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
        logger.error(f"[_generate_test_cases] Parse error: {e} | raw[:300]={raw[:300]}")
        test_cases = []

    now = datetime.utcnow().isoformat()
    for tc in test_cases:
        tc["generated_at"] = now
        tc["source_pages"] = session.get("page_ids", [])
        tc["manually_edited"] = False
    return test_cases


def submit_tc_review(session_id: str, test_cases: List[Dict], approved: bool) -> Dict:
    session = _sessions.get(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")

    if not test_cases:
        test_cases = session.get("test_cases", [])
    else:
        for tc in test_cases:
            tc["manually_edited"] = True

    session["test_cases"] = test_cases
    session["review_2_approved"] = approved

    if approved:
        bdd = []
        for tc in test_cases:
            bdd.append({
                "id": tc["id"], "title": tc["title"],
                "type": tc["type"], "priority": tc["priority"],
                "bdd": f"Scenario: {tc['title']}\n  Given {tc['given']}\n  When {tc['when']}\n  Then {tc['then']}",
                "raw": tc, "source_pages": tc.get("source_pages", []),
                "manually_edited": tc.get("manually_edited", False),
            })
        session.update({"test_cases": bdd, "export_ready": True,
                        "exported_at": datetime.utcnow().isoformat(), "current_stage": "completed"})
        _sessions[session_id] = session
        return {"session_id": session_id, "stage": "completed",
                "export_ready": True, "test_cases": bdd}

    session["current_stage"] = "human_review_2"
    _sessions[session_id] = session
    return {"session_id": session_id, "stage": "human_review_2",
            "export_ready": False, "test_cases": test_cases}


def get_session_state(session_id: str) -> Optional[Dict]:
    return _sessions.get(session_id)