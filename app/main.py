import uuid
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from app.confluence import ConfluenceClient
from app.database import create_tables, save_chunks, find_related_pages
from app.graph import (
    start_analysis,
    submit_gap_review,
    submit_tc_review,
    submit_bdd_review,
    get_session_state,
    list_sessions,
    delete_session,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Pulsar — AI-Assisted QA Platform",
    description="Requirement analysis and test case generation engine",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

confluence = ConfluenceClient()


@app.on_event("startup")
async def startup():
    try:
        create_tables()
        logger.info("✅ Database ready")
    except Exception as e:
        logger.error(f"❌ Database startup error: {e}")


# ── Request models ────────────────────────────────────────────────────────

class BulkSyncRequest(BaseModel):
    space_key: str

class StartAnalysisRequest(BaseModel):
    page_ids: List[str]
    analysis_prompt: Optional[str] = None
    tc_prompt: Optional[str] = None
    bdd_prompt: Optional[str] = None

class GapReviewRequest(BaseModel):
    session_id: str
    gap_id: str
    action: str
    comment: str = ""

class TCReviewRequest(BaseModel):
    session_id: str
    test_cases: List[dict] = []
    approved: bool
    tc_prompt: Optional[str] = None

class BDDReviewRequest(BaseModel):
    session_id: str
    test_cases: List[dict] = []
    approved: bool

class UpdatePromptRequest(BaseModel):
    session_id: str
    tc_prompt: Optional[str] = None
    bdd_prompt: Optional[str] = None
    analysis_prompt: Optional[str] = None



# ── Health ────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Pulsar Backend", "version": "2.0.0"}


# ── Confluence ────────────────────────────────────────────────────────────

@app.get("/confluence/search")
async def search_confluence_pages(query: str, limit: int = 20):
    try:
        pages = confluence.search_pages(query=query, limit=limit)
        return {"pages": pages, "total": len(pages)}
    except Exception as e:
        logger.error(f"Confluence search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/confluence/sync")
async def bulk_sync_confluence(request: BulkSyncRequest):
    try:
        chunks = confluence.bulk_sync_space(request.space_key)
        saved = save_chunks(chunks)
        return {"status": "success", "space_key": request.space_key,
                "total_chunks": len(chunks), "new_chunks_saved": saved}
    except Exception as e:
        logger.error(f"Bulk sync error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/confluence/related")
async def get_related_pages(page_ids: str, limit: int = 5):
    try:
        ids = [pid.strip() for pid in page_ids.split(",")]
        related = find_related_pages(ids, limit=limit)
        return {"related_pages": related}
    except Exception as e:
        logger.error(f"Related pages error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Sessions (History) ────────────────────────────────────────────────────

@app.get("/sessions")
async def get_sessions():
    try:
        return {"sessions": list_sessions()}
    except Exception as e:
        logger.error(f"List sessions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    try:
        deleted = delete_session(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "deleted", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete session error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Analysis ──────────────────────────────────────────────────────────────

@app.post("/analysis/start")
async def start_analysis_endpoint(request: StartAnalysisRequest):
    try:
        session_id = str(uuid.uuid4())
        chunks = confluence.fetch_selected_pages(request.page_ids)

        if not chunks:
            raise HTTPException(status_code=404,
                detail="No content found for selected pages.")

        page_titles = {}
        for chunk in chunks:
            if chunk["page_id"] not in page_titles:
                page_titles[chunk["page_id"]] = chunk["page_title"]

        save_chunks(chunks)

        result = start_analysis(
            session_id=session_id,
            page_ids=request.page_ids,
            page_titles=page_titles,
            chunks=chunks,
            analysis_prompt=request.analysis_prompt,
            tc_prompt=request.tc_prompt,
            bdd_prompt=request.bdd_prompt,
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis/{session_id}")
async def get_analysis_state(session_id: str):
    try:
        state = get_session_state(session_id)
        if not state:
            raise HTTPException(status_code=404, detail="Session not found")
        return state
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Reviews ───────────────────────────────────────────────────────────────

@app.post("/review/gap")
async def review_gap(request: GapReviewRequest):
    try:
        if request.action == "comment" and not request.comment:
            raise HTTPException(status_code=400,
                detail="Comment is required when action is 'comment'")
        if request.action not in ["approve", "comment", "skip"]:
            raise HTTPException(status_code=400,
                detail="Action must be: approve, comment, or skip")

        result = submit_gap_review(
            session_id=request.session_id,
            gap_id=request.gap_id,
            action=request.action,
            comment=request.comment,
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gap review error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/review/test-cases")
async def review_test_cases(request: TCReviewRequest):
    try:
        result = submit_tc_review(
            session_id=request.session_id,
            test_cases=request.test_cases,
            approved=request.approved,
        )
        return result
    except Exception as e:
        logger.error(f"TC review error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/review/bdd")
async def review_bdd(request: BDDReviewRequest):
    try:
        result = submit_bdd_review(
            session_id=request.session_id,
            test_cases=request.test_cases,
            approved=request.approved,
        )
        return result
    except Exception as e:
        logger.error(f"BDD review error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Export ────────────────────────────────────────────────────────────────

@app.get("/export/{session_id}")
async def export_results(session_id: str, format: str = "bdd"):
    try:
        state = get_session_state(session_id)
        if not state:
            raise HTTPException(status_code=404, detail="Session not found")
        if not state.get("export_ready"):
            raise HTTPException(status_code=400,
                detail="Export not ready. Complete BDD Review first.")

        test_cases = state.get("bdd_test_cases", [])
        return {
            "session_id": session_id,
            "format": format,
            "exported_at": state.get("exported_at"),
            "total": len(test_cases),
            "test_cases": test_cases,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/session/prompts")
async def update_session_prompts(request: UpdatePromptRequest):
    from app.database import session_get, session_save
    session = session_get(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    updates = {"session_id": request.session_id}
    if request.tc_prompt is not None:
        updates["tc_prompt"] = request.tc_prompt
    if request.bdd_prompt is not None:
        updates["bdd_prompt"] = request.bdd_prompt
    if request.analysis_prompt is not None:
        updates["analysis_prompt"] = request.analysis_prompt
    session_save(updates)
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)