import uuid
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from app.confluence import ConfluenceClient
from app.database import create_tables, save_chunks, find_related_pages
from app.graph import start_analysis, submit_gap_review, submit_tc_review, get_session_state

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# APP SETUP
# -------------------------------------------------------------------------

app = FastAPI(
    title="Pulsar — AI-Assisted QA Platform",
    description="Requirement analysis and test case generation engine",
    version="1.0.0"
)

# CORS — разрешаем фронтенду (Next.js) обращаться к бэкенду
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

confluence = ConfluenceClient()


# -------------------------------------------------------------------------
# STARTUP
# -------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    """Создаем таблицы при старте если их нет."""
    try:
        create_tables()
        logger.info("✅ Database ready")
    except Exception as e:
        logger.error(f"❌ Database startup error: {e}")


# -------------------------------------------------------------------------
# REQUEST / RESPONSE MODELS
# -------------------------------------------------------------------------

class SearchPagesRequest(BaseModel):
    query: str
    limit: int = 20

class BulkSyncRequest(BaseModel):
    space_key: str  # Например: "PROD", "DEV", "QA"

class StartAnalysisRequest(BaseModel):
    page_ids: List[str]
    analysis_prompt: Optional[str] = None  # Кастомный prompt из UI
    tc_prompt: Optional[str] = None        # Кастомный TC prompt из UI

class GapReviewRequest(BaseModel):
    session_id: str
    gap_id: str
    action: str        # "approve" | "comment" | "skip"
    comment: str = ""  # Обязателен если action == "comment"

class TCReviewRequest(BaseModel):
    session_id: str
    test_cases: List[dict]  # Отредактированные пользователем TC
    approved: bool


# -------------------------------------------------------------------------
# HEALTH CHECK
# -------------------------------------------------------------------------

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "Pulsar Backend",
        "version": "1.0.0"
    }


# -------------------------------------------------------------------------
# CONFLUENCE ENDPOINTS
# -------------------------------------------------------------------------

@app.get("/confluence/search")
async def search_confluence_pages(query: str, limit: int = 20):
    """
    STAGE 1 — Поиск страниц по названию.
    Вызывается когда пользователь вводит текст в поиск в UI.

    GET /confluence/search?query=Payment&limit=10

    Response:
    [
        {
            "page_id": "123",
            "title": "Payment Module",
            "space": "Backend",
            "breadcrumb": "Backend > Payments > Payment Module",
            "url": "https://..."
        }
    ]
    """
    try:
        pages = confluence.search_pages(query=query, limit=limit)
        return {"pages": pages, "total": len(pages)}
    except Exception as e:
        logger.error(f"Confluence search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/confluence/sync")
async def bulk_sync_confluence(request: BulkSyncRequest):
    """
    Bulk Sync — синхронизирует весь Confluence Space в pgvector.
    Запускается один раз (или по расписанию).
    После этого related pages работают мгновенно.

    POST /confluence/sync
    Body: {"space_key": "PROD"}
    """
    try:
        logger.info(f"Starting bulk sync for space: {request.space_key}")
        chunks = confluence.bulk_sync_space(request.space_key)
        saved = save_chunks(chunks)

        return {
            "status": "success",
            "space_key": request.space_key,
            "total_chunks": len(chunks),
            "new_chunks_saved": saved
        }
    except Exception as e:
        logger.error(f"Bulk sync error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/confluence/related")
async def get_related_pages(page_ids: str, limit: int = 5):
    """
    Находит potentially related pages для выбранных страниц.
    Использует pgvector similarity search.

    GET /confluence/related?page_ids=123,456&limit=5

    Response:
    [
        {
            "page_id": "789",
            "page_title": "Refund Policy",
            "similarity_score": 0.87,
            "matched_sections": ["Error Handling"]
        }
    ]
    """
    try:
        ids = [pid.strip() for pid in page_ids.split(",")]
        related = find_related_pages(ids, limit=limit)
        return {"related_pages": related}
    except Exception as e:
        logger.error(f"Related pages error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------------------
# ANALYSIS PIPELINE ENDPOINTS
# -------------------------------------------------------------------------

@app.post("/analysis/start")
async def start_analysis_endpoint(request: StartAnalysisRequest):
    """
    STAGE 2 — Запускает AI анализ для выбранных страниц.
    Создает новую сессию и запускает LangGraph pipeline.

    POST /analysis/start
    Body: {
        "page_ids": ["123", "456"],
        "analysis_prompt": "Optional custom prompt...",
        "tc_prompt": "Optional custom TC prompt..."
    }

    Response:
    {
        "session_id": "uuid",
        "stage": "human_review_1",
        "gaps": [...],
        "score": {"total": 72, "breakdown": [...]},
        "related_pages": [...]
    }
    """
    try:
        # Генерируем уникальный ID сессии
        session_id = str(uuid.uuid4())

        # Загружаем chunks выбранных страниц из Confluence
        chunks = confluence.fetch_selected_pages(request.page_ids)

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="No content found for selected pages. Check page IDs."
            )

        # Получаем titles для page_ids
        page_titles = {}
        for chunk in chunks:
            if chunk["page_id"] not in page_titles:
                page_titles[chunk["page_id"]] = chunk["page_title"]

        # Сохраняем chunks в pgvector (для future related pages поиска)
        save_chunks(chunks)

        # Запускаем LangGraph pipeline
        result = start_analysis(
            session_id=session_id,
            page_ids=request.page_ids,
            page_titles=page_titles,
            chunks=chunks,
            analysis_prompt=request.analysis_prompt,
            tc_prompt=request.tc_prompt
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis start error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis/{session_id}")
async def get_analysis_state(session_id: str):
    """
    Возвращает текущее состояние pipeline для сессии.
    Фронтенд использует для polling или восстановления после перезагрузки.

    GET /analysis/uuid-session-id
    """
    try:
        state = get_session_state(session_id)
        if not state:
            raise HTTPException(status_code=404, detail="Session not found")
        return state
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------------------
# HUMAN REVIEW ENDPOINTS
# -------------------------------------------------------------------------

@app.post("/review/gap")
async def review_gap(request: GapReviewRequest):
    """
    STAGE 3 — Пользователь обработал один gap.

    Три варианта action:
    - "approve" → gap принят как есть
    - "comment" → пользователь добавил clarification (обязателен comment)
    - "skip"    → gap игнорируется

    Если comment → score пересчитывается автоматически.
    Clarification сохраняется в память платформы (pgvector).

    POST /review/gap
    Body: {
        "session_id": "uuid",
        "gap_id": "gap_1",
        "action": "comment",
        "comment": "Empty fields should show inline validation error"
    }

    Response:
    {
        "session_id": "uuid",
        "stage": "human_review_1",
        "score": {"total": 80, "breakdown": [...]},
        "review_1_approved": false
    }
    """
    try:
        if request.action == "comment" and not request.comment:
            raise HTTPException(
                status_code=400,
                detail="Comment is required when action is 'comment'"
            )

        if request.action not in ["approve", "comment", "skip"]:
            raise HTTPException(
                status_code=400,
                detail="Action must be one of: approve, comment, skip"
            )

        result = submit_gap_review(
            session_id=request.session_id,
            gap_id=request.gap_id,
            action=request.action,
            comment=request.comment
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Gap review error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/review/test-cases")
async def review_test_cases(request: TCReviewRequest):
    """
    STAGE 5 — Пользователь проверил и одобрил test cases.

    Пользователь может редактировать каждый TC перед approve.
    После approve — pipeline переходит к экспорту в BDD формат.

    POST /review/test-cases
    Body: {
        "session_id": "uuid",
        "test_cases": [...],  // отредактированные TC
        "approved": true
    }

    Response:
    {
        "session_id": "uuid",
        "stage": "completed",
        "export_ready": true,
        "test_cases": [
            {
                "id": "TC_001",
                "bdd": "Scenario: ...\n  Given ...\n  When ...\n  Then ..."
            }
        ]
    }
    """
    try:
        result = submit_tc_review(
            session_id=request.session_id,
            test_cases=request.test_cases,
            approved=request.approved
        )
        return result
    except Exception as e:
        logger.error(f"TC review error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------------------
# EXPORT ENDPOINT
# -------------------------------------------------------------------------

@app.get("/export/{session_id}")
async def export_results(session_id: str, format: str = "bdd"):
    """
    STAGE 6 — Экспорт результатов.

    Форматы:
    - bdd (default) → Gherkin/BDD (Given/When/Then) для автоматизации
    - json          → Raw JSON для интеграций

    GET /export/uuid-session-id?format=bdd

    Response (bdd):
    {
        "session_id": "uuid",
        "format": "bdd",
        "exported_at": "2026-05-12T...",
        "test_cases": [
            {
                "id": "TC_001",
                "title": "...",
                "type": "positive",
                "priority": "high",
                "bdd": "Scenario: Valid login\n  Given user is on login page\n  When user enters valid credentials\n  Then user is redirected to dashboard"
            }
        ]
    }
    """
    try:
        state = get_session_state(session_id)

        if not state:
            raise HTTPException(status_code=404, detail="Session not found")

        if not state.get("export_ready"):
            raise HTTPException(
                status_code=400,
                detail="Export not ready. Complete Human Review #2 first."
            )

        test_cases = state.get("test_cases", [])

        if format == "bdd":
            return {
                "session_id": session_id,
                "format": "bdd",
                "exported_at": state.get("exported_at"),
                "total": len(test_cases),
                "test_cases": [
                    {
                        "id": tc.get("id"),
                        "title": tc.get("title"),
                        "type": tc.get("type"),
                        "priority": tc.get("priority"),
                        "bdd": tc.get("bdd")
                    }
                    for tc in test_cases
                ]
            }
        else:
            # Raw JSON формат
            return {
                "session_id": session_id,
                "format": "json",
                "exported_at": state.get("exported_at"),
                "total": len(test_cases),
                "test_cases": test_cases
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------------------------------------------------------
# RUN
# -------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload при изменении файлов (dev mode)
    )