import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
import logging

from sqlalchemy import create_engine, Column, Integer, Text, JSON, DateTime, String, text
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector
from sentence_transformers import SentenceTransformer
from datetime import datetime

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# SETUP
# -------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

logger.info("Loading sentence-transformers model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
VECTOR_DIM = 384
logger.info("Model loaded.")


# -------------------------------------------------------------------------
# ТАБЛИЦЫ
# -------------------------------------------------------------------------

class ConfluenceChunk(Base):
    """Chunks из Confluence. Используется для similarity search и AI анализа."""
    __tablename__ = "confluence_chunks"

    id = Column(Integer, primary_key=True)
    page_id = Column(String, nullable=False, index=True)
    page_title = Column(Text, nullable=False)
    section = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    breadcrumb = Column(Text)
    chunk_index = Column(Integer, default=0)
    embedding = Column(Vector(VECTOR_DIM))
    metadata_ = Column("metadata", JSON, default={})
    synced_at = Column(DateTime, default=datetime.utcnow)


class Clarification(Base):
    """Память платформы — все clarifications от пользователей."""
    __tablename__ = "clarifications"

    id = Column(Integer, primary_key=True)
    page_id = Column(String, nullable=False, index=True)
    page_title = Column(Text)
    gap_description = Column(Text, nullable=False)
    clarification_text = Column(Text, nullable=False)
    embedding = Column(Vector(VECTOR_DIM))
    project_key = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


def create_tables():
    """Создает таблицы при старте. Безопасно запускать повторно."""
    # CREATE EXTENSION требует autocommit
    raw_conn = engine.raw_connection()
    try:
        raw_conn.set_isolation_level(0)  # AUTOCOMMIT
        cursor = raw_conn.cursor()
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            logger.info("✅ pgvector extension ready")
        except Exception as ext_err:
            # Уже установлено или нет прав — не критично, продолжаем
            logger.warning(f"⚠️ CREATE EXTENSION skipped: {ext_err}")
        cursor.close()
    finally:
        raw_conn.close()

    Base.metadata.create_all(engine)
    logger.info("✅ Tables created successfully")


# -------------------------------------------------------------------------
# EMBEDDINGS
# -------------------------------------------------------------------------

def get_embedding(text_input: str) -> List[float]:
    """Конвертирует текст в вектор через sentence-transformers (384 dims)."""
    return embedding_model.encode(text_input, normalize_embeddings=True).tolist()


# -------------------------------------------------------------------------
# CHUNK OPERATIONS
# -------------------------------------------------------------------------

def save_chunks(chunks: List[Dict]) -> int:
    """Upsert chunks в pgvector. Не создает дубли при повторном sync."""
    session = SessionLocal()
    saved_count = 0

    try:
        for chunk in chunks:
            existing = session.query(ConfluenceChunk).filter_by(
                page_id=chunk["page_id"],
                chunk_index=chunk["chunk_index"]
            ).first()

            embedding = get_embedding(chunk["content"])

            if existing:
                existing.content = chunk["content"]
                existing.section = chunk["section"]
                existing.breadcrumb = chunk["breadcrumb"]
                existing.embedding = embedding
                existing.metadata_ = chunk.get("metadata", {})
                existing.synced_at = datetime.utcnow()
            else:
                new_chunk = ConfluenceChunk(
                    page_id=chunk["page_id"],
                    page_title=chunk["page_title"],
                    section=chunk["section"],
                    content=chunk["content"],
                    breadcrumb=chunk["breadcrumb"],
                    chunk_index=chunk["chunk_index"],
                    embedding=embedding,
                    metadata_=chunk.get("metadata", {})
                )
                session.add(new_chunk)
                saved_count += 1

        session.commit()
        logger.info(f"✅ Saved {saved_count} new chunks to pgvector")
        return saved_count

    except Exception as e:
        session.rollback()
        logger.error(f"❌ Error saving chunks: {e}")
        raise
    finally:
        session.close()


def find_related_pages(
    page_ids: List[str],
    limit: int = 5,
    similarity_threshold: float = 0.5
) -> List[Dict]:
    """
    Находит related pages через pgvector cosine similarity.

    FIX #4: order_by использует sqlalchemy text() вместо строки — работает надежно.
    """
    session = SessionLocal()

    try:
        selected_chunks = session.query(ConfluenceChunk).filter(
            ConfluenceChunk.page_id.in_(page_ids)
        ).all()

        if not selected_chunks:
            logger.warning("⚠️ No chunks found for selected pages. Run bulk_sync first.")
            return []

        related_map = {}

        for chunk in selected_chunks:
            if chunk.embedding is None:
                continue

            # FIX #4: используем .label() + text() для надежного order_by
            similarity_expr = (
                1 - ConfluenceChunk.embedding.cosine_distance(chunk.embedding)
            ).label("similarity")

            results = session.query(
                ConfluenceChunk,
                similarity_expr
            ).filter(
                ConfluenceChunk.page_id.notin_(page_ids),
                ConfluenceChunk.embedding.isnot(None)
            ).order_by(
                text("similarity DESC")  # FIX: text() делает ORDER BY надежным
            ).limit(10).all()

            for similar_chunk, similarity in results:
                sim_float = float(similarity)
                if sim_float < similarity_threshold:
                    continue

                pid = similar_chunk.page_id
                if pid not in related_map:
                    related_map[pid] = {
                        "page_id": pid,
                        "page_title": similar_chunk.page_title,
                        "similarity_score": 0.0,
                        "matched_sections": [],
                        "match_count": 0
                    }

                related_map[pid]["similarity_score"] = max(
                    related_map[pid]["similarity_score"],
                    round(sim_float, 2)
                )
                related_map[pid]["match_count"] += 1

                section = similar_chunk.section
                if section not in related_map[pid]["matched_sections"]:
                    related_map[pid]["matched_sections"].append(section)

        related = sorted(
            related_map.values(),
            key=lambda x: (x["similarity_score"], x["match_count"]),
            reverse=True
        )[:limit]

        logger.info(f"✅ Found {len(related)} related pages")
        return related

    finally:
        session.close()


# -------------------------------------------------------------------------
# CLARIFICATION MEMORY
# -------------------------------------------------------------------------

def save_clarification(
    page_id: str,
    page_title: str,
    gap_description: str,
    clarification_text: str,
    project_key: str = None
) -> int:
    """Сохраняет ответ пользователя на gap в память платформы."""
    session = SessionLocal()

    try:
        embedding = get_embedding(gap_description)

        clarification = Clarification(
            page_id=page_id,
            page_title=page_title,
            gap_description=gap_description,
            clarification_text=clarification_text,
            embedding=embedding,
            project_key=project_key
        )
        session.add(clarification)
        session.commit()

        logger.info(f"✅ Clarification saved for page '{page_title}'")
        return clarification.id

    except Exception as e:
        session.rollback()
        logger.error(f"❌ Error saving clarification: {e}")
        raise
    finally:
        session.close()


def find_similar_clarifications(
    gap_description: str,
    limit: int = 3,
    similarity_threshold: float = 0.7
) -> List[Dict]:
    """
    Ищет похожие gaps которые уже были решены.
    Cross-project память: находит решения из других модулей.

    FIX #4: order_by через text() вместо строки.
    """
    session = SessionLocal()

    try:
        query_embedding = get_embedding(gap_description)

        similarity_expr = (
            1 - Clarification.embedding.cosine_distance(query_embedding)
        ).label("similarity")

        results = session.query(
            Clarification,
            similarity_expr
        ).filter(
            Clarification.embedding.isnot(None)
        ).order_by(
            text("similarity DESC")  # FIX: надежный order_by
        ).limit(20).all()

        similar = []
        for clarification, similarity in results:
            sim_float = float(similarity)
            if sim_float < similarity_threshold:
                continue

            similar.append({
                "gap": clarification.gap_description,
                "clarification": clarification.clarification_text,
                "page_title": clarification.page_title,
                "page_id": clarification.page_id,
                "project_key": clarification.project_key,
                "similarity": round(sim_float, 2),
                "created_at": clarification.created_at.isoformat()
            })

        return similar[:limit]

    finally:
        session.close()


def get_page_clarifications(page_id: str) -> List[Dict]:
    """Возвращает все прошлые clarifications для страницы."""
    session = SessionLocal()

    try:
        clarifications = session.query(Clarification).filter_by(
            page_id=page_id
        ).order_by(Clarification.created_at.desc()).all()

        return [
            {
                "gap": c.gap_description,
                "clarification": c.clarification_text,
                "created_at": c.created_at.isoformat()
            }
            for c in clarifications
        ]

    finally:
        session.close()

# -- SESSION PERSISTENCE ---------------------------------------------------

class PulsarSession(Base):
    __tablename__ = "pulsar_sessions"

    id = Column(String, primary_key=True)
    page_ids = Column(JSON, default=[])
    page_titles = Column(JSON, default={})
    page_title_display = Column(Text, default="")
    chunks = Column(JSON, default=[])
    related_pages = Column(JSON, default=[])
    past_clarifications = Column(JSON, default=[])
    analysis_prompt = Column(Text, nullable=True)
    tc_prompt = Column(Text, nullable=True)
    bdd_prompt = Column(Text, nullable=True)
    gaps = Column(JSON, default=[])
    score = Column(JSON, default={})
    summary = Column(Text, default="")
    gap_reviews = Column(JSON, default={})
    review_1_approved = Column(JSON, default=False)
    manual_test_cases = Column(JSON, default=[])
    bdd_test_cases = Column(JSON, default=[])
    review_2_approved = Column(JSON, default=False)
    review_3_approved = Column(JSON, default=False)
    export_ready = Column(JSON, default=False)
    exported_at = Column(Text, nullable=True)
    current_stage = Column(String, default="human_review_1")
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)


# -- SESSION CRUD ----------------------------------------------------------

def _session_to_dict(row) -> dict:
    return {
        "session_id": row.id,
        "page_ids": row.page_ids or [],
        "page_titles": row.page_titles or {},
        "page_title_display": row.page_title_display or "",
        "chunks": row.chunks or [],
        "related_pages": row.related_pages or [],
        "past_clarifications": row.past_clarifications or [],
        "analysis_prompt": row.analysis_prompt,
        "tc_prompt": row.tc_prompt,
        "bdd_prompt": row.bdd_prompt,
        "gaps": row.gaps or [],
        "score": row.score or {},
        "summary": row.summary or "",
        "gap_reviews": row.gap_reviews or {},
        "review_1_approved": row.review_1_approved or False,
        "manual_test_cases": row.manual_test_cases or [],
        "bdd_test_cases": row.bdd_test_cases or [],
        "review_2_approved": row.review_2_approved or False,
        "review_3_approved": row.review_3_approved or False,
        "export_ready": row.export_ready or False,
        "exported_at": row.exported_at,
        "current_stage": row.current_stage,
        "error": row.error,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def session_save(data: dict) -> None:
    db = SessionLocal()
    try:
        existing = db.query(PulsarSession).filter_by(id=data["session_id"]).first()
        if existing:
            skip = {"session_id", "created_at"}
            for k, v in data.items():
                if k not in skip and hasattr(existing, k):
                    setattr(existing, k, v)
            existing.updated_at = datetime.utcnow()
        else:
            skip = {"session_id"}
            row = PulsarSession(
                id=data["session_id"],
                **{k: v for k, v in data.items() if k not in skip and hasattr(PulsarSession, k)}
            )
            db.add(row)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"session_save error: {e}")
        raise
    finally:
        db.close()


def session_get(session_id: str) -> Optional[dict]:
    db = SessionLocal()
    try:
        row = db.query(PulsarSession).filter_by(id=session_id).first()
        if not row:
            return None
        return _session_to_dict(row)
    finally:
        db.close()


def session_list() -> list:
    db = SessionLocal()
    try:
        rows = db.query(PulsarSession).order_by(PulsarSession.updated_at.desc()).all()
        return [
            {
                "session_id": r.id,
                "page_title_display": r.page_title_display,
                "page_ids": r.page_ids,
                "current_stage": r.current_stage,
                "score": r.score,
                "created_at": r.created_at.isoformat() if r.created_at else None,
                "updated_at": r.updated_at.isoformat() if r.updated_at else None,
            }
            for r in rows
        ]
    finally:
        db.close()


def session_delete(session_id: str) -> bool:
    db = SessionLocal()
    try:
        row = db.query(PulsarSession).filter_by(id=session_id).first()
        if not row:
            return False
        db.delete(row)
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        logger.error(f"session_delete error: {e}")
        raise
    finally:
        db.close()
