from __future__ import annotations

import contextlib
from datetime import datetime, timezone
from typing import Generator, List, Optional, Dict
from pathlib import Path
from uuid import uuid4

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

from ..config import get_settings

Base = declarative_base()


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String(64), primary_key=True)
    user_id = Column(String(128), index=True, nullable=True)
    title = Column(String(256), nullable=True)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_now, nullable=False)

    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(String(64), primary_key=True)
    conversation_id = Column(String(64), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_now, nullable=False)

    conversation = relationship("Conversation", back_populates="messages")


class UserMemory(Base):
    __tablename__ = "user_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(128), nullable=False, index=True)
    key = Column(String(128), nullable=False)
    value = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=_now, nullable=False)

    __table_args__ = (UniqueConstraint("user_id", "key", name="uq_user_memory"),)


class MemoryService:
    """Persists conversational history and summaries."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._session_factory = sessionmaker(bind=self._engine, expire_on_commit=False, future=True)
        Base.metadata.create_all(self._engine)

    @contextlib.contextmanager
    def _session(self) -> Generator[Session, None, None]:
        session: Session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_conversation(self, user_id: Optional[str] = None, title: Optional[str] = None) -> str:
        conversation_id = uuid4().hex
        self.ensure_conversation(conversation_id, user_id=user_id, title=title)
        return conversation_id

    def ensure_conversation(
        self,
        conversation_id: str,
        *,
        user_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        with self._session() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                conversation = Conversation(id=conversation_id, user_id=user_id, title=title)
                session.add(conversation)
            else:
                if user_id and not conversation.user_id:
                    conversation.user_id = user_id
                if title and not conversation.title:
                    conversation.title = title
                conversation.updated_at = _now()

    def set_title_if_absent(self, conversation_id: str, title: Optional[str]) -> None:
        if not title:
            return
        with self._session() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation and not conversation.title:
                conversation.title = title
                conversation.updated_at = _now()

    def append_message(self, conversation_id: str, role: str, content: str) -> None:
        with self._session() as session:
            conversation = session.get(Conversation, conversation_id)
            if conversation is None:
                raise ValueError(f"Conversation {conversation_id} does not exist")
            message = Message(id=uuid4().hex, conversation_id=conversation_id, role=role, content=content)
            session.add(message)
            conversation.updated_at = _now()

    def get_recent_messages(self, conversation_id: str, limit: int) -> List[Message]:
        if limit <= 0:
            return []
        with self._session() as session:
            stmt = (
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.desc())
                .limit(limit)
            )
            rows = session.execute(stmt).scalars().all()
        return list(reversed(rows))

    def list_conversations(self, user_id: Optional[str] = None) -> List[Conversation]:
        with self._session() as session:
            stmt = select(Conversation)
            if user_id:
                stmt = stmt.where(Conversation.user_id == user_id)
            stmt = stmt.order_by(Conversation.updated_at.desc())
            rows = session.execute(stmt).scalars().all()
        return rows

    def get_summary(self, conversation_id: str) -> Optional[str]:
        with self._session() as session:
            convo = session.get(Conversation, conversation_id)
            return convo.summary if convo else None

    def set_summary(self, conversation_id: str, summary: Optional[str]) -> None:
        with self._session() as session:
            convo = session.get(Conversation, conversation_id)
            if convo is None:
                raise ValueError(f"Conversation {conversation_id} does not exist")
            convo.summary = summary
            convo.updated_at = _now()

    def reset_conversation(self, conversation_id: str) -> None:
        with self._session() as session:
            convo = session.get(Conversation, conversation_id)
            if convo is None:
                return
            session.query(Message).filter(Message.conversation_id == conversation_id).delete()
            convo.summary = None
            convo.updated_at = _now()

    def delete_conversation(self, conversation_id: str) -> None:
        with self._session() as session:
            convo = session.get(Conversation, conversation_id)
            if convo:
                session.delete(convo)

    def conversation_exists(self, conversation_id: str) -> bool:
        with self._session() as session:
            return session.get(Conversation, conversation_id) is not None

    def get_message_count(self, conversation_id: str) -> int:
        with self._session() as session:
            count_stmt = select(func.count(Message.id)).where(Message.conversation_id == conversation_id)
            return int(session.execute(count_stmt).scalar_one())

    def iter_messages(self, conversation_id: str) -> List[Message]:
        with self._session() as session:
            stmt = (
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.asc())
            )
            return session.execute(stmt).scalars().all()

    # --- User key-value memory ---

    def set_user_memory(self, user_id: str, key: str, value: str) -> None:
        if not user_id:
            return
        with self._session() as session:
            stmt = select(UserMemory).where(UserMemory.user_id == user_id, UserMemory.key == key)
            record = session.execute(stmt).scalar_one_or_none()
            if record is None:
                session.add(UserMemory(user_id=user_id, key=key, value=value))
            else:
                record.value = value
                record.updated_at = _now()

    def get_user_memory(self, user_id: str, key: str) -> Optional[str]:
        if not user_id:
            return None
        with self._session() as session:
            stmt = select(UserMemory.value).where(UserMemory.user_id == user_id, UserMemory.key == key)
            result = session.execute(stmt).scalar_one_or_none()
            return result

    def get_all_user_memory(self, user_id: str) -> Dict[str, str]:
        if not user_id:
            return {}
        with self._session() as session:
            stmt = select(UserMemory).where(UserMemory.user_id == user_id)
            rows = session.execute(stmt).scalars().all()
            return {row.key: row.value for row in rows}


def _normalize_sqlite_dsn(dsn: str) -> str:
    if not dsn.startswith("sqlite///") and not dsn.startswith("sqlite:///"):
        return dsn
    # Normalize to absolute path rooted at the backend directory
    prefix = "sqlite:///"
    if not dsn.startswith(prefix):
        return dsn
    raw_path = dsn[len(prefix):]
    p = Path(raw_path)
    if p.is_absolute():
        full = p
    else:
        base_dir = Path(__file__).resolve().parents[2]  # <repo>/backend
        # If path redundantly starts with 'backend/', strip it to avoid backend/backend
        if raw_path.startswith("backend/"):
            raw_path = raw_path.split("backend/", 1)[1]
        full = base_dir / raw_path
    full.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{full}"


def _create_engine() -> Engine:
    settings = get_settings()
    dsn = settings.memory_dsn
    if dsn.startswith("sqlite"):
        dsn = _normalize_sqlite_dsn(dsn)
    connect_args = {}
    if dsn.startswith("sqlite"):
        connect_args = {"check_same_thread": False}
    try:
        engine = create_engine(dsn, future=True, echo=False, connect_args=connect_args)
    except SQLAlchemyError as exc:  # pragma: no cover - engine init failure
        raise RuntimeError(f"Failed to initialize memory database: {exc}") from exc
    return engine


def get_memory_service() -> Optional[MemoryService]:
    settings = get_settings()
    if not settings.enable_memory:
        return None
    if not hasattr(get_memory_service, "_instance"):
        engine = _create_engine()
        get_memory_service._instance = MemoryService(engine)
    return get_memory_service._instance  # type: ignore[attr-defined]
