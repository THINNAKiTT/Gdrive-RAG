"""
Unit tests for src/storage/chat_history.py (ChatHistoryStore)

Uses a real (ephemeral, tmp_path-backed) SQLite database rather than
mocking sqlite3 -- SQLite against a temp file is fast and gives real
confidence that the schema, migrations, and queries actually work,
without touching the developer's real chroma_db/chat_history.db.
"""
import pytest

from src.storage.chat_history import ChatHistoryStore

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path):
    return ChatHistoryStore(db_path=str(tmp_path / "chat_history.db"))


# ---------------------------------------------------------------------------
# Lazy session creation (add_message creates the session row)
# ---------------------------------------------------------------------------


def test_generate_pending_session_id_does_not_write_to_db(store):
    session_id = store.generate_pending_session_id()

    assert store.session_exists(session_id) is False
    assert store.list_sessions() == []


def test_add_message_lazily_creates_session(store):
    session_id = store.generate_pending_session_id()

    store.add_message(session_id, "user", "Hello")

    assert store.session_exists(session_id) is True
    sessions = store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["id"] == session_id


def test_add_message_on_existing_session_does_not_duplicate_session_row(store):
    session_id = store.create_session()

    store.add_message(session_id, "user", "First")
    store.add_message(session_id, "assistant", "Reply")

    assert len(store.list_sessions()) == 1


def test_pending_session_never_persisted_if_no_message_sent(store):
    """
    Regression guard for the original session-duplication bug: a
    session id that was generated but never used for a message must
    never show up in list_sessions().
    """
    store.generate_pending_session_id()
    store.generate_pending_session_id()
    store.generate_pending_session_id()

    assert store.list_sessions() == []


# ---------------------------------------------------------------------------
# create_session (explicit, immediate creation)
# ---------------------------------------------------------------------------


def test_create_session_persists_immediately(store):
    session_id = store.create_session()

    assert store.session_exists(session_id) is True


def test_create_session_default_title(store):
    session_id = store.create_session()

    sessions = store.list_sessions()
    assert sessions[0]["title"] == "New Chat"


def test_create_session_custom_title(store):
    session_id = store.create_session(title="My Custom Chat")

    sessions = store.list_sessions()
    assert sessions[0]["title"] == "My Custom Chat"


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


def test_add_message_rejects_invalid_role(store):
    session_id = store.create_session()

    with pytest.raises(ValueError):
        store.add_message(session_id, "system", "not allowed")


def test_get_messages_returns_in_chronological_order(store):
    session_id = store.create_session()
    store.add_message(session_id, "user", "First question")
    store.add_message(session_id, "assistant", "First answer")
    store.add_message(session_id, "user", "Second question")

    messages = store.get_messages(session_id)

    assert [m["content"] for m in messages] == [
        "First question",
        "First answer",
        "Second question",
    ]


def test_get_messages_only_returns_messages_for_that_session(store):
    """Session isolation: messages from one session must never leak
    into another session's history."""
    session_a = store.create_session()
    session_b = store.create_session()
    store.add_message(session_a, "user", "Message in A")
    store.add_message(session_b, "user", "Message in B")

    messages_a = store.get_messages(session_a)

    assert len(messages_a) == 1
    assert messages_a[0]["content"] == "Message in A"


def test_add_message_updates_session_updated_at(store):
    session_id = store.create_session()
    sessions_before = store.list_sessions()

    store.add_message(session_id, "user", "Hello")

    sessions_after = store.list_sessions()
    assert sessions_after[0]["updated_at"] >= sessions_before[0]["updated_at"]


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


def test_add_message_with_citations_round_trips(store):
    session_id = store.create_session()
    citations = ["[doc1.pdf (Page 1)](url1) - Score : 0.85", "[doc2.pdf (Page 3)](url2) - Score : 0.77"]

    store.add_message(session_id, "assistant", "Here's the answer.", citations=citations)

    messages = store.get_messages(session_id)
    assert messages[0]["citations"] == citations


def test_add_message_without_citations_returns_empty_list(store):
    session_id = store.create_session()

    store.add_message(session_id, "user", "Just a question, no citations expected.")

    messages = store.get_messages(session_id)
    assert messages[0]["citations"] == []


def test_add_message_with_empty_citations_list_returns_empty_list(store):
    session_id = store.create_session()

    store.add_message(session_id, "assistant", "No sources found.", citations=[])

    messages = store.get_messages(session_id)
    assert messages[0]["citations"] == []


# ---------------------------------------------------------------------------
# get_recent_turns (used by QueryRewriter)
# ---------------------------------------------------------------------------


def test_get_recent_turns_returns_all_when_fewer_than_max(store):
    session_id = store.create_session()
    store.add_message(session_id, "user", "Q1")
    store.add_message(session_id, "assistant", "A1")

    turns = store.get_recent_turns(session_id, max_turns=6)

    assert len(turns) == 2


def test_get_recent_turns_caps_at_max_turns(store):
    session_id = store.create_session()
    for i in range(10):
        store.add_message(session_id, "user", f"Q{i}")
        store.add_message(session_id, "assistant", f"A{i}")

    turns = store.get_recent_turns(session_id, max_turns=3)

    assert len(turns) == 6  # 3 turns = 6 messages


def test_get_recent_turns_keeps_most_recent_messages(store):
    session_id = store.create_session()
    for i in range(10):
        store.add_message(session_id, "user", f"Q{i}")
        store.add_message(session_id, "assistant", f"A{i}")

    turns = store.get_recent_turns(session_id, max_turns=1)

    assert [t["content"] for t in turns] == ["Q9", "A9"]


def test_get_recent_turns_empty_session_returns_empty_list(store):
    session_id = store.create_session()

    turns = store.get_recent_turns(session_id, max_turns=6)

    assert turns == []


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


def test_delete_session_removes_session(store):
    session_id = store.create_session()

    store.delete_session(session_id)

    assert store.session_exists(session_id) is False


def test_delete_session_removes_its_messages(store):
    session_id = store.create_session()
    store.add_message(session_id, "user", "Hello")

    store.delete_session(session_id)

    # get_messages on a deleted session should just come back empty,
    # not error.
    assert store.get_messages(session_id) == []


def test_delete_session_does_not_affect_other_sessions(store):
    session_a = store.create_session()
    session_b = store.create_session()
    store.add_message(session_a, "user", "A message")
    store.add_message(session_b, "user", "B message")

    store.delete_session(session_a)

    assert store.session_exists(session_b) is True
    assert len(store.get_messages(session_b)) == 1


# ---------------------------------------------------------------------------
# Rename
# ---------------------------------------------------------------------------


def test_rename_session_updates_title(store):
    session_id = store.create_session(title="Old Title")

    store.rename_session(session_id, "New Title")

    sessions = store.list_sessions()
    assert sessions[0]["title"] == "New Title"


# ---------------------------------------------------------------------------
# Pin / Unpin (ordering)
# ---------------------------------------------------------------------------


def test_set_pinned_true_marks_session_pinned(store):
    session_id = store.create_session()

    store.set_pinned(session_id, True)

    sessions = store.list_sessions()
    assert bool(sessions[0]["pinned"]) is True


def test_set_pinned_false_unmarks_session(store):
    session_id = store.create_session()
    store.set_pinned(session_id, True)

    store.set_pinned(session_id, False)

    sessions = store.list_sessions()
    assert bool(sessions[0]["pinned"]) is False


def test_pinned_sessions_sort_before_unpinned_regardless_of_recency(store):
    """
    A pinned session must always sort above an unpinned session, even
    if the unpinned one was updated more recently.
    """
    older_pinned = store.create_session(title="Pinned")
    newer_unpinned = store.create_session(title="Unpinned")
    store.set_pinned(older_pinned, True)
    # Touch the unpinned session's updated_at so it would normally sort
    # first by recency alone.
    store.add_message(newer_unpinned, "user", "keep this fresh")

    sessions = store.list_sessions()

    assert sessions[0]["id"] == older_pinned
    assert sessions[1]["id"] == newer_unpinned


def test_multiple_pinned_sessions_sort_by_recency_among_themselves(store):
    session_a = store.create_session(title="A")
    session_b = store.create_session(title="B")
    store.set_pinned(session_a, True)
    store.set_pinned(session_b, True)
    store.add_message(session_a, "user", "touch A last")  # A now more recent

    sessions = store.list_sessions()

    assert sessions[0]["id"] == session_a
    assert sessions[1]["id"] == session_b


# ---------------------------------------------------------------------------
# Schema migration (upgrading a pre-existing DB file)
# ---------------------------------------------------------------------------


def test_opening_existing_db_without_citations_column_migrates_cleanly(tmp_path):
    """
    Simulates a chat_history.db created before the citations/pinned
    columns existed -- ChatHistoryStore must ALTER TABLE them in
    without losing any existing data.
    """
    import sqlite3

    db_path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO sessions VALUES ('sess-1', 'Legacy Chat', '2026-01-01', '2026-01-01')"
    )
    conn.execute(
        "INSERT INTO messages VALUES ('msg-1', 'sess-1', 'user', 'old message', '2026-01-01')"
    )
    conn.commit()
    conn.close()

    store = ChatHistoryStore(db_path=db_path)

    # Old data survived the migration.
    sessions = store.list_sessions()
    assert sessions[0]["title"] == "Legacy Chat"
    messages = store.get_messages("sess-1")
    assert messages[0]["content"] == "old message"
    assert messages[0]["citations"] == []  # new column, defaults gracefully

    # New columns are now usable.
    store.set_pinned("sess-1", True)
    assert bool(store.list_sessions()[0]["pinned"]) is True