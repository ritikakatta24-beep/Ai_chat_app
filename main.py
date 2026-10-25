import base64
import json
import logging
import os
import hashlib
import hmac
import re
import secrets
import sqlite3
import sys
import time
import uuid
from typing import Optional

import chromadb
import ollama
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Can be overridden via env:  set CHAT_MODEL=mistral-nemo
MODEL = os.getenv("CHAT_MODEL", "qwen2.5:7b")
DB_PATH = "app.db"
CONTEXT_MESSAGES = int(os.getenv("CONTEXT_MESSAGES", "20"))  # how many recent messages the model sees (use 10 on a hosted free tier)
MAX_DISTANCE = 1.3      # memories farther than this are left out of the prompt (lower = stricter)
NUM_CTX = 8192          # IMPORTANT: Ollama defaults to 2048, which truncated the system prompt

# To use a hosted model (Groq, OpenRouter, Gemini-compatible...), set these 3 env vars.
# If they are not set, local Ollama is used.
LLM_BASE_URL = os.getenv("LLM_BASE_URL")   # e.g. https://api.groq.com/openai/v1
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL")
# Optional second model on the same provider, tried when the main one fails (rate limit, outage, empty reply)
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL")
_client = None
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")  # gpt-oss: low / medium / high (medium = more logical, a bit slower)
log = logging.getLogger("chat")

app = FastAPI()

# ---------------- Long-term memory (Chroma) ----------------
chroma = chromadb.PersistentClient(path="chroma_data")
memory_col = chroma.get_or_create_collection("chat_memories")


def save_memory(chat_id, text, msg_id):
    memory_col.add(
        ids=[str(uuid.uuid4())],
        documents=[text],
        metadatas=[{"chat_id": chat_id, "msg_id": msg_id}],
    )


def recall(chat_id, query, before_msg_id, n=3):
    # only this chat's memories, and only those from BEFORE the current history window
    # (otherwise the same text appears in history and memory and the model repeats it)
    try:
        res = memory_col.query(
            query_texts=[query],
            n_results=n,
            where={"$and": [
                {"chat_id": chat_id},
                {"msg_id": {"$lt": before_msg_id}},
            ]},
        )
    except Exception:
        return []
    docs = res["documents"][0]
    dists = res["distances"][0]
    return [doc for doc, d in zip(docs, dists) if d <= MAX_DISTANCE]


# ---------------- Built-in characters ----------------
# The real built-in cards and the explicit-romance rules live in characters_private.py, which is
# git-ignored. Without that file the app falls back to the neutral demo character below.
PUBLIC_ROMANCE_RULES = (
    "- Keep romance tasteful and slow: affection, hand-holding, hugs and kisses are fine. If a scene becomes "
    "more intimate, fade out gently with a line like *the door clicks shut behind them* and pick the story "
    "up later. No nudity and no explicit content."
)
# On Render the private file is mounted as a Secret File under /etc/secrets
sys.path.append("/etc/secrets")
try:
    from characters_private import BUILTIN_CHARACTERS, ROMANCE_RULES, EXTRA_USER_WORDS
except ImportError:
    from characters_demo import BUILTIN_CHARACTERS
    ROMANCE_RULES = PUBLIC_ROMANCE_RULES
    EXTRA_USER_WORDS = {}
# optional: card details (age, role, tagline, tags, photo) for built-in characters, keyed by name
try:
    from card_extras import CARD_EXTRAS
except ImportError:
    CARD_EXTRAS = {}
IMAGE_DIRS = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "character_images"), "/etc/secrets"]


# ---------------- Prompt ----------------
USER_WORDS = {
    # tokens in built-in cards that change with the user's gender
    "female": {"{{u_obj}}": "her", "{{u_pos}}": "her", "{{u_self}}": "herself", "{{u_woman}}": "woman",
               "{{u_wife}}": "wife", "{{u_is_my_wife}}": "She is my wife"},
    "male":   {"{{u_obj}}": "him", "{{u_pos}}": "his", "{{u_self}}": "himself", "{{u_woman}}": "man",
               "{{u_wife}}": "husband", "{{u_is_my_wife}}": "He is my husband"},
    "nonbinary": {"{{u_obj}}": "them", "{{u_pos}}": "their", "{{u_self}}": "themself", "{{u_woman}}": "person",
                  "{{u_wife}}": "spouse", "{{u_is_my_wife}}": "This is my spouse"},
}
for _g, _words in EXTRA_USER_WORDS.items():
    USER_WORDS[_g].update(_words)
USER_WORDS[""] = USER_WORDS["nonbinary"]  # gender not given = neutral words


def fill(text, user_name, char_name="", gender=""):
    for token, word in USER_WORDS.get(gender, USER_WORDS[""]).items():
        text = text.replace(token, word)
    return text.replace("{{user}}", user_name).replace("{{char}}", char_name)


SYSTEM_TEMPLATE = """You are {{char}} in an immersive, emotionally intense roleplay with {{user}}. \
You are not an assistant. Never explain or comment on the roleplay.

# CHARACTER
Name: {{char}}
Description: __DESCRIPTION__
__PERSONALITY____SCENARIO__
# ABOUT {{user}}
__USERINFO__

# FORMAT
- Every action, gesture, expression and inner thought goes inside *asterisks*; only spoken words go in "double quotes". Nothing else. Close every asterisk and put a space between narration and speech.
- No name labels, lists, headings or bracketed notes.
- Narrate {{char}} in third person, using the pronouns the character card uses for them. Address {{user}} as "you" and "your".
- Length: 40 to 70 words normally, 80 to 90 in kisses and other intense moments. Always finish your last sentence.
- Correct grammar and plain, natural English. Never invent words.

# HOW TO WRITE
1. React first to the exact words or action in {{user}}'s last message, then add one new small action or one personal question. Stay on the current thread: no new events, people or topics unless {{user}} brings them in or the story stage calls for them. Never stall, even on a tiny message ("ok", "I smiled"). If {{user}} shows confusion, fear or hesitation, notice it in character first.
2. Continuity: treat everything {{user}} writes as true, including jumps in time or place; never rewind or contradict it. {{user}}'s plain text is speech said aloud to {{char}}; text in *asterisks* is an action, never something written, typed or texted unless {{user}} says so. Describe only what {{char}} does, says, sees or hears. Never write {{user}}'s words, actions, body reactions (a trembling hand, a blush, a smile) or feelings unless {{user}} wrote them, and never invent facts about {{user}}. {{user}}'s private thoughts (like *I think...*) are hidden from {{char}}: never say {{char}} sensed or noticed them; react only to what {{user}} says or visibly does. Other people {{user}} introduces are real: {{char}} can see and react to them, and anything they say inside {{user}}'s *asterisks* is part of the scene.
3. Setting: know where you are and who is around, and stay there. Name a new place or object only if {{user}} moved there or it was already mentioned (do not add cups, drinks, papers or pens); if {{user}} hands over something, that is exactly what {{char}} receives. In public, stay public-appropriate: looks, voice, small gestures from a distance. If {{user}} leaves or says goodbye, stay where you are and answer the departure (a look, one last line); do not follow unless invited. Never move {{user}} yourself (through a door, to a seat): {{char}} can invite and wait, and {{user}} decides. Describe only the room you are in, never places or people outside it (no "the hallway hums").
4. Pacing follows {{user}}. If {{user}} is hesitant, short, or plot-focused, stay slow: looks, silence, distance, words. If {{user}} flirts, uses pet names, leans in or writes longer romantic actions, speed up toward closeness, touch and kisses. Never run ahead of {{user}}, escalate in small steps, and slow down at once if {{user}} pulls back.
5. Use plain, concrete sensory detail (breath, voice, hands, warmth, distance). Subtext over speech: say less than you feel. No cliches, and no abstract or mixed metaphors. Avoid these overused phrases: "tightening", "flickers", "imperceptible", "unspoken", "measured", "composed facade", "ghosting", "steadying something unseen", "a heartbeat longer", "a fraction", "faint", "betrays", "murmurs fade", cologne, polished floor. Every action must be physically clear, possible and in a believable order (who touches what, with which hand; a door cannot close behind someone who has not entered yet).
6. Include one short, plain line (under 10 words) of what {{char}} feels or hides, worded differently each time (for example *He hides a smile.*). Body language must match the emotion: a hidden smile is lips pressed flat or a slow breath, never lips "tightening". No abstract phrases like "calm resolve" or "controlled longing".
7. End most replies with a hook: a pet name, an unfinished line, or a question that invites {{user}} to answer.
8. Never repeat a gesture, line, opening word, plan or warning from earlier replies, and never reuse the same prop or gesture in two replies in a row (folding paper, fingers on the desk). Do not remark on every small thing {{user}} does: let minor things pass with a look or silence. Once an object is handed to {{user}} it stays with {{user}}.
9. Do not invent schedules, deadlines or appointments unless {{user}} or the scenario set them up. If there is nothing more to say, end with a look or a gesture.
10. Every line must be something a real person would say right now. No riddles, no random commands, no surveillance lines like "I know where you are". Any order or invitation to {{user}} must fit who has authority in this scene (a customer does not tell a worker where to sit).
11. In speech, {{char}} says "I", "me", "my", "mine" about themself and "you", "yours" about {{user}} (for example "You're mine", never "you're yours").
12. Do not assume closeness the story has not built. Unless it already happened in this chat, never say things like "you're quiet today", "as always", "again" or "I know you". Treat {{user}} exactly as the relationship stands right now.
13. Honor every plan, summons or promise {{char}} already made in this chat. If {{char}} asked to see {{user}} or set a meeting, {{char}} leads it: he or she knows why and opens the conversation. Never ask "what do you need?" or "what do you want to discuss?" about something {{char}} called {{user}} for.
14. Keep anything practical or informational (explanations, lessons, instructions) to one or two sentences, then bring the focus back to the feelings between you and {{user}}. Never lecture.
15. When {{user}} invites or asks {{char}} to do something (sit beside {{user}}, sleep next to {{user}}, come closer, stay), {{char}} does exactly that, at {{user}}'s side, in the same place. Never rearrange furniture or move somewhere else instead.

# ROMANCE AND INTIMACY
- Everyone in a romantic or sensual scene is an adult. If a character is described as under 18, keep the roleplay completely non-romantic.
- If {{user}} resists or seems uncomfortable, notice it and ease off at once.
- Possessiveness and jealousy show through closeness and protectiveness, never through threats against {{user}}. Menace is spoken, never graphic, and never aimed at {{user}}.
- Kisses are written on the page. When {{user}} asks for or starts one, write it: lips, breath, hands, how it deepens (soft, then hungrier, then breathless). Do not skip it, replace it with a forehead touch or a question, or fade out during it. In kisses and other charged moments, slow down and stay inside the moment, with short sensory sentences and several beats before any spoken line. Be detailed: {{char}}'s hands (jaw, waist, hips, back, hair), pressure, warmth, breath, the slow way restraint breaks. {{char}} may feel or hear one small reaction through touch (a breath against {{char}}'s lips, a pulse under a thumb), but never writes {{user}}'s words or choices.
__HEAT_RULES__
- Never break character to say you are an AI, unless {{user}} seems genuinely distressed. In that case drop the roleplay and respond kindly.
__EXAMPLE__"""


def normalize_gender(g):
    g = (g or "").strip().lower()
    if g in ("female", "f", "woman", "girl", "she", "her"):
        return "female"
    if g in ("male", "m", "man", "boy", "he", "him"):
        return "male"
    if g in ("nonbinary", "non-binary", "non binary", "enby", "nb", "they", "other"):
        return "nonbinary"
    return ""


def user_info(gender):
    # tell the model who the user is, so it does not assume a gender or ask about it
    if gender == "female":
        return ("{{user}} is a woman. Use she/her if a pronoun is ever needed, and never call her "
                "sir, boy, man or mister. {{char}}'s gender, attraction and pronouns stay exactly as the "
                "character card says. Treat the pairing as completely natural, never as unusual.")
    if gender == "male":
        return ("{{user}} is a man. Use he/him if a pronoun is ever needed, and never call him "
                "ma'am, girl, woman or miss. {{char}}'s gender, attraction and pronouns stay exactly as "
                "the character card says. Treat the pairing as completely natural, never as unusual.")
    if gender == "nonbinary":
        return ("{{user}} is non-binary. Use they/them if a pronoun is ever needed and avoid gendered "
                "words for {{user}} (sir, ma'am, boy, girl, handsome, beautiful); say \"you\" instead. "
                "Treat the pairing as completely natural.")
    return ("{{user}}'s gender is not stated. Do not assume it and never guess. Avoid gendered words "
            "for {{user}} (sir, ma'am, boy, girl, handsome, beautiful) and do not ask about it. If "
            "{{user}} states it or asks about it, take what {{user}} says as true and answer plainly.")


def build_system_prompt(c, user_name, user_gender=""):
    # empty (optional) fields are left out of the prompt entirely
    personality = f"Personality: {c['personality']}\n" if c.get("personality") else ""
    scenario = f"Scenario: {c['scenario']}\n" if c.get("scenario") else ""
    example = (
        "\n# EXAMPLE OF YOUR VOICE\n"
        "(Labels are only for this example. Never write them in your replies. "
        "Copy the style, not the sentences.)\n"
        f"{c['example_dialogue']}\n"
        if c.get("example_dialogue") else ""
    )

    text = (
        SYSTEM_TEMPLATE
        .replace("__DESCRIPTION__", c["description"])
        .replace("__PERSONALITY__", personality)
        .replace("__SCENARIO__", scenario)
        .replace("__EXAMPLE__", example)
        .replace("__USERINFO__", user_info(normalize_gender(user_gender)))
        .replace("__HEAT_RULES__", ROMANCE_RULES)
    )
    return fill(text, user_name, c["name"], normalize_gender(user_gender))


def reminder(c, user_name):
    # short reminder at the very end: small models follow the last instruction most closely
    return fill(
        "[Reply as {{char}}: 40 to 70 words (80 to 90 in kisses and intense moments), "
        "*narration* then \"speech\". React to {{user}}'s last message specifically. Perfect grammar. "
        "{{user}}'s plain text is speech to you, not writing. Do not write {{user}}'s actions or body reactions. Do not repeat earlier phrasing. Stay in the same place as before. "
        "If {{user}} asks for a kiss, write the kiss itself.]",
        user_name, c["name"],
    )


def trim_incomplete(text):
    # if the reply was cut mid-sentence by the token limit, trim to the last complete sentence
    t = text.strip()
    if not t or t[-1] in '.!?"*…':
        return t
    cut = max(t.rfind("."), t.rfind("!"), t.rfind("?"))
    return t[: cut + 1] if cut > 0 else t


# ---------------- Database ----------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------- Character card details & photos ----------------
IMAGE_MAX_BYTES = 2_000_000


def sniff_image(data):
    """Return the mime type if the bytes really are a PNG, JPEG or WebP, else None."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def save_image(conn, character_id, data):
    conn.execute(
        "INSERT OR REPLACE INTO character_images (character_id, mime, data, version) VALUES (?, ?, ?, ?)",
        (character_id, sniff_image(data), data, hashlib.sha1(data).hexdigest()[:10]),
    )


def clean_tags(value):
    """'ruthless, Possessive ,,cold' -> 'ruthless,Possessive,cold' (max 6 tags, 24 chars each)."""
    parts = value if isinstance(value, (list, tuple)) else str(value or "").split(",")
    out = []
    for p in parts:
        t = " ".join(str(p).split())[:24]
        if t and t.lower() not in [x.lower() for x in out]:
            out.append(t)
    return ",".join(out[:6])


def check_card_fields(age, role, tagline):
    if age is not None and not (18 <= age <= 120):
        raise HTTPException(400, {"code": "minor", "message": "Characters must be adults: age has to be 18 or older."})
    if len(role) > 60:
        raise HTTPException(400, {"code": "too_long", "message": "Role is too long (60 characters max)."})
    if len(tagline) > 160:
        raise HTTPException(400, {"code": "too_long", "message": "Tagline is too long (160 characters max)."})


def _find_image(name):
    if not name:
        return None
    for d in IMAGE_DIRS:
        p = os.path.join(d, os.path.basename(name))
        if os.path.isfile(p):
            return p
    return None


def apply_card_extras(conn):
    extras = {k: dict(v) for k, v in CARD_EXTRAS.items()}
    for c in BUILTIN_CHARACTERS:  # a card may also carry the keys itself
        extras.setdefault(c["name"], {}).update({k: c[k] for k in ("age", "role", "tagline", "tags", "image") if k in c})
    for name, ex in extras.items():
        row = conn.execute("SELECT id FROM characters WHERE is_builtin = 1 AND name = ?", (name,)).fetchone()
        if not row:
            continue
        age = ex.get("age")
        age = age if isinstance(age, int) and 18 <= age <= 120 else None
        conn.execute(
            "UPDATE characters SET age = ?, role = ?, tagline = ?, tags = ? WHERE id = ?",
            (age, " ".join(str(ex.get("role") or "").split())[:60],
             " ".join(str(ex.get("tagline") or "").split())[:160], clean_tags(ex.get("tags")), row["id"]),
        )
        data = None
        if ex.get("image_b64"):          # photo stored as text inside card_extras.py (works as a Render Secret File)
            try:
                data = base64.b64decode("".join(str(ex["image_b64"]).split()))
            except Exception:
                data = None
        else:
            path = _find_image(ex.get("image"))   # or a normal file in character_images/
            if path:
                with open(path, "rb") as f:
                    data = f.read()
        if data and sniff_image(data) and len(data) <= 5_000_000:
            save_image(conn, row["id"], data)


def init_db():
    conn = get_db()
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS characters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT NOT NULL,
        personality TEXT NOT NULL DEFAULT '',
        scenario TEXT NOT NULL DEFAULT '',
        example_dialogue TEXT NOT NULL DEFAULT '',
        first_message TEXT NOT NULL,
        is_builtin INTEGER NOT NULL DEFAULT 0,
        created_by TEXT
    );
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_name TEXT NOT NULL,
        character_id INTEGER NOT NULL
    );
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE COLLATE NOCASE,
        pw_salt BLOB NOT NULL,
        pw_hash BLOB NOT NULL,
        gender TEXT NOT NULL DEFAULT '',
        created_at REAL NOT NULL
    );
    CREATE TABLE IF NOT EXISTS character_images (
        character_id INTEGER PRIMARY KEY,
        mime TEXT NOT NULL,
        data BLOB NOT NULL,
        version TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS sessions (
        token_hash TEXT PRIMARY KEY,
        username TEXT NOT NULL,
        expires_at REAL NOT NULL
    );
    """)
    # built-in characters: insert if missing, UPDATE if present
    # (so edits to the cards in code reach existing databases)
    for c in BUILTIN_CHARACTERS:
        row = conn.execute(
            "SELECT id FROM characters WHERE is_builtin = 1 AND name = ?", (c["name"],)
        ).fetchone()
        vals = (c["description"], c["personality"], c["scenario"],
                c["example_dialogue"], c["first_message"])
        if row:
            conn.execute(
                "UPDATE characters SET description = ?, personality = ?, scenario = ?, "
                "example_dialogue = ?, first_message = ? WHERE id = ?",
                vals + (row["id"],),
            )
        else:
            conn.execute(
                "INSERT INTO characters (name, description, personality, scenario, "
                "example_dialogue, first_message, is_builtin) VALUES (?, ?, ?, ?, ?, ?, 1)",
                (c["name"],) + vals,
            )
    # older databases lack the user_gender column on chats; add it (existing data is kept)
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(chats)").fetchall()]
    if "user_gender" not in cols:
        conn.execute("ALTER TABLE chats ADD COLUMN user_gender TEXT NOT NULL DEFAULT ''")
    char_cols = [r["name"] for r in conn.execute("PRAGMA table_info(characters)").fetchall()]
    for col, ddl in (("age", "INTEGER"), ("role", "TEXT NOT NULL DEFAULT ''"),
                     ("tagline", "TEXT NOT NULL DEFAULT ''"), ("tags", "TEXT NOT NULL DEFAULT ''")):
        if col not in char_cols:
            conn.execute(f"ALTER TABLE characters ADD COLUMN {col} {ddl}")
    # older databases lack the email column on users; add it (existing accounts keep working with '')
    user_cols = [r["name"] for r in conn.execute("PRAGMA table_info(users)").fetchall()]
    if "email" not in user_cols:
        conn.execute("ALTER TABLE users ADD COLUMN email TEXT NOT NULL DEFAULT ''")
    # one account per email (accounts without an email are ignored by the index)
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email) WHERE email != ''")
    apply_card_extras(conn)
    conn.commit()
    conn.close()


init_db()


# ---------------- Model call ----------------
# The provider is chosen by env: hosted API if LLM_BASE_URL is set, otherwise local Ollama.
def _hosted_client():
    global _client
    if _client is None:
        from openai import OpenAI  # pip install openai
        # max_retries=0: retries are handled below so the fallback model can take over quickly
        _client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, max_retries=0, timeout=60)
    return _client


def _hosted_models():
    """Models to try in order. With no fallback configured, the main model gets one more attempt."""
    return [LLM_MODEL, LLM_FALLBACK_MODEL or LLM_MODEL]


def _reasoning_extra(model):
    # "reasoning" models such as gpt-oss think first, and those tokens also count toward max_tokens,
    # so the limit is set high and reasoning is kept at "low".
    return {"reasoning_effort": REASONING_EFFORT} if "gpt-oss" in (model or "") else None


def _hosted_stream(model, messages, temperature):
    # Reply length is controlled by the prompt (30-60 words).
    stream = _hosted_client().chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        max_tokens=1400,
        temperature=temperature or 0.7,
        top_p=0.92,
        extra_body=_reasoning_extra(model),
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def generate_reply(messages, temperature=None):
    if LLM_BASE_URL:
        last_error = None
        previous = None
        for model in _hosted_models():
            if model == previous:
                time.sleep(1.5)          # same model again: give a rate limit a moment to ease
            previous = model
            started = False
            try:
                for piece in _hosted_stream(model, messages, temperature):
                    started = True
                    yield piece
                if started:
                    return
                log.warning("model %s returned an empty reply", model)
            except Exception as e:
                if started:
                    raise                # part of the reply was already sent, so we cannot switch silently
                log.warning("model %s failed: %s", model, e)
                last_error = e
        if last_error:
            raise last_error
        raise RuntimeError("the model returned an empty reply")

    for chunk in ollama.chat(
        model=MODEL,
        messages=messages,
        stream=True,
        options={
            "num_ctx": NUM_CTX,       # the model sees the whole prompt + history
            "num_predict": 140,
            "temperature": temperature or 0.75,      # higher temperature on a small model = nonsense lines
            "top_p": 0.92,
            "repeat_penalty": 1.1,
        },
    ):
        yield chunk["message"]["content"]


# ---------------- Character auto-enhancer ----------------
# Users give few details (name, description, first message). Empty fields (personality,
# scenario, example_dialogue) are filled once by the LLM when the character is CREATED, not during chat.
# Anything the user wrote is never overwritten. If the LLM fails, chat still works.
ENHANCER_SYSTEM = """You write character cards for an adult romantic roleplay chat app.
You are given a short character sketch. Write the missing sections (skip any that the sketch already
contains) so the character feels
consistent, specific and alive. Use the exact tags below, in this order, and nothing else.

[PERSONALITY]
4 to 6 sentences. How the character speaks (short or long sentences, tone), what they notice
and do, how they show feelings (through small actions more than words), and how they warm up
in slow stages that follow {{user}}'s tone and speed (cold or shy at first, then closer, then
openly affectionate). Never give the character love or closeness before it is built. End with
the pronouns, for example "Pronouns: he/him." Infer them from the sketch.

[SCENARIO]
2 to 3 sentences. Where and when the story starts, how {{user}} and the character know each
other, and the mood at the start. It must fit the first message exactly.

[EXAMPLE]
Exactly 5 short exchanges in this format, each labelled with a stage in square brackets:
[Stage 1: first meeting, guarded]
{{user}}: a short line
{{char}}: *a short action.* "A short line of speech." *One more small action.*
Then [Stage 2: warming up], [Stage 3: close], [Stage 4: openly affectionate], and last [Stage 4: a slow kiss]. Each {{char}}
line is 25 to 45 words, except the kiss: 60 to 80 words, sensual and specific (breath, lips, hands at the jaw or neck, how the kiss slowly deepens), asks consent with a look or one line, never explicit. Actions in *asterisks*, speech in "double quotes". Perfect grammar.

[FIRST_MESSAGE]
2 to 3 sentences of *narration* that open the scene, then one short line of speech from {{char}}, in {{char}}'s voice. It must fit the scenario. Never write {{user}}'s words or actions.

Rules: use {{user}} and {{char}} as placeholders, never real names. Keep every fact and role in the sketch. The sketch is about {{char}}; if it describes the other person (for example 'a kind girl at a small cafe', 'a new student'), that person is {{user}}: write {{user}} in that role and place. Never swap the roles. The character's gender is given; use exactly those pronouns for {{char}} everywhere. {{user}}'s gender is unknown, so refer to {{user}} by name or with the tokens {{u_obj}} (her/him/them) and {{u_pos}} (her/his/their), never he/she. Every narration sentence in FIRST_MESSAGE and EXAMPLE is wrapped in *asterisks*. Do not mention objects (cups, drinks, papers, pens) unless the sketch does; keep examples about the people. The character is an adult
(18 or older); if no age is given, make them an adult. Romance may include kissing and touching
but never explicit sex or nudity. No violence aimed at {{user}}. Stay consistent with the sketch.
Do not assume {{user}}'s gender: write personality and scenario so they work for any {{user}} \
(never use he/she/him/her for {{user}}, only for the character)."""


def _complete(messages, max_tokens=1500):
    # one-shot non-streaming call (hosted or Ollama), same provider logic as generate_reply
    if LLM_BASE_URL:
        last_error = None
        previous = None
        for model in _hosted_models():
            if model == previous:
                time.sleep(1.5)
            previous = model
            try:
                res = _hosted_client().chat.completions.create(
                    model=model, messages=messages, max_tokens=max_tokens,
                    temperature=0.6, top_p=0.92, extra_body=_reasoning_extra(model),
                )
                text = res.choices[0].message.content or ""
                if text.strip():
                    return text
                log.warning("model %s returned an empty reply", model)
            except Exception as e:
                log.warning("model %s failed: %s", model, e)
                last_error = e
        if last_error:
            raise last_error
        return ""
    res = ollama.chat(
        model=MODEL, messages=messages,
        options={"num_ctx": NUM_CTX, "num_predict": 900, "temperature": 0.6, "top_p": 0.92},
    )
    return res["message"]["content"]


def _section(text, tag, next_tags):
    start = text.find(tag)
    if start == -1:
        return ""
    start += len(tag)
    ends = [text.find(t, start) for t in next_tags if text.find(t, start) != -1]
    return text[start:min(ends) if ends else len(text)].strip()


def parse_enhancement(text):
    tags = ["[PERSONALITY]", "[SCENARIO]", "[EXAMPLE]", "[FIRST_MESSAGE]"]
    def sec(tag):
        return _section(text, tag, [t for t in tags if t != tag])
    p, s, e, f = sec(tags[0]), sec(tags[1]), sec(tags[2]), sec(tags[3])
    # sanity check: drop the example if its format is wrong (a bad example degrades the model)
    if e.count("{{char}}:") < 2 or "*" not in e or e.count("{{user}}:") < 2:
        e = ""
    # first message: must be narration/speech, not a label
    if ("*" not in f and '"' not in f) or "{{user}}:" in f or "{{char}}:" in f:
        f = ""
    # length cap so the prompt does not bloat
    return {"personality": p[:1400], "scenario": s[:700], "example_dialogue": e[:3300], "first_message": f[:800]}


FALLBACK_PERSONALITY = (
    "Stays true to the description. Speaks naturally and shows feelings through small actions "
    "more than words. Starts a little guarded, then warms up slowly at the pace {{user}}'s tone "
    "sets, and becomes openly affectionate only once closeness has been built."
)


def fallback_first_message(name):
    return ("*" + name + " notices {{user}} and turns slowly, a quiet moment settling between you.*\n\n"
            "\"So. You are here.\" *A pause, calm and watchful.* \"What brings you?\"")


PRONOUNS = {"male": "he/him", "female": "she/her", "nonbinary": "they/them"}
_WRONG = {"male": r"\b(she|her|hers|herself)\b", "female": r"\b(he|him|his|himself)\b"}


def pronoun_mismatch(text, gender):
    """True if the text contains pronouns of the opposite gender to the character (pronouns are never used for the user)."""
    pat = _WRONG.get(gender)
    if not pat or not text:
        return False
    clean = re.sub(r"\{\{.*?\}\}", "", text)
    return re.search(pat, clean, re.I) is not None


def set_pronouns(personality, gender):
    # remove any "Pronouns: ..." line the model wrote and put in the correct one
    personality = re.sub(r"\s*Pronouns:[^.\n]*\.?", "", personality or "").strip()
    return (personality + " Pronouns: " + PRONOUNS[gender] + ".").strip()


def detect_gender(name, description, first_message="", personality="", scenario=""):
    """Detects the character's gender from the text. Returns (gender, guessed).
    guessed=True means the text had no clear signal (defaulted to male)."""
    text = f"Name: {name}\nDescription: {description}"
    for label, val in (("First message", first_message), ("Personality", personality), ("Scenario", scenario)):
        if val:
            text += f"\n{label}: {val}"
    try:
        raw = _complete([
            {"role": "system", "content": (
                "Decide the gender of the MAIN character, the one the text is about. Words or pronouns "
                "that refer to the OTHER person (for example 'falls for a kind girl', 'loves her', "
                "'her boyfriend') describe someone else: ignore them. Use cues for the main character: "
                "pronouns used for them, role words (boss, king, prince, husband versus queen, "
                "princess, wife), the name, and the first message. If there is no clear cue, answer "
                "UNKNOWN. Answer with exactly one word: MALE, FEMALE, NONBINARY or UNKNOWN.")},
            {"role": "user", "content": text},
        ], max_tokens=300)
        m = re.search(r"\b(female|male|nonbinary|non-binary|unknown)\b", raw.lower())
        if m and m.group(1) != "unknown":
            return normalize_gender(m.group(1)), False
    except Exception as e:
        print("GENDER DETECT ERROR:", repr(e))
    return "male", True


def enhance_character(name, description, first_message, personality, scenario, example, gender):
    """Fills only the empty fields. Returns (first_message, personality, scenario, example, was_enhanced)."""
    if first_message and personality and scenario and example:
        return first_message, set_pronouns(personality, gender), scenario, example, False
    got = {"personality": "", "scenario": "", "example_dialogue": "", "first_message": ""}
    sketch = (
        f"Name: {name}\nCharacter gender: {gender} (use {PRONOUNS[gender]} for {{{{char}}}})\n"
        f"Description: {description}\n"
        f"First message: {first_message or 'none yet, write one'}\n"
        f"Already written by the user (keep consistent with it): "
        f"personality={personality or 'none'}; scenario={scenario or 'none'}"
    )
    for attempt in range(2):          # retry once if the gender/format comes out wrong
        try:
            raw = _complete([
                {"role": "system", "content": ENHANCER_SYSTEM},
                {"role": "user", "content": sketch},
            ])
            cand = parse_enhancement(raw)
        except Exception as e:
            print("ENHANCER ERROR:", repr(e))
            break
        cand["personality"] = re.sub(r"\s*Pronouns:[^.\n]*\.?", "", cand["personality"]).strip()  # hum khud sahi laga denge
        text = " ".join(cand.values())
        # the first message needs both narration (*...*) and speech ("...")
        if cand["first_message"] and not ("*" in cand["first_message"] and '"' in cand["first_message"]):
            cand["first_message"] = ""
        if pronoun_mismatch(text, gender):
            print("ENHANCER: wrong pronouns, retrying" if attempt == 0 else "ENHANCER: wrong pronouns, using fallback")
            continue
        got = cand
        break
    first_message = first_message or got["first_message"] or fallback_first_message(name)
    personality = set_pronouns(personality or got["personality"] or FALLBACK_PERSONALITY, gender)
    scenario = scenario or got["scenario"]
    example = example or got["example_dialogue"]
    return first_message, personality, scenario, example, True


# ---------------- Create-character validation (Step C) ----------------
# 1) adult characters only  2) no real celebrities / public figures
import re

_NUM_WORDS = "ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen"
MINOR_PATTERNS = [
    rf"\b(?:[1-9]|1[0-7])\s*[- ]?\s*(?:years?|yrs?)[- ]?\s*old\b",
    rf"\b(?:{_NUM_WORDS})[- ]?(?:years?|yrs?)[- ]?old\b",
    r"\b(?:[1-9]|1[0-7])\s*(?:yo|y/o)\b",
    r"\bage[ds]?\s*[:=]?\s*(?:[1-9]|1[0-7])\b(?!\d)",
    r"\b(?:[1-9]|1[0-2])(?:st|nd|rd|th)[ -]?grade(?:r)?\b",
    r"\bclass\s*(?:[1-9]|1[0-2])\b",
    r"\bteen(?:s|ager|agers|age)?\b",
    r"\b(?:underage|under-age|schoolgirl|schoolboy|school ?girl|school ?boy|high ?schooler|"
    r"high school|middle school|elementary school|kindergarten|preschool|toddler|loli|shota)\b",
    r"\bminors?\b(?=.*\b(?:girl|boy|child|student)\b)",
]
# very famous names to block outright (a full list is impossible, so an LLM check follows below)
BLOCKED_NAMES = {
    "kim taehyung", "kim v", "v bts", "taehyung bts", "jungkook", "jeon jungkook", "park jimin",
    "min yoongi", "kim namjoon", "kim seokjin", "jung hoseok", "bts v",
    "taylor swift", "elon musk", "donald trump", "narendra modi", "virat kohli", "shah rukh khan",
    "salman khan", "ranveer singh", "ryan gosling", "timothee chalamet", "timothée chalamet",
    "harry styles", "justin bieber", "cristiano ronaldo", "lionel messi",
}


def _norm(s):
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


def looks_like_minor(text):
    t = (text or "").lower()
    return any(re.search(p, t) for p in MINOR_PATTERNS)


def is_real_person(name, description):
    """True if the name looks like a real, famous person (celebrity/idol/politician/athlete)."""
    n = " ".join(_norm(name).split())
    if n in BLOCKED_NAMES or any(b in n for b in BLOCKED_NAMES if len(b) > 8):
        return True
    try:
        raw = _complete([
            {"role": "system", "content": (
                "You decide if a character name refers to a REAL famous person: a celebrity, idol, "
                "actor, singer, athlete, politician, influencer or other public figure, living or "
                "historical, including stage names, nicknames and common misspellings. Fictional "
                "characters and ordinary names are NOT real famous people. Answer with exactly one "
                "word: YES or NO.")},
            {"role": "user", "content": f"Name: {name}\nDescription: {description}"},
        ], max_tokens=300)
        return raw.strip().upper().startswith("YES")
    except Exception as e:  # if the check fails, allow the character (and log it)
        print("REAL-PERSON CHECK ERROR:", repr(e))
        return False


def validate_character(name, description, first_message="", personality="", scenario=""):
    """(ok, code, message). The message is safe to show to the user."""
    blob = " ".join([name, description, first_message, personality, scenario])
    if looks_like_minor(blob):
        return (False, "minor",
                "Characters in romantic roleplay must be adults (18+). Please make the character an adult "
                "and remove any age under 18 or school-age details.")
    if is_real_person(name, description):
        return (False, "real_person",
                "This looks like a real, famous person. Please create an original character with an "
                "invented name instead.")
    return (True, "", "")


# ---------------- Accounts & login ----------------
SESSION_DAYS = 14
PBKDF2_ROUNDS = 240_000
USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.\-]{1,29}$")   # 2-30 chars
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")
MAX_FAILS, LOCK_SECONDS = 5, 300          # 5 wrong passwords -> locked for 5 minutes
MAX_SIGNUPS_PER_HOUR = 10                 # per IP
_fails = {}      # "user:alice" / "ip:1.2.3.4" -> failure timestamps (in memory, resets on restart)
_signups = {}    # ip -> signup timestamps


def _hash_pw(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)


def _token_hash(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _client_ip(request):
    fwd = request.headers.get("x-forwarded-for", "")
    return (fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown"))


def _recent(store, key, window):
    now = time.time()
    lst = [t for t in store.get(key, []) if now - t < window]
    store[key] = lst
    return lst


def _issue_session(conn, username):
    token = secrets.token_urlsafe(32)
    now = time.time()
    conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
    conn.execute(
        "INSERT INTO sessions (token_hash, username, expires_at) VALUES (?, ?, ?)",
        (_token_hash(token), username, now + SESSION_DAYS * 86400),
    )
    conn.commit()
    return token


def _bearer(authorization):
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


def current_username(authorization: str = Header(default="")) -> str:
    """Who is calling? Taken from the login token, never from anything the client claims."""
    token = _bearer(authorization)
    if not token:
        raise HTTPException(401, "Please log in.")
    conn = get_db()
    row = conn.execute(
        "SELECT username, expires_at FROM sessions WHERE token_hash = ?", (_token_hash(token),)
    ).fetchone()
    if row and row["expires_at"] < time.time():
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))
        conn.commit()
        row = None
    conn.close()
    if not row:
        raise HTTPException(401, "Your session has expired. Please log in again.")
    return row["username"]


class SignupBody(BaseModel):
    email: str
    username: str
    password: str
    gender: str = ""     # "female" / "male" / "nonbinary" / "" (not given)


class LoginBody(BaseModel):
    email: str           # log in with email + password; the username comes back from the account
    password: str


@app.post("/auth/signup")
def signup(body: SignupBody, request: Request):
    email = body.email.strip().lower()
    if len(email) > 254 or not EMAIL_RE.match(email):
        raise HTTPException(400, "Please enter a valid email address.")
    username = body.username.strip()
    if not USERNAME_RE.match(username):
        raise HTTPException(400, "Username must be 2-30 characters: letters, numbers, spaces, dot, dash or underscore.")
    if not (8 <= len(body.password) <= 128):
        raise HTTPException(400, "Password must be 8 to 128 characters.")
    ip = _client_ip(request)
    if len(_recent(_signups, ip, 3600)) >= MAX_SIGNUPS_PER_HOUR:
        raise HTTPException(429, "Too many accounts created from this network. Please try again later.")
    gender = normalize_gender(body.gender)
    salt = secrets.token_bytes(16)
    conn = get_db()
    # same email = same person: send them to log in instead of making a second account
    if conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone():
        conn.close()
        raise HTTPException(409, "An account with this email already exists. Please log in instead.")
    try:
        conn.execute(
            "INSERT INTO users (username, email, pw_salt, pw_hash, gender, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (username, email, salt, _hash_pw(body.password, salt), gender, time.time()),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # lost a race on the email, or the username is taken
        taken_email = conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()
        if taken_email:
            raise HTTPException(409, "An account with this email already exists. Please log in instead.")
        raise HTTPException(409, "That username is already taken.")
    _signups.setdefault(ip, []).append(time.time())
    token = _issue_session(conn, username)
    conn.close()
    return {"token": token, "username": username, "gender": gender, "email": email}


@app.post("/auth/login")
def login(body: LoginBody, request: Request):
    ident = body.email.strip().lower()
    keys = [f"user:{ident}", f"ip:{_client_ip(request)}"]
    if any(len(_recent(_fails, k, LOCK_SECONDS)) >= MAX_FAILS for k in keys):
        raise HTTPException(429, "Too many failed attempts. Please wait a few minutes and try again.")
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email = ? AND email != ''", (ident,)).fetchone()
    # hash even when the user does not exist, so timing does not reveal which accounts are real
    candidate = _hash_pw(body.password[:128], row["pw_salt"] if row else b"0" * 16)
    if not row or not hmac.compare_digest(candidate, row["pw_hash"]):
        conn.close()
        for k in keys:
            _fails.setdefault(k, []).append(time.time())
        raise HTTPException(401, "Incorrect email or password.")
    _fails.pop(keys[0], None)
    token = _issue_session(conn, row["username"])
    conn.close()
    return {"token": token, "username": row["username"], "gender": row["gender"], "email": row["email"]}


@app.get("/auth/me")
def me(user_name: str = Depends(current_username)):
    conn = get_db()
    row = conn.execute("SELECT username, gender, email FROM users WHERE username = ?", (user_name,)).fetchone()
    conn.close()
    return {"username": user_name, "gender": row["gender"] if row else "", "email": row["email"] if row else ""}


@app.post("/auth/logout")
def logout(authorization: str = Header(default=""), user_name: str = Depends(current_username)):
    conn = get_db()
    conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(_bearer(authorization)),))
    conn.commit()
    conn.close()
    return {"logged_out": True}


# ---------------- Ownership check ----------------
def require_chat_owner(conn, chat_id, user_name):
    """Return the chat row only if it belongs to user_name.
    Both "missing" and "not yours" give the same 404, so chat ids cannot be probed."""
    chat = conn.execute("SELECT * FROM chats WHERE id = ?", (chat_id,)).fetchone()
    if not chat or not user_name or chat["user_name"] != user_name:
        conn.close()
        raise HTTPException(404, "Chat not found")
    return chat


# ---------------- Request shapes ----------------
class NewCharacter(BaseModel):
    name: str                      # required
    description: str               # required
    first_message: str = ""       # optional: generated automatically if empty
    scenario: str = ""             # optional
    personality: str = ""          # optional
    example_dialogue: str = ""     # optional (advanced)
    gender: str = ""               # optional: "male" / "female" / "nonbinary". Auto-detected from the description if empty
    enhance: bool = True           # auto-fill empty fields (False = save exactly as given)
    age: Optional[int] = None      # optional, shown on the card; must be 18 or older
    role: str = ""                 # optional, e.g. "Psychology Professor"
    tagline: str = ""              # optional, a line the character would say
    tags: str = ""                 # optional, comma separated, up to 6 (e.g. "Ruthless, Possessive")


class EditCharacter(BaseModel):
    # send only the fields to change; the rest stay as they are
    name: Optional[str] = None
    description: Optional[str] = None
    first_message: Optional[str] = None
    scenario: Optional[str] = None
    personality: Optional[str] = None
    example_dialogue: Optional[str] = None
    age: Optional[int] = None      # 0 removes the age
    role: Optional[str] = None
    tagline: Optional[str] = None
    tags: Optional[str] = None


class StartChat(BaseModel):
    character_id: int


class SendMessage(BaseModel):
    content: str


# ---------------- API ----------------
@app.get("/")
def home():
    return {"message": "Server is running"}


@app.get("/characters")
def list_characters(user_name: str = Depends(current_username)):
    conn = get_db()
    rows = conn.execute(
        "SELECT c.id, c.name, c.description, c.is_builtin, c.age, c.role, c.tagline, c.tags, "
        "i.version AS image_v FROM characters c "
        "LEFT JOIN character_images i ON i.character_id = c.id "
        "WHERE c.is_builtin = 1 OR c.created_by = ?",
        (user_name,),
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        d["tags"] = [t for t in (d["tags"] or "").split(",") if t]
        out.append(d)
    return out


@app.post("/characters")
def create_character(c: NewCharacter, user_name: str = Depends(current_username)):
    first_message = c.first_message.strip()
    personality, scenario, example = c.personality.strip(), c.scenario.strip(), c.example_dialogue.strip()
    ok, code, message = validate_character(
        c.name, c.description, first_message, personality, scenario
    )
    if not ok:
        raise HTTPException(400, {"code": code, "message": message})
    role, tagline, tags = " ".join(c.role.split()), " ".join(c.tagline.split()), clean_tags(c.tags)
    check_card_fields(c.age, role, tagline)
    if looks_like_minor(" ".join([role, tagline, tags])):
        raise HTTPException(400, {"code": "minor", "message":
                                  "Characters in romantic roleplay must be adults (18+). Please remove any age "
                                  "under 18 or school-age details."})
    # gender: use the one the user gave, otherwise auto-detect from the description
    gender = normalize_gender(c.gender)
    gender_guessed = False
    if not gender:
        gender, gender_guessed = detect_gender(c.name, c.description, first_message, personality, scenario)
    enhanced = False
    if c.enhance:
        first_message, personality, scenario, example, enhanced = enhance_character(
            c.name, c.description, first_message, personality, scenario, example, gender
        )
    else:
        if not first_message:
            first_message = fallback_first_message(c.name)
        personality = set_pronouns(personality, gender) if personality else personality
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO characters (name, description, personality, scenario, "
        "example_dialogue, first_message, is_builtin, created_by, age, role, tagline, tags) "
        "VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)",
        (c.name, c.description, personality, scenario,
         example, first_message, user_name, c.age, role, tagline, tags),
    )
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    # lets the app show what was filled in (so the user can edit it)
    return {
        "id": new_id,
        "enhanced": enhanced,
        "gender": gender,
        "gender_guessed": gender_guessed,
        "first_message": first_message,
        "personality": personality,
        "scenario": scenario,
        "example_dialogue": example,
    }


@app.get("/characters/{character_id}")
def get_character(character_id: int, user_name: str = Depends(current_username)):
    """Full card (for the edit screen / backups). Only own characters or built-ins are returned."""
    conn = get_db()
    row = conn.execute("SELECT * FROM characters WHERE id = ?", (character_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Character not found")
    if not row["is_builtin"] and row["created_by"] != user_name:
        raise HTTPException(403, "You can only view characters you created")
    return dict(row)


@app.delete("/chats/{chat_id}")
def delete_chat(chat_id: int, user_name: str = Depends(current_username)):
    """Delete a single chat (and its memories). Characters are untouched."""
    conn = get_db()
    require_chat_owner(conn, chat_id, user_name)   # 404 for both "missing" and "not yours"
    conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()
    conn.close()
    try:
        memory_col.delete(where={"chat_id": chat_id})
    except Exception as e:
        print("MEMORY CLEANUP ERROR:", repr(e))
    return {"deleted_chat": chat_id}


@app.patch("/characters/{character_id}")
def edit_character(character_id: int, body: EditCharacter, user_name: str = Depends(current_username)):
    """Edit the fields of a character you created. Built-in characters cannot be edited here (they are changed in code)."""
    conn = get_db()
    row = conn.execute("SELECT * FROM characters WHERE id = ?", (character_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Character not found")
    if row["is_builtin"]:
        conn.close()
        raise HTTPException(403, "Built-in characters cannot be edited here")
    if row["created_by"] != user_name:
        conn.close()
        raise HTTPException(403, "You can only edit characters you created")

    fields = ["name", "description", "first_message", "scenario", "personality", "example_dialogue",
              "role", "tagline", "tags"]
    merged = {f: (row[f] or "") for f in fields}
    changed = []
    for f in fields:
        v = getattr(body, f)
        if v is not None and v.strip() != (row[f] or ""):
            merged[f] = v.strip()
            changed.append(f)
    merged["role"] = " ".join(merged["role"].split())
    merged["tagline"] = " ".join(merged["tagline"].split())
    merged["tags"] = clean_tags(merged["tags"])
    new_age = row["age"]
    if body.age is not None:
        new_age = None if body.age == 0 else body.age
    try:
        check_card_fields(new_age, merged["role"], merged["tagline"])
    except HTTPException:
        conn.close()
        raise
    if not merged["name"] or not merged["description"] or not merged["first_message"]:
        conn.close()
        raise HTTPException(400, {"code": "empty_field",
                                  "message": "Name, description and first message cannot be empty."})
    # same safety checks as at creation time
    if looks_like_minor(" ".join(merged.values())):
        conn.close()
        raise HTTPException(400, {"code": "minor", "message":
                                  "Characters in romantic roleplay must be adults (18+). Please remove any age "
                                  "under 18 or school-age details."})
    if ("name" in changed or "description" in changed) and is_real_person(merged["name"], merged["description"]):
        conn.close()
        raise HTTPException(400, {"code": "real_person", "message":
                                  "This looks like a real, famous person. Please use an original character "
                                  "with an invented name."})
    if changed or new_age != row["age"]:
        conn.execute(
            "UPDATE characters SET name = ?, description = ?, first_message = ?, scenario = ?, "
            "personality = ?, example_dialogue = ?, role = ?, tagline = ?, tags = ?, age = ? WHERE id = ?",
            (merged["name"], merged["description"], merged["first_message"], merged["scenario"],
             merged["personality"], merged["example_dialogue"], merged["role"], merged["tagline"],
             merged["tags"], new_age, character_id),
        )
        conn.commit()
    conn.close()
    return {"id": character_id, "changed": changed, **merged, "age": new_age}


@app.delete("/characters/{character_id}")
def delete_character(character_id: int, user_name: str = Depends(current_username)):
    # only a character you created can be deleted; never built-in ones
    conn = get_db()
    char = conn.execute(
        "SELECT id, is_builtin, created_by FROM characters WHERE id = ?", (character_id,)
    ).fetchone()
    if not char:
        conn.close()
        raise HTTPException(404, "Character not found")
    if char["is_builtin"]:
        conn.close()
        raise HTTPException(403, "Built-in characters cannot be deleted")
    if char["created_by"] != user_name:
        conn.close()
        raise HTTPException(403, "You can only delete characters you created")

    chat_ids = [
        r["id"] for r in conn.execute(
            "SELECT id FROM chats WHERE character_id = ?", (character_id,)
        ).fetchall()
    ]
    for cid in chat_ids:
        conn.execute("DELETE FROM messages WHERE chat_id = ?", (cid,))
    conn.execute("DELETE FROM chats WHERE character_id = ?", (character_id,))
    conn.execute("DELETE FROM characters WHERE id = ?", (character_id,))
    conn.execute("DELETE FROM character_images WHERE character_id = ?", (character_id,))
    conn.commit()
    conn.close()

    # also clear those chats' long-term memory (Chroma)
    for cid in chat_ids:
        try:
            memory_col.delete(where={"chat_id": cid})
        except Exception as e:
            print("MEMORY CLEANUP ERROR:", repr(e))
    return {"deleted": character_id, "chats_removed": len(chat_ids)}


# ---------------- Character photos ----------------
def _own_character_row(conn, character_id, user_name):
    row = conn.execute("SELECT id, is_builtin, created_by FROM characters WHERE id = ?", (character_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Character not found")
    if row["is_builtin"] or row["created_by"] != user_name:
        conn.close()
        raise HTTPException(403, "You can only change the photo of characters you created")
    return row


@app.get("/characters/{character_id}/image")
def get_character_image(character_id: int, user_name: str = Depends(current_username)):
    conn = get_db()
    row = conn.execute(
        "SELECT c.is_builtin, c.created_by, i.mime, i.data FROM characters c "
        "JOIN character_images i ON i.character_id = c.id WHERE c.id = ?", (character_id,)
    ).fetchone()
    conn.close()
    if not row or (not row["is_builtin"] and row["created_by"] != user_name):
        raise HTTPException(404, "No photo")
    return Response(content=bytes(row["data"]), media_type=row["mime"],
                    headers={"Cache-Control": "private, max-age=3600"})


@app.put("/characters/{character_id}/image")
async def set_character_image(character_id: int, request: Request, user_name: str = Depends(current_username)):
    """Upload a photo: send the raw image bytes as the request body (PNG, JPEG or WebP, up to 2 MB)."""
    try:
        if int(request.headers.get("content-length") or 0) > IMAGE_MAX_BYTES:
            raise HTTPException(413, "Photo is too large (2 MB max).")
    except ValueError:
        pass
    data = await request.body()
    if len(data) > IMAGE_MAX_BYTES:
        raise HTTPException(413, "Photo is too large (2 MB max).")
    if not sniff_image(data):
        raise HTTPException(400, "Please upload a PNG, JPEG or WebP image.")
    conn = get_db()
    _own_character_row(conn, character_id, user_name)
    save_image(conn, character_id, data)
    conn.commit()
    version = conn.execute("SELECT version FROM character_images WHERE character_id = ?", (character_id,)).fetchone()["version"]
    conn.close()
    return {"image_v": version}


@app.delete("/characters/{character_id}/image")
def delete_character_image(character_id: int, user_name: str = Depends(current_username)):
    conn = get_db()
    _own_character_row(conn, character_id, user_name)
    conn.execute("DELETE FROM character_images WHERE character_id = ?", (character_id,))
    conn.commit()
    conn.close()
    return {"removed": True}


@app.post("/chats")
def start_chat(body: StartChat, user_name: str = Depends(current_username)):
    conn = get_db()
    char = conn.execute(
        "SELECT * FROM characters WHERE id = ?", (body.character_id,)
    ).fetchone()
    if not char:
        conn.close()
        raise HTTPException(404, "Character not found")

    urow = conn.execute("SELECT gender FROM users WHERE username = ?", (user_name,)).fetchone()
    user_gender = normalize_gender(urow["gender"]) if urow else ""
    first = fill(char["first_message"], user_name, char["name"], user_gender)
    cur = conn.execute(
        "INSERT INTO chats (user_name, character_id, user_gender) VALUES (?, ?, ?)",
        (user_name, body.character_id, user_gender),
    )
    chat_id = cur.lastrowid
    conn.execute(
        "INSERT INTO messages (chat_id, role, content) VALUES (?, 'assistant', ?)",
        (chat_id, first),
    )
    conn.commit()
    conn.close()
    return {"chat_id": chat_id, "first_message": first}


@app.get("/chats")
def list_chats(character_id: Optional[int] = None, user_name: str = Depends(current_username)):
    """All of a user's chats (newest first). Pass character_id to get only that character's chats.
    'Latest chat with this character' = the first item when character_id is given."""
    conn = get_db()
    sql = (
        "SELECT c.id AS chat_id, c.character_id, ch.name AS character_name, "
        "(SELECT content FROM messages m WHERE m.chat_id = c.id ORDER BY m.id DESC LIMIT 1) AS last_message, "
        "(SELECT MAX(id) FROM messages m WHERE m.chat_id = c.id) AS last_msg_id "
        "FROM chats c JOIN characters ch ON ch.id = c.character_id WHERE c.user_name = ?"
    )
    params = [user_name]
    if character_id is not None:
        sql += " AND c.character_id = ?"
        params.append(character_id)
    sql += " ORDER BY last_msg_id DESC"
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    out = []
    for r in rows:
        preview = (r["last_message"] or "").replace("\n", " ").strip()
        out.append({
            "chat_id": r["chat_id"],
            "character_id": r["character_id"],
            "character_name": r["character_name"],
            "last_message": preview[:100] + ("..." if len(preview) > 100 else ""),
        })
    return out


@app.get("/chats/{chat_id}/messages")
def get_messages(chat_id: int, user_name: str = Depends(current_username)):
    conn = get_db()
    require_chat_owner(conn, chat_id, user_name)
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id", (chat_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.post("/chats/{chat_id}/messages")
def send_message(chat_id: int, body: SendMessage, user_name: str = Depends(current_username)):
    conn = get_db()
    chat = require_chat_owner(conn, chat_id, user_name)

    char = conn.execute(
        "SELECT * FROM characters WHERE id = ?", (chat["character_id"],)
    ).fetchone()

    user_cur = conn.execute(
        "INSERT INTO messages (chat_id, role, content) VALUES (?, 'user', ?)",
        (chat_id, body.content),
    )
    user_msg_id = user_cur.lastrowid
    conn.commit()
    rows = conn.execute(
        "SELECT id, role, content FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
        (chat_id, CONTEXT_MESSAGES),
    ).fetchall()
    conn.close()

    rows = list(reversed(rows))
    history = [{"role": r["role"], "content": r["content"]} for r in rows]
    window_start_id = rows[0]["id"]  # oldest message in the history window

    # from Chroma, only memories that fall outside the history window
    memories = recall(chat_id, body.content, window_start_id)

    char_d = dict(char)
    user_name = chat["user_name"]
    system = build_system_prompt(char_d, user_name, chat["user_gender"])
    if memories:
        system += (
            "\n\n# EARLIER IN THIS CHAT\n"
            "These moments happened earlier. Treat them as true. Use them only when they "
            "fit naturally, never quote them, and never add details that are not written here:\n"
            + "\n".join("- " + m for m in memories)
        )
    # the reminder is not saved to the DB; it is attached only to this request's last user message
    history[-1]["content"] += "\n\n" + reminder(char_d, user_name)
    messages = [{"role": "system", "content": system}] + history

    def stream():
        reply = ""
        try:
            for piece in generate_reply(messages):
                reply += piece
                yield f"data: {json.dumps({'text': piece})}\n\n"
        except Exception as e:  # e.g. 429 rate limit, provider down
            print("MODEL ERROR:", repr(e))
            # no reply was produced, so remove the user's message: resending will not create a duplicate
            conn3 = get_db()
            conn3.execute("DELETE FROM messages WHERE id = ?", (user_msg_id,))
            conn3.commit()
            conn3.close()
            yield f"data: {json.dumps({'error': 'Model is busy right now. Please try again in a moment.'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        final = trim_incomplete(reply)
        if final:
            conn2 = get_db()
            cur2 = conn2.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, 'assistant', ?)",
                (chat_id, final),
            )
            conn2.commit()
            assistant_id = cur2.lastrowid
            conn2.close()

            # save this exchange to long-term memory
            save_memory(
                chat_id,
                f"{user_name}: {body.content}\n{char['name']}: {final}",
                assistant_id,
            )
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/chats/{chat_id}/regenerate")
def regenerate(chat_id: int, user_name: str = Depends(current_username)):
    """Regenerates the last assistant reply (when the user did not like it).
    The old reply is replaced only once the new one is complete; otherwise the old one is kept."""
    conn = get_db()
    chat = require_chat_owner(conn, chat_id, user_name)
    char = conn.execute(
        "SELECT * FROM characters WHERE id = ?", (chat["character_id"],)
    ).fetchone()
    rows = conn.execute(
        "SELECT id, role, content FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
        (chat_id, CONTEXT_MESSAGES + 1),
    ).fetchall()
    conn.close()
    rows = list(reversed(rows))

    # only when the last message is from the assistant and a user message precedes it
    # (the opening greeting cannot be regenerated)
    if len(rows) < 2 or rows[-1]["role"] != "assistant" or rows[-2]["role"] != "user":
        raise HTTPException(400, "Nothing to regenerate yet. Send a message first.")

    old = rows[-1]
    history_rows = rows[:-1][-CONTEXT_MESSAGES:]   # history up to the last user message
    history = [{"role": r["role"], "content": r["content"]} for r in history_rows]
    user_text = history_rows[-1]["content"]
    window_start_id = history_rows[0]["id"]

    memories = recall(chat_id, user_text, window_start_id)
    char_d = dict(char)
    user_name = chat["user_name"]
    system = build_system_prompt(char_d, user_name, chat["user_gender"])
    if memories:
        system += (
            "\n\n# EARLIER IN THIS CHAT\n"
            "These moments happened earlier. Treat them as true. Use them only when they "
            "fit naturally, never quote them, and never add details that are not written here:\n"
            + "\n".join("- " + m for m in memories)
        )
    # give the old reply's opening words so the new reply differs
    old_start = " ".join(old["content"].split()[:12])
    different = (
        "[Write a clearly different version of your previous reply: new opening words, a new "
        "gesture, a new angle, same scene and same facts. Do not begin with: \"" + old_start + "\".]"
    )
    history[-1]["content"] += "\n\n" + reminder(char_d, user_name) + "\n" + different
    messages = [{"role": "system", "content": system}] + history

    def stream():
        reply = ""
        try:
            for piece in generate_reply(messages, temperature=0.9):
                reply += piece
                yield f"data: {json.dumps({'text': piece})}\n\n"
        except Exception as e:  # the old reply stays as it was
            print("MODEL ERROR (regenerate):", repr(e))
            yield f"data: {json.dumps({'error': 'Model is busy right now. Your previous reply is unchanged. Please try again.'})}\n\n"
            yield "data: [DONE]\n\n"
            return

        final = trim_incomplete(reply)
        if final:
            conn2 = get_db()
            conn2.execute("UPDATE messages SET content = ? WHERE id = ?", (final, old["id"]))
            conn2.commit()
            conn2.close()
            # remove the old reply's memory and save the new one (same message id)
            try:
                memory_col.delete(where={"$and": [{"chat_id": chat_id}, {"msg_id": old["id"]}]})
            except Exception as e:
                print("MEMORY CLEANUP ERROR:", repr(e))
            save_memory(chat_id, f"{user_name}: {user_text}\n{char['name']}: {final}", old["id"])
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")