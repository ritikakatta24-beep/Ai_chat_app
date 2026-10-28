import base64
import html
import inspect
import io
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from functools import lru_cache

import requests
import streamlit as st
from PIL import Image, ImageOps

API = os.environ.get("API_URL", "http://localhost:8000")
APP_NAME = "VelvetTalk"  # change this to your own app name

st.set_page_config(page_title=APP_NAME, page_icon="🌹", layout="wide",
                   initial_sidebar_state="collapsed")

# ---------------------------------------------------------------- styling
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;0,700;1,500&family=Inter:wght@400;500&display=swap');

html, body, [class*="st-"], .stApp { font-family: 'Inter', sans-serif; }
span[data-testid="stIconMaterial"], [data-testid="stIconMaterial"] {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important;
    font-feature-settings: 'liga'; letter-spacing: normal; }
.stApp { background: radial-gradient(1200px 600px at 50% -10%, #1d0d12 0%, #0b0809 60%); }
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stSidebar"], [data-testid="collapsedControl"] { display: none; }
.block-container { padding-top: 1.6rem; max-width: 1280px; }

.brand { font-family: 'Cormorant Garamond', serif; font-size: 2rem; font-weight: 700;
         color: #f1e6e3; letter-spacing: .5px; }
.brand span { color: #b4374b; }
.hero { font-family: 'Cormorant Garamond', serif; font-size: 3.6rem; font-weight: 700;
        text-align: center; color: #f1e6e3; line-height: 1.05; margin-top: 8vh; }
.hero span { color: #b4374b; }
.hero-sub { text-align: center; color: #9a8c8a; font-style: italic;
            font-family: 'Cormorant Garamond', serif; font-size: 1.3rem; margin: .4rem 0 2rem; }
.section-title { font-family: 'Cormorant Garamond', serif; font-size: 1.9rem; font-weight: 600;
                 color: #f1e6e3; margin: .6rem 0 .8rem; }

/* pill buttons */
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    border-radius: 999px; border: 1px solid #3a2328; font-weight: 500; }
.stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"] {
    background: #6f2531; border-color: #8a2c3c; color: #f6e9e6; }
.stButton > button[kind="primary"]:hover { background: #8a2c3c; }

/* character cards: the photo sits in the top part of the card and fades into the dark; all the text
   starts BELOW the photo (the photo is set per card on ::before). Every text block has a fixed height,
   so all cards line up exactly the same no matter how many lines the quote or the tags take. */
[class*="st-key-card_"] { position: relative; border: 1px solid #2c1c20; border-radius: 22px;
    padding: 190px 1rem 1rem; gap: 0; background-color: #0b0809; overflow: hidden; }
[class*="st-key-card_"]::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 250px;
    background-size: cover; background-position: center 18%; background-repeat: no-repeat;
    -webkit-mask-image: linear-gradient(180deg, #000 0%, #000 62%, transparent 100%);
    mask-image: linear-gradient(180deg, #000 0%, #000 62%, transparent 100%); }
[class*="st-key-card_"] > div { position: relative; z-index: 1; flex: 0 0 auto !important; }   /* above the photo; nothing shrinks onto the button */
.c-name { font-family: 'Cormorant Garamond', serif; font-size: 1.9rem; font-weight: 600; line-height: 1.15;
          height: 2.19rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
          color: #f6efec; text-shadow: 0 2px 12px rgba(0,0,0,.7); }
.c-sub { font-family: 'Cormorant Garamond', serif; font-size: 1rem; line-height: 1.3; height: 1.3rem;
         white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #b9acaa; }
.c-quote { font-family: 'Cormorant Garamond', serif; font-style: italic; font-size: 1.08rem; line-height: 1.3;
           color: #ddd2cf; margin: .3rem 0 .5rem; height: 2.81rem;
           display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
/* tags: smaller pills, room for 2 rows, packed against the bottom so they always sit the same distance
   above the button, with clear gaps between the pills (fewer rows just leaves empty space above them) */
.c-tags { display: flex; flex-wrap: wrap; gap: .45rem .4rem; margin: 0 0 .9rem; min-height: 3.7rem;
          align-content: flex-end; }
.c-tags span { border: 1px solid #7a2a35; background: rgba(8,5,6,.7); color: #efe5e2; border-radius: 999px;
               padding: .1rem .65rem; font-family: 'Cormorant Garamond', serif; font-size: .92rem;
               line-height: 1.25; white-space: nowrap; }

/* hide the "Press Enter to submit form" hint: it sat on top of the placeholder text */
[data-testid="stTextInput"] [data-testid="InputInstructions"] { display: none !important; }
/* the "Press Cmd+Enter to submit" hint of text areas: sit under the box instead of on top of the text */
[data-testid="stTextArea"] { padding-bottom: 1.5rem; }
[data-testid="stTextArea"] [data-testid="InputInstructions"] { bottom: -1.6rem !important; right: 0 !important; }

/* smooth screens: each page fades in once when you switch to it, and old content is not greyed out
   while the next run loads (that grey flash is what looked abrupt) */
@keyframes vtFade { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
[class*="st-key-page_"] { animation: vtFade .4s ease both; }
@keyframes vtPulse { 0%, 100% { opacity: .25; transform: scale(.8); } 50% { opacity: 1; transform: scale(1.15); } }
.vt-loading { display: flex; align-items: center; justify-content: center; gap: .8rem; padding: 4.5rem 0;
              font-family: 'Cormorant Garamond', serif; font-size: 1.5rem; color: #a8999a; }
.vt-loading i { width: 14px; height: 14px; border-radius: 50%; background: #8a2c3c; animation: vtPulse 1s ease-in-out infinite; }
[data-stale="true"], .stApp [data-stale="true"] { opacity: 1 !important; transition: none !important; }
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    transition: background-color .2s ease, border-color .2s ease, transform .15s ease; }
.stButton > button:active, [data-testid="stFormSubmitButton"] > button:active { transform: scale(.98); }
[data-testid="stMain"] { scroll-behavior: smooth; }

/* chat list buttons */
.stButton > button p { white-space: pre-line; text-align: left; }
[class*="st-key-chat_"] button { justify-content: flex-start; text-align: left; border-radius: 16px;
    padding: .7rem 1rem; }
.chat-head { display: flex; align-items: center; gap: .8rem; }
.chat-head img { width: 46px; height: 46px; border-radius: 50%; }
.chat-head .n { font-weight: 600; color: #f1e6e3; line-height: 1.1; }
.chat-head .s { color: #6fbf8a; font-size: .8rem; }

/* chat bubbles */
[data-testid="stChatMessage"] { background: #1a1416; border: 1px solid #2a2023;
    border-radius: 18px; padding: .8rem 1rem; }
[class*="st-key-umsg_"] [data-testid="stChatMessage"] {
    background: #4b1c27; border-color: #6f2531; }
[data-testid="stChatInput"] { border-radius: 999px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PALETTES = [("#6b1f2f", "#1a0a0f"), ("#4a1a30", "#0e0709"), ("#5a2216", "#150a08"),
            ("#3a2060", "#0b0810"), ("#5a1d2a", "#120809")]
GENDERS = {"Female": "female", "Male": "male", "Non-binary": "nonbinary", "Prefer not to say": ""}
NAV = ["Discover", "Chats", "My Characters", "Create"]


# ---------------------------------------------------------------- helpers
def _err_text(r):
    try:
        d = r.json().get("detail", r.text)
    except Exception:
        return r.text
    return d.get("message", str(d)) if isinstance(d, dict) else str(d)


COOKIE = "vt_token"


def run_js(js):
    """Run a tiny script in the browser (works on old and new Streamlit versions)."""
    doc = f"<script>try{{{js}}}catch(e){{}}</script>"
    if hasattr(st, "iframe"):
        st.iframe(doc, height=1)
    else:
        import streamlit.components.v1 as components
        components.html(doc, height=0)


def auth_headers():
    t = st.session_state.get("token")
    return {"Authorization": f"Bearer {t}"} if t else {}


def drop_session(notice=None):
    """Forget the login on this side (the cookie is cleared on the next render)."""
    for k in ("token", "user_name", "user_gender", "chat_id", "_chars_cache"):
        st.session_state.pop(k, None)
    st.session_state.cookie_op = ("clear", "")
    st.session_state.login_notice = notice


CHARS_TTL = 30  # seconds the character list is reused, so switching tabs does not wait on the server


def call(method, path, **kw):
    cacheable = method == "GET" and path == "/characters"
    if cacheable:
        hit = st.session_state.get("_chars_cache")
        if hit and time.time() - hit[0] < CHARS_TTL and hit[2] == st.session_state.get("token"):
            return hit[1]
    try:
        r = requests.request(method, API + path, timeout=90, headers=auth_headers(), **kw)
    except requests.RequestException:
        raise RuntimeError("Could not reach the server. It may be waking up, please try again in a moment.")
    if r.status_code == 401 and st.session_state.get("token"):
        drop_session("Your session has expired. Please log in again.")
        raise RuntimeError("Your session has expired. Please log in again.")
    if not r.ok:
        raise RuntimeError(_err_text(r))
    data = r.json()
    if cacheable:
        st.session_state._chars_cache = (time.time(), data, st.session_state.get("token"))
    elif method != "GET":
        st.session_state.pop("_chars_cache", None)   # something changed: fetch fresh next time
    return data


def _cached_chars():
    hit = st.session_state.get("_chars_cache")
    if hit and time.time() - hit[0] < CHARS_TTL and hit[2] == st.session_state.get("token"):
        return hit[1]
    return None


def call_many(paths):
    """Fetch several GET endpoints at the same time, so a screen waits for one round trip, not one per call.
    Returns a list in the same order; an entry is an Exception if that one request failed."""
    headers, token = auth_headers(), st.session_state.get("token")
    out, todo = {}, []
    for p in paths:
        if p == "/characters" and _cached_chars() is not None:
            out[p] = _cached_chars()
        else:
            todo.append(p)

    def one(p):
        try:
            return requests.get(API + p, headers=headers, timeout=90)
        except requests.RequestException:
            return None

    if todo:
        with ThreadPoolExecutor(max_workers=len(todo)) as ex:
            results = list(ex.map(one, todo))
        for p, r in zip(todo, results):
            if r is None:
                out[p] = RuntimeError("Could not reach the server. It may be waking up, please try again in a moment.")
            elif r.status_code == 401 and token:
                drop_session("Your session has expired. Please log in again.")
                raise RuntimeError("Your session has expired. Please log in again.")
            elif not r.ok:
                out[p] = RuntimeError(_err_text(r))
            else:
                out[p] = r.json()
                if p == "/characters":
                    st.session_state._chars_cache = (time.time(), out[p], token)
    return [out[p] for p in paths]


def stream_reply(path, payload=None):
    """Yields text pieces from the server's event stream."""
    try:
        with requests.post(API + path, json=payload, stream=True, timeout=120,
                           headers=auth_headers()) as r:
            if r.status_code == 401:
                drop_session("Your session has expired. Please log in again.")
                st.session_state.stream_error = "Your session has expired. Please log in again."
                return
            if not r.ok:
                st.session_state.stream_error = _err_text(r)
                return
            r.encoding = "utf-8"
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if "error" in obj:
                    st.session_state.stream_error = obj["error"]
                    break
                yield obj["text"]
    except requests.RequestException:
        st.session_state.stream_error = "Connection lost. Please try again."


def palette(name):
    return PALETTES[sum(map(ord, name)) % len(PALETTES)]


@lru_cache(maxsize=256)
def avatar(name):
    c1, c2 = palette(name)
    ch = html.escape((name.strip()[:1] or "?").upper())
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="96" height="96" viewBox="0 0 100 100">'
        f'<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{c1}"/>'
        f'<stop offset="1" stop-color="{c2}"/></linearGradient></defs>'
        '<rect width="100" height="100" fill="url(#g)"/>'
        '<text x="50" y="52" font-family="Georgia,serif" font-size="46" fill="#e8c9cf" '
        f'text-anchor="middle" dominant-baseline="central">{ch}</text></svg>'
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def clean_desc(text, char_name):
    t = (text or "").replace("{{user}}'s", "your").replace("{{user}}", "you").replace("{{char}}", char_name)
    for k, v in {"{{u_obj}}": "you", "{{u_pos}}": "your", "{{u_self}}": "yourself", "{{u_woman}}": "person",
                 "{{u_wife}}": "partner", "{{u_is_my_wife}}": "You are my partner"}.items():
        t = t.replace(k, v)
    return re.sub(r"\{\{[^}]*\}\}", "you", t)


def nl(text):
    """People paste text with a literal backslash-n; turn it into a real line break."""
    return (text or "").replace("\\n", "\n").strip()


@st.cache_data(show_spinner=False, max_entries=128)
def _fetch_image(char_id, version, _token):
    r = requests.get(f"{API}/characters/{char_id}/image", headers={"Authorization": f"Bearer {_token}"}, timeout=60)
    r.raise_for_status()
    return r.content, r.headers.get("content-type", "image/jpeg")


@st.cache_data(show_spinner=False, max_entries=128)
def _photo_uri(char_id, version, _token):
    raw, mime = _fetch_image(char_id, version, _token)
    return f"data:{mime};base64," + base64.b64encode(raw).decode()


@st.cache_data(show_spinner=False, max_entries=128)
def _square_uri(char_id, version, _token):
    raw, _ = _fetch_image(char_id, version, _token)
    im = ImageOps.fit(Image.open(io.BytesIO(raw)).convert("RGB"), (128, 128), centering=(0.5, 0.25))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def photo_uri(char, square=False):
    """Photo of a character as a data URI, or None (no photo, or the server could not be reached)."""
    if not char or not char.get("image_v"):
        return None
    try:
        return (_square_uri if square else _photo_uri)(char["id"], char["image_v"], st.session_state.get("token", ""))
    except Exception:
        return None


def char_avatar(char, name):
    return photo_uri(char, square=True) or avatar(name)


def prepare_photo(upload):
    """Shrink and clean an uploaded picture (max 900 px wide, JPEG, no metadata) before sending it."""
    im = ImageOps.exif_transpose(Image.open(upload))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, "#0b0809")
        bg.paste(im, mask=im.split()[-1])
        im = bg
    else:
        im = im.convert("RGB")
    im.thumbnail((900, 1200))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=86, optimize=True)
    return buf.getvalue()


def upload_photo(char_id, data):
    try:
        r = requests.put(f"{API}/characters/{char_id}/image", data=data, timeout=90,
                         headers={**auth_headers(), "Content-Type": "image/jpeg"})
    except requests.RequestException:
        raise RuntimeError("Could not upload the photo. Please try again.")
    if not r.ok:
        raise RuntimeError(_err_text(r))
    st.session_state.pop("_chars_cache", None)   # the photo changed: refresh the card list


def ss(key, default):
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


ss("chat_id", None)
ss("stream_error", None)
ss("flash", None)
ss("regen", False)
ss("nav", "Discover")
LOADER = None   # set by the router: the small "Opening..." line that is removed once the data has arrived
BODY = None     # set by the router: the block that holds the current screen
LOADING_TEXT = {"Discover": "Loading characters…", "Chats": "Opening your chat…",
                "My Characters": "Loading your characters…", "Create": "Just a moment…"}

# wake the (free-tier) server in the background while the user is typing
if "warmed" not in st.session_state:
    st.session_state.warmed = True

    def _ping():
        try:
            requests.get(API + "/", timeout=90)
        except Exception:
            pass
    threading.Thread(target=_ping, daemon=True).start()


def apply_cookie_op():
    """Set / clear the browser cookie that keeps the login across page refreshes."""
    op = st.session_state.pop("cookie_op", None)
    if not op:
        return
    if op[0] == "set":
        js = (f"document.cookie='{COOKIE}={op[1]}; max-age={14 * 86400}; path=/; SameSite=Lax'"
              "+(window.parent.location.protocol==='https:'?'; Secure':'');")
    else:
        js = f"document.cookie='{COOKIE}=; max-age=0; path=/; SameSite=Lax';"
    js = js.replace("document.cookie", "window.parent.document.cookie")
    run_js(js)


def restore_session():
    """After a refresh, sign back in from the cookie (once per browser session)."""
    if st.session_state.get("restore_tried") or st.session_state.get("token"):
        return
    st.session_state.restore_tried = True
    try:
        token = st.context.cookies.get(COOKIE, "")
    except Exception:
        token = ""
    if not token:
        return
    try:
        with st.spinner("Signing you in..."):
            r = requests.get(API + "/auth/me", headers={"Authorization": f"Bearer {token}"}, timeout=90)
        if r.ok:
            d = r.json()
            st.session_state.token = token
            st.session_state.user_name = d["username"]
            st.session_state.user_gender = d.get("gender", "")
        else:
            st.session_state.cookie_op = ("clear", "")
    except requests.RequestException:
        pass


restore_session()
apply_cookie_op()


# ---------------------------------------------------------------- welcome screen
def sign_in(data):
    st.session_state.token = data["token"]
    st.session_state.user_name = data["username"]
    st.session_state.user_gender = data.get("gender", "")
    st.session_state.cookie_op = ("set", data["token"])
    st.session_state.login_notice = None
    st.rerun()


EMAIL_OK = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")


def welcome():
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid, st.container(key="page_welcome"):
        st.markdown(f'<div class="hero">{APP_NAME[:-4]}<span>{APP_NAME[-4:]}</span></div>'
                    '<div class="hero-sub">Every story starts with a name.</div>',
                    unsafe_allow_html=True)
        if st.session_state.get("login_notice"):
            st.warning(st.session_state.login_notice)
        tab_in, tab_up = st.tabs(["Log in", "Create account"])

        with tab_in:
            with st.form("login"):
                login_email = st.text_input("Email", placeholder="you@example.com")
                pw = st.text_input("Password", type="password")
                go = st.form_submit_button("Log in", type="primary", use_container_width=True)
            if go:
                if not login_email.strip() or not pw:
                    st.error("Please enter your email and password.")
                else:
                    try:
                        with st.spinner("Logging in..."):
                            data = call("POST", "/auth/login", json={"email": login_email, "password": pw})
                        sign_in(data)
                    except RuntimeError as e:
                        st.error(str(e))

        with tab_up:
            with st.form("signup"):
                email = st.text_input("Email", placeholder="you@example.com",
                                      help="One account per email. You log in with this email.")
                name = st.text_input("Choose a username", placeholder="This is the name characters will call you",
                                     help="2-30 characters: letters, numbers, spaces, . _ -")
                pw = st.text_input("Choose a password", type="password", help="At least 8 characters")
                pw2 = st.text_input("Repeat password", type="password")
                gender = st.radio("I am", list(GENDERS), horizontal=True)
                go = st.form_submit_button("Create account", type="primary", use_container_width=True)
            if go:
                if not EMAIL_OK.match(email.strip()):
                    st.error("Please enter a valid email address.")
                elif not name.strip():
                    st.error("Please choose a username.")
                elif pw != pw2:
                    st.error("The two passwords do not match.")
                else:
                    try:
                        with st.spinner("Creating your account..."):
                            data = call("POST", "/auth/signup", json={
                                "email": email.strip(), "username": name, "password": pw,
                                "gender": GENDERS[gender]})
                        sign_in(data)
                    except RuntimeError as e:
                        st.error(str(e))


if not st.session_state.get("user_name"):
    welcome()
    st.stop()

USER = st.session_state.user_name


# ---------------------------------------------------------------- callbacks
def logout():
    try:
        requests.post(API + "/auth/logout", headers=auth_headers(), timeout=15)
    except requests.RequestException:
        pass
    drop_session(None)


def open_chat(char_id, chat_id=None):
    """Open the latest chat with a character, or start a new one."""
    try:
        if chat_id is None:
            chat_id = call("POST", "/chats", json={"character_id": char_id})["chat_id"]
        st.session_state.chat_id = chat_id
        st.session_state.nav = "Chats"
    except RuntimeError as e:
        st.session_state.flash = str(e)


def new_chat(char_id):
    open_chat(char_id, None)


def select_chat(chat_id):
    st.session_state.chat_id = chat_id


def delete_chat(chat_id):
    try:
        call("DELETE", f"/chats/{chat_id}")
        st.session_state.chat_id = None
    except RuntimeError as e:
        st.session_state.flash = str(e)


def start_regen():
    st.session_state.regen = True


# ---------------------------------------------------------------- header
def header():
    c1, c2, c3 = st.columns([2, 5, 2], vertical_alignment="center")
    c1.markdown(f'<div class="brand">{APP_NAME[:-4]}<span>{APP_NAME[-4:]}</span></div>',
                unsafe_allow_html=True)
    with c2:
        st.segmented_control("Menu", NAV, key="nav", label_visibility="collapsed")
    with c3:
        with st.popover(f"👤 {USER}", use_container_width=True):
            st.button("Log out", on_click=logout, use_container_width=True)
    if st.session_state.flash:
        st.error(st.session_state.flash)
        st.session_state.flash = None
    if st.session_state.get("notice"):
        st.success(st.session_state.pop("notice"))


# ---------------------------------------------------------------- pages
CHAT_ICON = {"icon": ":material/chat_bubble:"} if "icon" in inspect.signature(st.button).parameters else {}


def card_html(c):
    sub = " • ".join(x for x in (str(c["age"]) if c.get("age") else "", c.get("role") or "") if x)
    sub = sub or ("Original" if c["is_builtin"] else "Created by you")
    if c.get("tagline"):
        quote = "“" + c["tagline"] + "”"
    else:
        text = clean_desc(c["description"], c["name"])
        quote = text[:150].rstrip() + ("…" if len(text) > 150 else "")
    tags = "".join(f"<span>{html.escape(t)}</span>" for t in (c.get("tags") or [])[:6])
    return (f'<div class="c-name">{html.escape(c["name"])}</div>'
            f'<div class="c-sub">{html.escape(sub)}</div>'
            f'<div class="c-quote">{html.escape(quote)}</div>'
            f'<div class="c-tags">{tags}</div>')


def page_discover():
    st.markdown('<div class="section-title">Discover characters</div>', unsafe_allow_html=True)
    try:
        characters, chats = call_many(["/characters", "/chats"])
        for r in (characters, chats):
            if isinstance(r, Exception):
                raise r
    except RuntimeError as e:
        LOADER.empty()
        st.error(str(e))
        return
    LOADER.empty()

    latest = {}
    for c in chats:  # newest first
        latest.setdefault(c["character_id"], c["chat_id"])

    q = st.text_input("Search", placeholder="Search characters...", label_visibility="collapsed")
    items = [c for c in characters
             if q.lower() in c["name"].lower() or q.lower() in (c["description"] or "").lower()]
    if not items:
        st.info("No characters found.")
        return

    # one style block: each card gets its own photo (or a colour gradient when it has none)
    rules = []
    for c in items:
        uri = photo_uri(c)
        c1, c2 = palette(c["name"])
        bg = f'url("{uri}")' if uri else f"linear-gradient(160deg,{c1},{c2})"
        rules.append(f".st-key-card_{c['id']}::before {{ background-image: {bg}; }}")
    st.markdown("<style>" + "\n".join(rules) + "</style>", unsafe_allow_html=True)

    for i in range(0, len(items), 3):
        cols = st.columns(3, gap="medium")
        for col, c in zip(cols, items[i:i + 3]):
            with col:
                with st.container(key=f"card_{c['id']}"):
                    st.markdown(card_html(c), unsafe_allow_html=True)
                    existing = latest.get(c["id"])
                    st.button("Continue Chat" if existing else "Start Chat",
                              key=f"start_{c['id']}", type="primary", use_container_width=True,
                              on_click=open_chat, args=(c["id"], existing), **CHAT_ICON)


def page_chats():
    guess = st.session_state.chat_id       # we usually already know which chat to open, so fetch its messages too
    paths = ["/chats", "/characters"] + ([f"/chats/{guess}/messages"] if guess else [])
    try:
        got = call_many(paths)
        chats = got[0]
        if isinstance(chats, Exception):
            raise chats
    except RuntimeError as e:
        with BODY:
            LOADER.empty()
            st.error(str(e))
        return
    if not chats:
        with BODY:
            LOADER.empty()
            st.markdown('<div class="section-title">No chats yet</div>', unsafe_allow_html=True)
            st.info("Open Discover and start a chat with any character.")
        return

    ids = [c["chat_id"] for c in chats]
    if st.session_state.chat_id not in ids:
        st.session_state.chat_id = ids[0]
    cid = st.session_state.chat_id
    cur = next(c for c in chats if c["chat_id"] == cid)
    cname = cur["character_name"]
    chars = {c["id"]: c for c in got[1]} if isinstance(got[1], list) else {}
    av = char_avatar(chars.get(cur["character_id"]), cname)
    messages = got[2] if (len(got) > 2 and guess == cid and not isinstance(got[2], Exception)) else None
    if messages is None:
        try:
            messages = call("GET", f"/chats/{cid}/messages")
        except RuntimeError as e:
            with BODY:
                LOADER.empty()
                st.error(str(e))
            return

    with BODY:
        LOADER.empty()
        left, right = st.columns([1, 2.4], gap="large")

    with left:
        st.markdown('<div class="section-title">Chats</div>', unsafe_allow_html=True)
        for c in chats:
            raw = c["last_message"].replace("*", "")
            preview = raw[:55].rstrip() + ("…" if len(raw) > 55 else "")
            st.button(f"{c['character_name']}\n{preview}", key=f"chat_{c['chat_id']}",
                      type="primary" if c["chat_id"] == cid else "secondary",
                      use_container_width=True, on_click=select_chat, args=(c["chat_id"],))

    with right:
        h1, h2 = st.columns([8, 1], vertical_alignment="center")
        h1.markdown(f'<div class="chat-head"><img src="{av}"/><div>'
                    f'<div class="n">{html.escape(cname)}</div><div class="s">● Online</div></div></div>',
                    unsafe_allow_html=True)
        with h2.popover("⋯"):
            st.button("Start a new chat", on_click=new_chat, args=(cur["character_id"],),
                      use_container_width=True)
            st.button("Delete this chat", on_click=delete_chat, args=(cid,), use_container_width=True)
        st.divider()

        can_regen = (len(messages) >= 2 and messages[-1]["role"] == "assistant"
                     and messages[-2]["role"] == "user")
        regenerating = st.session_state.regen and can_regen
        shown = messages[:-1] if regenerating else messages

        for i, m in enumerate(shown):
            if m["role"] == "user":
                with st.container(key=f"umsg_{i}"):
                    with st.chat_message("user", avatar=avatar(USER)):
                        st.markdown(m["content"])
            else:
                with st.chat_message("assistant", avatar=av):
                    st.markdown(m["content"])

        if regenerating:
            st.session_state.regen = False
            with st.chat_message("assistant", avatar=av):
                st.write_stream(stream_reply(f"/chats/{cid}/regenerate"))
            st.rerun()

        if st.session_state.stream_error:
            st.error(st.session_state.stream_error)
            st.session_state.stream_error = None

        if can_regen:
            st.button("↻ Regenerate reply", on_click=start_regen)

    prompt = st.chat_input("Type a message...")
    if prompt:
        with right:
            with st.container(key="umsg_live"):
                with st.chat_message("user", avatar=avatar(USER)):
                    st.markdown(prompt)
            with st.chat_message("assistant", avatar=av):
                st.write_stream(stream_reply(f"/chats/{cid}/messages", {"content": prompt}))
        st.rerun()

    # keep the newest message in view
    run_js("const m=window.parent.document.querySelector('[data-testid=\"stMain\"]');"
           "if(m){m.scrollTo({top:m.scrollHeight});}")


def card_fields(prefix, v=None):
    """The shared character form fields. v = existing values when editing."""
    v = v or {}
    name = st.text_input("Name *", v.get("name", ""), key=f"{prefix}_name")
    a, b = st.columns([1, 2])
    age = a.number_input("Age", min_value=18, max_value=120, value=v.get("age"), step=1,
                         placeholder="18+", key=f"{prefix}_age", help="Characters must be 18 or older")
    role = b.text_input("Role", v.get("role", ""), placeholder="e.g. Psychology Professor", key=f"{prefix}_role")
    desc = st.text_area("Description *", v.get("description", ""), height=130, key=f"{prefix}_desc",
                        placeholder="Who are they? Looks, job, nature, how they relate to you. "
                                    "Use {{user}} for the person chatting.")
    tagline = st.text_input("Tagline (optional)", v.get("tagline", ""), key=f"{prefix}_tag",
                            placeholder="A line they would say. Shown on the card.")
    tags = st.text_input("Tags (optional)", ", ".join(t for t in (v.get("tags") or "").split(",") if t),
                         key=f"{prefix}_tags", placeholder="Ruthless, Possessive, Dark Romance",
                         help="Up to 6, separated by commas")
    first = st.text_area("First message (optional)", v.get("first_message", ""), height=120, key=f"{prefix}_first")
    example = st.text_area("Example dialogue (optional)", v.get("example_dialogue", ""), height=140,
                           key=f"{prefix}_ex", placeholder="{{user}}: ...\n{{char}}: ...",
                           help="A short sample exchange that shows how they talk. Use {{user}} and {{char}}.")
    pers = st.text_area("Personality (optional)", v.get("personality", ""), height=120, key=f"{prefix}_pers")
    scen = st.text_area("Scenario (optional)", v.get("scenario", ""), height=100, key=f"{prefix}_scen")
    return {"name": name.strip(), "age": int(age) if age else None, "role": role.strip(),
            "description": nl(desc), "tagline": tagline.strip(), "tags": tags.strip(),
            "first_message": nl(first), "example_dialogue": nl(example),
            "personality": nl(pers), "scenario": nl(scen)}


def page_mine():
    st.markdown('<div class="section-title">My characters</div>', unsafe_allow_html=True)
    try:
        characters = call("GET", "/characters")
    except RuntimeError as e:
        LOADER.empty()
        st.error(str(e))
        return
    LOADER.empty()
    own = [c for c in characters if not c["is_builtin"]]
    if not own:
        st.info("You have not created any characters yet. Use the Create tab to make one.")
        return

    label = {f"{c['name']} (#{c['id']})": c for c in own}
    pick = label[st.selectbox("Choose a character to edit", list(label))]
    full = call("GET", f"/characters/{pick['id']}")

    _, mid, _ = st.columns([1, 2.2, 1])
    with mid:
        current = photo_uri(pick, square=False)
        if current:
            st.image(current, width=180, caption="Current photo")
        with st.form(f"edit_{pick['id']}"):
            vals = card_fields(f"edit{pick['id']}", full)
            new_photo = st.file_uploader("Replace photo (optional)", type=["png", "jpg", "jpeg", "webp"],
                                         key=f"edit{pick['id']}_photo")
            drop_photo = st.checkbox("Remove current photo", key=f"edit{pick['id']}_drop") if current else False
            save = st.form_submit_button("Save changes", type="primary", use_container_width=True)
        if save:
            try:
                photo = prepare_photo(new_photo) if new_photo else None
            except Exception:
                st.error("That file could not be read as an image.")
                return
            try:
                call("PATCH", f"/characters/{pick['id']}",
                     json={**vals, "age": vals["age"] or 0})
                if photo:
                    upload_photo(pick["id"], photo)
                elif drop_photo:
                    call("DELETE", f"/characters/{pick['id']}/image")
                st.session_state.notice = "Character updated."
                st.rerun()
            except RuntimeError as e:
                st.error(str(e))

        with st.popover("Delete this character"):
            st.warning("This also removes every chat with this character.")
            if st.button("Yes, delete it", key=f"del_{pick['id']}"):
                try:
                    call("DELETE", f"/characters/{pick['id']}")
                    st.rerun()
                except RuntimeError as e:
                    st.error(str(e))


def page_create():
    LOADER.empty()
    _, mid, _ = st.columns([1, 2.2, 1])
    with mid:
        st.markdown('<div class="section-title">Create a character</div>', unsafe_allow_html=True)
        with st.form("create"):
            vals = card_fields("new")
            photo_file = st.file_uploader("Photo (optional)", type=["png", "jpg", "jpeg", "webp"],
                                          help="Shown on the character card and next to their messages")
            gender = st.selectbox("Character gender", ["Auto-detect", "Female", "Male", "Non-binary"])
            enhance = st.checkbox("Auto-fill empty fields", value=True)
            go = st.form_submit_button("Create character", type="primary", use_container_width=True)
        if go:
            if not vals["name"] or not vals["description"]:
                st.error("Name and description are required.")
                return
            photo = None
            if photo_file:
                try:
                    photo = prepare_photo(photo_file)
                except Exception:
                    st.error("That file could not be read as an image.")
                    return
            try:
                with st.spinner("Creating your character..."):
                    res = call("POST", "/characters", json={
                        **vals, "enhance": enhance,
                        "gender": "" if gender == "Auto-detect" else gender.lower().replace("-", "")})
                if photo:
                    try:
                        upload_photo(res["id"], photo)
                    except RuntimeError as e:
                        st.warning(f"Character created, but the photo could not be saved: {e}")
                st.success("Character created. You can find it in Discover.")
                with st.expander("See what was generated"):
                    st.markdown(f"**First message**\n\n{res['first_message']}")
                    st.markdown(f"**Personality**\n\n{res['personality']}")
                    st.markdown(f"**Scenario**\n\n{res['scenario']}")
            except RuntimeError as e:
                st.error(str(e))


# ---------------------------------------------------------------- router
header()
_page = st.session_state.nav or "Discover"
_changed = st.session_state.get("_shown_page") != _page
st.session_state._shown_page = _page
_key = "page_" + _page.replace(" ", "_")

# When you switch screens the old one is wiped at once and a small "Opening..." line shows while the new
# screen is fetched. (Before, the old screen stayed on and looked frozen until the new one popped in.)
BODY = st.empty().container(key=_key) if _changed else st.container(key=_key)
with BODY:
    LOADER = st.empty()
    if _changed:
        LOADER.markdown(f'<div class="vt-loading"><i></i>{LOADING_TEXT.get(_page, "Loading…")}</div>',
                        unsafe_allow_html=True)
if _changed and _page != "Chats":
    run_js("const m=window.parent.document.querySelector('[data-testid=\"stMain\"]');"
           "if(m){m.scrollTo({top:0,behavior:'instant'});}")

if _page == "Chats":
    page_chats()   # the chat box must stay at the root of the page, so this page draws into BODY itself
else:
    with BODY:
        {"Discover": page_discover, "My Characters": page_mine, "Create": page_create}[_page]()
        LOADER.empty()