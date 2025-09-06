#!/usr/bin/env python3
"""
grace_full.py

Grace — Ollama LLM assistant with:
 - Translator (Helsinki-NLP via transformers)
 - Sentiment (Hugging Face sentiment pipeline)
 - Perspective API moderation (user inputs & assistant outputs)
 - Music integrations: Spotify (primary) and YouTube (fallback)
 - MongoDB storage (user prefs + chat history) with JSON fallback
 - Firebase Auth verification (optional); local guest session fallback

Environment variables expected (optional / recommended):
  - OLLAMA_MODEL (default 'llama3.2')
  - PERSPECTIVE_API_KEY
  - SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
  - YOUTUBE_API_KEY
  - MONGO_URI
  - GOOGLE_APPLICATION_CREDENTIALS (for firebase-admin) - optional
  - FIREBASE_CRED_JSON (alternatively)
"""

import os
import sys
import time
import json
import requests
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

# Optional imports (fail gracefully)
try:
    import ollama
except Exception:
    ollama = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None

try:
    from langdetect import detect
except Exception:
    detect = None

try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None

try:
    import firebase_admin
    from firebase_admin import auth as firebase_auth, credentials as firebase_credentials
except Exception:
    firebase_admin = None
    firebase_auth = None
    firebase_credentials = None

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except Exception:
    spotipy = None
    SpotifyClientCredentials = None

try:
    from googleapiclient.discovery import build as build_youtube
except Exception:
    build_youtube = None

# ---------------------------
# Utilities / small helpers
# ---------------------------
def env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name, default)
    if v is None:
        return None
    return v

def safe_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

# ---------------------------
# Translator (lazy loaded)
# ---------------------------
class Translator:
    def __init__(self):
        if pipeline is None or detect is None:
            safe_print("[Translator] transformers/langdetect not installed. Translation disabled.")
        self.to_en: Dict[str, Any] = {}
        self.from_en: Dict[str, Any] = {}

    def _norm(self, lang: str) -> str:
        if not lang:
            return "en"
        return lang.split("-")[0].lower()

    def _load_to_en(self, src: str):
        src = self._norm(src)
        if src in self.to_en:
            return self.to_en[src]
        if pipeline is None:
            return None
        name = f"Helsinki-NLP/opus-mt-{src}-en"
        try:
            p = pipeline("translation", model=name)
            self.to_en[src] = p
            return p
        except Exception:
            return None

    def _load_from_en(self, tgt: str):
        tgt = self._norm(tgt)
        if tgt in self.from_en:
            return self.from_en[tgt]
        if pipeline is None:
            return None
        name = f"Helsinki-NLP/opus-mt-en-{tgt}"
        try:
            p = pipeline("translation", model=name)
            self.from_en[tgt] = p
            return p
        except Exception:
            return None

    def translate_to_en(self, text: str) -> (str, str):
        if not text:
            return "", "en"
        try:
            lang = detect(text) if detect is not None else "en"
        except Exception:
            lang = "en"
        lang = self._norm(lang)
        if lang == "en":
            return text, "en"
        pipe = self._load_to_en(lang)
        if pipe is None:
            return text, lang
        try:
            out = pipe(text, max_length=400)
            return out[0].get("translation_text"), lang
        except Exception:
            return text, lang

    def translate_from_en(self, text: str, target_lang: str) -> str:
        if not text:
            return ""
        tgt = self._norm(target_lang)
        if tgt == "en":
            return text
        pipe = self._load_from_en(tgt)
        if pipe is None:
            return text
        try:
            out = pipe(text, max_length=400)
            return out[0].get("translation_text")
        except Exception:
            return text

# ---------------------------
# Sentiment Analyzer (lazy)
# ---------------------------
class SentimentAnalyzer:
    def __init__(self, threshold: float = 0.65, model_name: Optional[str] = None):
        self.threshold = threshold
        self.model_name = model_name or "distilbert-base-uncased-finetuned-sst-2-english"
        self.pipe = None

    def _ensure(self):
        if self.pipe is None and pipeline is not None:
            try:
                self.pipe = pipeline("sentiment-analysis", model=self.model_name)
            except Exception:
                try:
                    self.pipe = pipeline("sentiment-analysis")
                except Exception:
                    self.pipe = None
        return self.pipe

    def analyze(self, text: str) -> Dict[str, Any]:
        if not text:
            return {"label": "neutral", "score": 0.0, "mood": "neutral"}
        pipe = self._ensure()
        if pipe is None:
            return {"label": "neutral", "score": 0.0, "mood": "neutral"}
        try:
            res = pipe(text, truncation=True)
            if isinstance(res, list) and len(res) > 0:
                label = res[0].get("label", "")
                score = float(res[0].get("score", 0.0))
            else:
                label = str(res.get("label", "")) if isinstance(res, dict) else ""
                score = float(res.get("score", 0.0)) if isinstance(res, dict) else 0.0
            mood = "neutral"
            lu = label.upper()
            if lu in ("POSITIVE", "LABEL_1", "POS"):
                mood = "happy" if score >= self.threshold else "neutral"
            elif lu in ("NEGATIVE", "LABEL_0", "NEG"):
                mood = "sad" if score >= self.threshold else "neutral"
            return {"label": label, "score": score, "mood": mood}
        except Exception:
            return {"label": "neutral", "score": 0.0, "mood": "neutral"}

# ---------------------------
# Perspective API moderator
# ---------------------------
class PerspectiveModerator:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or env("PERSPECTIVE_API_KEY")
        if not self.api_key:
            safe_print("[Moderator] PERSPECTIVE_API_KEY not set. Perspective moderation disabled.")
        self.url_base = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

    def analyze(self, text: str, attributes: Optional[List[str]] = None) -> Dict[str, float]:
        """Return attribute -> score 0..1. Basic attributes default to TOXICITY."""
        if not text:
            return {}
        if not self.api_key:
            return {}
        attrs = attributes or ["TOXICITY"]
        requestedAttributes = {a: {} for a in attrs}
        payload = {
            "comment": {"text": text},
            "requestedAttributes": requestedAttributes,
            "doNotStore": True
        }
        try:
            resp = requests.post(self.url_base, params={"key": self.api_key}, json=payload, timeout=8.0)
            resp.raise_for_status()
            data = resp.json()
            scores = {}
            for a in attrs:
                try:
                    val = data["attributeScores"][a]["summaryScore"]["value"]
                    scores[a] = float(val)
                except Exception:
                    scores[a] = 0.0
            return scores
        except Exception as e:
            safe_print("[Perspective] error:", e)
            return {}

# ---------------------------
# Music clients: Spotify (primary) + YouTube (fallback)
# ---------------------------
class SpotifyClient:
    def __init__(self, client_id: Optional[str], client_secret: Optional[str]):
        self.client_id = client_id
        self.client_secret = client_secret
        self.client = None
        if spotipy is None or SpotifyClientCredentials is None:
            safe_print("[Spotify] spotipy not installed; Spotify features disabled.")
            return
        if not (client_id and client_secret):
            safe_print("[Spotify] SPOTIFY_CLIENT_ID/SECRET missing; Spotify disabled.")
            return
        try:
            creds = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
            self.client = spotipy.Spotify(client_credentials_manager=creds)
        except Exception as e:
            safe_print("[Spotify] init error:", e)
            self.client = None

    def get_top_tracks_from_playlist_name(self, playlist_name_query="Top 50 - Global", limit=10):
        """Try to find a named playlist (e.g., 'Top 50 - Global') and return top tracks."""
        if not self.client:
            return []
        try:
            q = self.client.search(q=playlist_name_query, type="playlist", limit=1)
            items = q.get("playlists", {}).get("items", [])
            if not items:
                return []
            playlist_id = items[0]["id"]
            tracks = self.client.playlist_tracks(playlist_id, limit=limit).get("items", [])
            results = []
            for t in tracks:
                tr = t.get("track") or {}
                results.append({
                    "name": tr.get("name"),
                    "artists": [a.get("name") for a in tr.get("artists", [])],
                    "spotify_url": tr.get("external_urls", {}).get("spotify")
                })
            return results
        except Exception as e:
            safe_print("[Spotify] error fetching playlist:", e)
            return []

    def search_tracks(self, query: str, limit=5):
        if not self.client:
            return []
        try:
            res = self.client.search(q=query, type="track", limit=limit)
            items = res.get("tracks", {}).get("items", [])
            out = []
            for tr in items:
                out.append({
                    "name": tr.get("name"),
                    "artists": [a.get("name") for a in tr.get("artists", [])],
                    "spotify_url": tr.get("external_urls", {}).get("spotify")
                })
            return out
        except Exception as e:
            safe_print("[Spotify] search error:", e)
            return []

class YouTubeClient:
    def __init__(self, api_key: Optional[str]):
        self.api_key = api_key
        self.client = None
        if build_youtube is None:
            safe_print("[YouTube] google-api-python-client not installed; YouTube features disabled.")
            return
        if not api_key:
            safe_print("[YouTube] YOUTUBE_API_KEY not set; YouTube disabled.")
            return
        try:
            self.client = build_youtube("youtube", "v3", developerKey=api_key)
        except Exception as e:
            safe_print("[YouTube] init error:", e)
            self.client = None

    def get_trending(self, regionCode="US", limit=5):
        if not self.client:
            return []
        try:
            resp = self.client.videos().list(part="snippet,statistics", chart="mostPopular", regionCode=regionCode, maxResults=limit).execute()
            items = resp.get("items", [])
            out = []
            for it in items:
                snip = it.get("snippet", {})
                out.append({
                    "title": snip.get("title"),
                    "channel": snip.get("channelTitle"),
                    "url": f"https://www.youtube.com/watch?v={it.get('id')}"
                })
            return out
        except Exception as e:
            safe_print("[YouTube] trending error:", e)
            return []

    def search_videos(self, query, limit=5):
        if not self.client:
            return []
        try:
            res = self.client.search().list(part="snippet", q=query, maxResults=limit, type="video").execute()
            items = res.get("items", [])
            out = []
            for it in items:
                out.append({
                    "title": it["snippet"]["title"],
                    "channel": it["snippet"]["channelTitle"],
                    "url": f"https://www.youtube.com/watch?v={it['id']['videoId']}"
                })
            return out
        except Exception as e:
            safe_print("[YouTube] search error:", e)
            return []

# ---------------------------
# Music manager: prefer Spotify, fallback to YouTube
# ---------------------------
class MusicManager:
    def __init__(self):
        self.spotify = SpotifyClient(env("SPOTIFY_CLIENT_ID"), env("SPOTIFY_CLIENT_SECRET"))
        self.youtube = YouTubeClient(env("YOUTUBE_API_KEY"))

    def trending(self, prefer="spotify", limit=6, region="US"):
        """Return a list of items with name/artist/url. Prefer spotify if available."""
        if prefer == "spotify" and self.spotify.client:
            tracks = self.spotify.get_top_tracks_from_playlist_name(limit=limit)
            if tracks:
                return {"source": "spotify", "items": tracks}
        # fallback YouTube trending
        vids = self.youtube.get_trending(regionCode=region, limit=limit)
        return {"source": "youtube", "items": vids}

    def search_music(self, query, prefer="spotify", limit=6):
        # prefer Spotify track search
        if prefer == "spotify" and self.spotify.client:
            tracks = self.spotify.search_tracks(query, limit=limit)
            if tracks:
                return {"source": "spotify", "items": tracks}
        # fallback: search YouTube videos
        if self.youtube.client:
            vids = self.youtube.search_videos(query, limit=limit)
            if vids:
                return {"source": "youtube", "items": vids}
        return {"source": "none", "items": []}

# ---------------------------
# Database wrapper: MongoDB with local JSON fallback
# ---------------------------
class DBClient:
    def __init__(self, uri: Optional[str] = None, dbname: str = "grace_db"):
        self.uri = uri or env("MONGO_URI")
        self.client = None
        self.db = None
        self.local_file = "grace_local_db.json"
        self._local_cache = {}
        if self.uri and MongoClient is not None:
            try:
                self.client = MongoClient(self.uri)
                self.db = self.client.get_database(dbname)
                self.users = self.db.get_collection("users")
                self.chats = self.db.get_collection("chats")
            except Exception as e:
                safe_print("[DB] Mongo init error:", e)
                self.client = None
                self.db = None
        else:
            safe_print("[DB] No MONGO_URI or pymongo not installed. Using local JSON fallback.")

        # load local cache
        try:
            if os.path.exists(self.local_file):
                with open(self.local_file, "r", encoding="utf8") as f:
                    self._local_cache = json.load(f)
        except Exception:
            self._local_cache = {}

    def _save_local(self):
        try:
            with open(self.local_file, "w", encoding="utf8") as f:
                json.dump(self._local_cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            safe_print("[DB] local save error:", e)

    def upsert_user(self, user_id: str, data: Dict[str, Any]):
        if self.db:
            try:
                self.users.update_one({"user_id": user_id}, {"$set": data}, upsert=True)
                return True
            except Exception as e:
                safe_print("[DB] upsert_user mongo error:", e)
                return False
        self._local_cache.setdefault("users", {})
        self._local_cache["users"][user_id] = {**self._local_cache["users"].get(user_id, {}), **data}
        self._save_local()
        return True

    def get_user(self, user_id: str) -> Dict[str, Any]:
        if self.db:
            try:
                doc = self.users.find_one({"user_id": user_id}) or {}
                doc.pop("_id", None)
                return doc
            except Exception as e:
                safe_print("[DB] get_user mongo error:", e)
                return {}
        return self._local_cache.get("users", {}).get(user_id, {})

    def append_chat(self, user_id: str, role: str, text: str, meta: Optional[Dict[str, Any]] = None):
        rec = {"user_id": user_id, "role": role, "text": text, "ts": int(time.time()), "meta": meta or {}}
        if self.db:
            try:
                self.chats.insert_one(rec)
                return True
            except Exception as e:
                safe_print("[DB] append_chat mongo error:", e)
                return False
        self._local_cache.setdefault("chats", []).append(rec)
        self._save_local()
        return True

    # ✅ NEW METHOD
    def get_chat_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch the last N messages for a given user_id."""
        if self.db:
            try:
                cursor = (
                    self.chats.find({"user_id": user_id})
                    .sort("ts", -1)  # newest first
                    .limit(limit)
                )
                chats = list(cursor)
                for c in chats:
                    c.pop("_id", None)  # strip Mongo’s internal ID
                return list(reversed(chats))  # return oldest → newest order
            except Exception as e:
                safe_print("[DB] get_chat_history mongo error:", e)
                return []
        # fallback: JSON cache
        chats = [c for c in self._local_cache.get("chats", []) if c.get("user_id") == user_id]
        chats_sorted = sorted(chats, key=lambda x: x.get("ts", 0), reverse=True)
        return list(reversed(chats_sorted[:limit]))

# ---------------------------
# Auth manager: Firebase verification (server-side) or guest fallback
# ---------------------------
class AuthManager:
    def __init__(self, cred_json_path: Optional[str] = None):
        path = cred_json_path or env("FIREBASE_CRED_JSON") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        self.initialized = False
        if firebase_admin is None:
            safe_print("[Auth] firebase-admin not installed; Auth disabled.")
            return
        if not path or not os.path.exists(path):
            safe_print("[Auth] Firebase credential JSON not found; Auth disabled.")
            return
        try:
            cred = firebase_credentials.Certificate(path)
            firebase_admin.initialize_app(cred)
            self.initialized = True
        except Exception as e:
            safe_print("[Auth] firebase init error:", e)
            self.initialized = False

    def verify_token(self, id_token: str) -> Optional[Dict[str, Any]]:
        if not self.initialized or firebase_auth is None:
            return None
        try:
            decoded = firebase_auth.verify_id_token(id_token)
            return decoded
        except Exception as e:
            safe_print("[Auth] verify_token error:", e)
            return None

# ---------------------------
# Grace core (integrates everything)
# ---------------------------
@dataclass
class GraceConfig:
    name: str = "Grace"
    model: str = env("OLLAMA_MODEL") or "llama3.2"
    # keep these in config but not passed directly to ollama.chat()
    temperature: float = 0.7
    max_tokens: int = 500
    soft_blocks: List[str] = field(default_factory=lambda: [
        "nsfw", "nude", "sex", "suicide", "self-harm", "kill myself", "drugs"
    ])
    perspective_thresh_block: float = 0.85
    perspective_thresh_warn: float = 0.6

class Grace:
    def __init__(self, cfg: Optional[GraceConfig] = None):
        self.cfg = cfg or GraceConfig()
        self.history: List[Dict[str, str]] = [{"role": "system", "content": (
            "You are Grace, a friendly, teen-relatable, and safe chat buddy. Keep tone warm, casual, and inclusive."
        )}]
        # services
        self.translator = Translator()
        self.sentiment = SentimentAnalyzer()
        self.moderator = PerspectiveModerator(env("PERSPECTIVE_API_KEY"))
        self.music = MusicManager()
        self.db = DBClient(env("MONGO_URI"))
        self.auth = AuthManager()
        # session map: session_id -> user_id for simple CLI sessions
        self._last_session_user = {}

    def _is_soft_unsafe(self, text: str) -> bool:
        if not text:
            return False
        lower = text.lower()
        return any(k in lower for k in self.cfg.soft_blocks)

    def _append(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def _prune_history(self, turns=6):
        system = [m for m in self.history if m["role"] == "system"]
        others = [m for m in self.history if m["role"] != "system"]
        keep = turns * 2
        self.history = system + others[-keep:]

    def _detect_music_request(self, text: str) -> Dict[str, str]:
        """
        Detect whether user is asking for a specific track or trending music.
        Returns a dict: {"type": "search"|"trending"|"none", "query": original_text}
        """
        t = (text or "").lower()
        # trending indicators
        trending_kw = ["trending", "top", "popular", "hot right now", "chart", "top 50", "what's trending"]
        if any(k in t for k in trending_kw):
            return {"type": "trending", "query": text}

        # explicit search indicators
        search_kw = ["song", "music", "track", "by", "play", "listen", "spotify", "artists", "artist"]
        if any(k in t for k in search_kw):
            return {"type": "search", "query": text}

        return {"type": "none", "query": text}

    def _format_music_response(self, res: Dict[str, Any], original_query: Optional[str] = None) -> str:
        # nicer formatting: if search and single top item, produce focused reply
        src = res.get("source")
        items = res.get("items", []) or []
        if src == "spotify":
            if original_query and len(items) > 0:
                # try to find best match
                top = items[0]
                artists = ", ".join(top.get("artists", []))
                return f"Found on Spotify: {top.get('name')} — {artists}\nListen: {top.get('spotify_url')}"
            lines = [f"Top picks (from Spotify):"]
            for i, it in enumerate(items, 1):
                artists = ", ".join(it.get("artists", []))
                lines.append(f"{i}. {it.get('name')} — {artists} ({it.get('spotify_url')})")
            return "\n".join(lines)
        elif src == "youtube":
            if original_query and len(items) > 0:
                top = items[0]
                return f"Found on YouTube: {top.get('title')} — {top.get('channel')}\nWatch: {top.get('url')}"
            lines = [f"Trending videos (YouTube):"]
            for i, v in enumerate(items, 1):
                lines.append(f"{i}. {v.get('title')} — {v.get('channel')} ({v.get('url')})")
            return "\n".join(lines)
        else:
            return "Sorry — I couldn't find music right now. Try another query."

    def _moderate_text(self, text: str) -> Dict[str, float]:
        # Check with Perspective if available; otherwise fallback to soft-block check
        scores = self.moderator.analyze(text, attributes=["TOXICITY"])
        if not scores:
            # fallback approximate: return 1.0 if soft block found else 0.0
            return {"TOXICITY": 1.0} if self._is_soft_unsafe(text) else {"TOXICITY": 0.0}
        return scores

    def _rephrase_output(self, bad_text_en: str) -> str:
        """
        Use the local LLM to rephrase / sanitize assistant output.
        If Ollama not available, return a safe canned message.
        """
        try:
            if ollama is None:
                return "I want to keep our chat safe — let's switch topics or talk about how you're feeling. 💛"
            # ask the model to sanitize
            system_msg = {"role": "system", "content": "You are a safety filter assistant: rewrite the following assistant message so it removes any toxic, abusive or harmful language while preserving helpfulness and meaning. Keep it teen-friendly."}
            user_msg = {"role": "user", "content": bad_text_en}
            msgs = [self.history[0], system_msg, user_msg]
            # non-stream call
            resp = ollama.chat(model=self.cfg.model, messages=msgs)
            if isinstance(resp, dict):
                candidate = (resp.get("message") or {}).get("content") or resp.get("response") or ""
            else:
                candidate = getattr(resp, "message", {}).get("content", "") if getattr(resp, "message", None) else str(resp)
            candidate = (candidate or "").strip()
            if not candidate:
                return "I want to keep our chat safe — let's switch topics. 💛"
            return candidate
        except Exception as e:
            safe_print("[Rephrase] error:", e)
            return "I want to keep our chat safe — let's switch topics. 💛"

    def chat(self, user_text: str, session_id: Optional[str] = None, id_token: Optional[str] = None, stream: bool = True) -> str:
        """
        session_id: optional CLI/web session identifier -> maps to user_id (guest fallback)
        id_token: optional Firebase ID token to verify user identity
        """
        # 0) Resolve user identity
        user_id = None
        if id_token and self.auth and firebase_auth:
            decoded = self.auth.verify_token(id_token)
            if decoded:
                user_id = decoded.get("uid")
        if session_id:
            # if we have previously tied a session -> user
            user_id = self._last_session_user.get(session_id, user_id or f"guest_{session_id}")
            self._last_session_user[session_id] = user_id
        if not user_id:
            user_id = "guest"

        # Quick music intent detect (improved)
        music_req = self._detect_music_request(user_text)
        if music_req["type"] in ("search", "trending"):
            # Choose search or trending
            if music_req["type"] == "search":
                res = self.music.search_music(music_req["query"], prefer="spotify", limit=3)
            else:
                res = self.music.trending(prefer="spotify", limit=6)
            out = self._format_music_response(res, original_query=music_req["query"])
            # store preference / last action
            self.db.append_chat(user_id, "user", user_text)
            self.db.append_chat(user_id, "assistant", out)
            return out

        # 1) Run Perspective moderation on user input
        user_scores = self._moderate_text(user_text)
        toxicity = user_scores.get("TOXICITY", 0.0)
        if toxicity >= self.cfg.perspective_thresh_block:
            safe_msg = ("I want to keep our chat safe. I can't help with that. "
                        "If someone is in danger or you're feeling unsafe, please contact a trusted adult or a professional. 💛")
            self.db.append_chat(user_id, "user", user_text, meta={"toxicity": toxicity})
            self.db.append_chat(user_id, "assistant", safe_msg, meta={"blocked": True})
            return safe_msg
        elif toxicity >= self.cfg.perspective_thresh_warn:
            # gentle redirect for moderately toxic input
            warn_msg = ("That message might be hurtful — let's try to keep things kind. "
                        "If you want to vent, I can listen and try to help. 💬")
            self.db.append_chat(user_id, "user", user_text, meta={"toxicity": toxicity})
            self.db.append_chat(user_id, "assistant", warn_msg, meta={"warned": True})
            return warn_msg

        # 2) Translate user -> English (if needed)
        english_text, user_lang = self.translator.translate_to_en(user_text)

        # 3) Sentiment detection on english_text
        sentiment = self.sentiment.analyze(english_text)
        mood = sentiment.get("mood", "neutral")

        # 4) Safety soft-check on english_text
        if self._is_soft_unsafe(english_text):
            safe_msg = ("Let's keep our conversation safe and friendly. If you're upset, I'm here to listen. 💛")
            self.db.append_chat(user_id, "user", user_text)
            self.db.append_chat(user_id, "assistant", safe_msg)
            return self.translator.translate_from_en(safe_msg, user_lang)

        # 5) Append English user text into history
        self._append("user", english_text)
        self._prune_history()

        # 6) Create per-call tone injection based on mood
        tone_instr = None
        if mood == "sad":
            tone_instr = {"role": "system", "content": (
                "User appears sad. For this response, be empathetic, validate feelings, and offer supportive suggestions; keep language simple and kind."
            )}
        elif mood == "happy":
            tone_instr = {"role": "system", "content": (
                "User appears happy. For this response, be upbeat and encouraging."
            )}

        messages_for_call = [self.history[0]]
        if tone_instr:
            messages_for_call.append(tone_instr)
        messages_for_call.extend(self.history[1:])

        # 7) Query local Ollama model
        final_text_en = ""
        try:
            if ollama is None:
                raise RuntimeError("Ollama python client not available.")
            # choose streaming only when user's language is English (streaming translation is fragile)
            stream_effective = stream and (user_lang == "en")
            if stream_effective:
                pieces = []
                for chunk in ollama.chat(model=self.cfg.model, messages=messages_for_call, stream=True):
                    # robust chunk extraction
                    token = ""
                    try:
                        token = (chunk.get("message") or {}).get("content") or chunk.get("response") or ""
                    except Exception:
                        token = ""
                    if token:
                        sys.stdout.write(token)
                        sys.stdout.flush()
                        pieces.append(token)
                print()
                final_text_en = "".join(pieces).strip()
            else:
                resp = ollama.chat(model=self.cfg.model, messages=messages_for_call)
                if isinstance(resp, dict):
                    final_text_en = (resp.get("message") or {}).get("content", "") or resp.get("response", "") or ""
                else:
                    try:
                        final_text_en = getattr(resp, "message", {}).get("content", "")
                    except Exception:
                        final_text_en = str(resp)
                final_text_en = (final_text_en or "").strip()
        except Exception as e:
            safe_print("[Grace Error] Ollama call failed:", e)
            fallback = "Hmm, I hit a snag with my brain. Could you try rephrasing? 🙏"
            self.db.append_chat(user_id, "assistant", fallback)
            return fallback

        # 8) Run Perspective on assistant output; if toxic -> rephrase or block
        out_scores = self._moderate_text(final_text_en)
        out_toxicity = out_scores.get("TOXICITY", 0.0)
        if out_toxicity >= self.cfg.perspective_thresh_block:
            # rephrase via LLM safety rewriter
            safe_text = self._rephrase_output(final_text_en)
            final_text_en = safe_text
            self.db.append_chat(user_id, "assistant", safe_text, meta={"rephrased": True})
        elif out_toxicity >= self.cfg.perspective_thresh_warn:
            # gentle replacement
            final_text_en = "Let's keep things positive. Tell me more about what's going on or ask me for music or hobbies!"
            self.db.append_chat(user_id, "assistant", final_text_en, meta={"rephrase_warn": True})
        else:
            # normal store
            self.db.append_chat(user_id, "assistant", final_text_en)

        # 9) Translate back to user's language if needed
        final_out = final_text_en
        if user_lang != "en":
            final_out = self.translator.translate_from_en(final_text_en, user_lang)
            if not stream_effective:
                print(final_out)

        return final_out

# ---------------------------
# CLI runner (simple)
# ---------------------------
def run_cli():
    cfg = GraceConfig()
    grace = Grace(cfg)
    print(f"{cfg.name}: Hey, I'm {cfg.name}! Chat with me. (type 'exit' to leave) ✨")
    session_id = "cli"
    while True:
        try:
            user_text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 👋")
            break
        if user_text.strip().lower() in {"exit", "quit"}:
            print("Goodbye! 👋")
            break
        print(f"{cfg.name}: ", end="", flush=True)
        out = grace.chat(user_text, session_id)
        if out:
            # if not already streamed
            if not user_text.strip().lower().endswith("..."):
                print(out)

if __name__ == "__main__":
    run_cli()
