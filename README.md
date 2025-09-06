# Grace-Chatbot-For-Teens
Grace — a teen-friendly, safety-first chatbot built to connect with teens worldwide. Grace speaks casually (emoji-friendly), detects user language and translates on the fly, senses mood to adapt tone, recommends trending music, and stores lightweight profiles and chat history for continuity.

# Key features
Multilingual translation: Helsinki-NLP (Hugging Face) + langdetect for source detection.
Mood-aware responses: Hugging Face sentiment pipeline adjusts tone (empathetic if sad, upbeat if happy).
Safety-first moderation: Perspective API integration when available, plus a keyword-based soft-filter fallback.
Local LLM: Runs on your machine via Ollama (LLaMA family) so you can keep inference private.
Music recommendations: Spotify primary + YouTube fallback for trending songs & videos.
Persistence & auth: MongoDB (Atlas) with JSON fallback; optional Firebase Auth for cross-device profiles.

# Tech stack & quick start
Python, Ollama, Transformers, Spotipy, google-api-python-client, PyMongo, Firebase Admin. Clone the repo, install requirements (pip install -r requirements.txt), set env vars (OLLAMA_MODEL, MONGO_URI, SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET, YOUTUBE_API_KEY, PERSPECTIVE_API_KEY — optional), then python grace_full.py.
Design philosophy
Built to be relatable, safe, and local-first—great for demos, learning, or as a foundation for a production assistant.
Contributions welcome.
