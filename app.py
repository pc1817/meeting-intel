import json
import re
import difflib
import streamlit as st
from groq import Groq

PROMPT_VERSION = "v2"  # note this in every result you record

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

SYSTEM = """You are a meeting analyst. Extract only what was explicitly said.
Never infer owners or dates. A decision is a choice the group made; a task assigned
to a person is an action, never a decision. Use EXACTLY these key names.
Return ONLY JSON:
{"summary": "string",
 "decisions": [{"text": "string", "evidence_quote": "exact words from transcript"}],
 "actions": [{"task": "string", "owner": "string or null", "due": "string or null",
              "evidence_quote": "exact words from transcript"}],
 "open_questions": ["string"]}"""


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()


def grounded(quote, transcript, thr=0.9):
    q, t = norm(quote), norm(transcript)
    n = len(q)
    if n == 0:
        return False, 0.0
    best = 0.0
    for i in range(max(1, len(t) - n + 1)):
        best = max(best, difflib.SequenceMatcher(None, q, t[i:i + n]).ratio())
    return best >= thr, best


@st.cache_data(ttl=3600)
def chat_models():
    ids = [m.id for m in client.models.list().data]
    skip = ("whisper", "guard", "tts", "orpheus", "playai", "distil")
    return sorted(i for i in ids if not any(s in i for s in skip))


st.title("Meeting Intelligence")
audio = st.file_uploader("Upload a short meeting recording", type=["mp3", "wav", "m4a"])
model = st.selectbox("Model", chat_models())

if audio and st.button("Analyze"):
    with st.spinner("Transcribing..."):
        tr = client.audio.transcriptions.create(
            file=(audio.name, audio.read()),
            model="whisper-large-v3-turbo",
        )
        transcript = tr.text
    with st.expander("Transcript"):
        st.write(transcript)

    with st.spinner("Extracting..."):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": transcript[:20000]}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        try:
            result = json.loads(resp.choices[0].message.content)
        except json.JSONDecodeError:
            st.error("Model did not return valid JSON.")
            st.code(resp.choices[0].message.content)
            st.stop()

    st.subheader("Summary")
    st.write(result.get("summary", ""))

    st.subheader("Decisions")
    st.json(result.get("decisions", []))

    st.subheader("Action items")
    actions = result.get("actions", [])
    rows = []
    for a in actions:
        ok, score = grounded(a.get("evidence_quote", ""), transcript)
        rows.append({**a, "grounded": ok, "match": round(score, 2)})
    st.dataframe(rows)

    REQ = {"task", "owner", "due", "evidence_quote"}
    c1, c2 = st.columns(2)
    c1.metric("Ungrounded actions", f"{sum(not r['grounded'] for r in rows)} / {len(rows)}")
    c2.metric("Schema-valid actions", f"{sum(REQ <= set(a) for a in actions)} / {len(actions)}")
    st.caption(f"model: {model} | prompt: {PROMPT_VERSION}")
