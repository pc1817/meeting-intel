import json
import streamlit as st
from groq import Groq

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

SYSTEM = """You are a meeting analyst. Extract only what was explicitly said.
Never infer owners or dates. Return ONLY JSON:
{"summary": "string",
 "decisions": [{"text": "string", "evidence_quote": "exact words from transcript"}],
 "actions": [{"task": "string", "owner": "string or null", "due": "string or null",
              "evidence_quote": "exact words from transcript"}],
 "open_questions": ["string"]}"""

st.title("Meeting Intelligence")
audio = st.file_uploader("Upload a short meeting recording", type=["mp3", "wav", "m4a"])
@st.cache_data(ttl=3600)
def chat_models():
    ids = [m.id for m in client.models.list().data]
    skip = ("whisper", "guard", "tts", "orpheus", "playai", "distil")
    return sorted(i for i in ids if not any(s in i for s in skip))

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
        result = json.loads(resp.choices[0].message.content)

    st.subheader("Summary")
    st.write(result["summary"])
    st.subheader("Action items")
    st.dataframe(result["actions"])
    st.subheader("Decisions")
    st.json(result["decisions"])

    # grounding check: the seed of your eval harness
    bad = [a for a in result["actions"] if a["evidence_quote"].lower() not in transcript.lower()]
    st.metric("Ungrounded action items", f"{len(bad)} / {len(result['actions'])}")
