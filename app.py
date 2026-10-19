import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st
from groq import Groq

from scoring import aggregate, grounded, score_clip, REQ

PROMPT_VERSION = "v2"  # bump when you change SYSTEM, and note it with every result
ROOT = Path(__file__).parent

SYSTEM = """You are a meeting analyst. Extract only what was explicitly said.
Never infer owners or dates. A decision is a choice the group made; a task assigned
to a person is an action, never a decision. Use EXACTLY these key names.
Return ONLY JSON:
{"summary": "string",
 "decisions": [{"text": "string", "evidence_quote": "exact words from transcript"}],
 "actions": [{"task": "string", "owner": "string or null", "due": "string or null",
              "evidence_quote": "exact words from transcript"}],
 "open_questions": ["string"]}"""

client = Groq(api_key=st.secrets["GROQ_API_KEY"])


@st.cache_data(ttl=3600)
def chat_models():
    ids = [m.id for m in client.models.list().data]
    skip = ("whisper", "guard", "tts", "orpheus", "playai", "distil")
    return sorted(i for i in ids if not any(s in i for s in skip))


def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
        return None


def extract(model, transcript):
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": transcript[:20000]}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    latency = time.time() - t0
    raw = resp.choices[0].message.content
    usage = getattr(resp, "usage", None)
    return parse_json(raw), raw, latency, getattr(usage, "total_tokens", None)


def load_clips():
    clips = []
    for gp in sorted((ROOT / "data" / "gold").glob("*.json")):
        tp = ROOT / "data" / "transcripts" / (gp.stem + ".txt")
        if tp.exists():
            clips.append((gp.stem, json.loads(gp.read_text(encoding="utf-8")),
                          tp.read_text(encoding="utf-8").strip()))
    return clips


st.title("Meeting Intelligence")
tab_run, tab_eval = st.tabs(["Analyze audio", "Evaluate models"])
models = chat_models()

# ---------------- Tab 1: analyze one recording ----------------
with tab_run:
    audio = st.file_uploader("Upload a short meeting recording", type=["mp3", "wav", "m4a"])
    model = st.selectbox("Model", models, key="run_model")

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
            result, raw, latency, tokens = extract(model, transcript)
        if result is None:
            st.error("Model did not return valid JSON.")
            st.code(raw)
            st.stop()

        st.subheader("Summary")
        st.write(result.get("summary", ""))
        st.subheader("Decisions")
        st.json(result.get("decisions", []))

        st.subheader("Action items")
        actions = [a for a in result.get("actions", []) if isinstance(a, dict)]
        rows = []
        for a in actions:
            ok, score = grounded(a.get("evidence_quote", ""), transcript)
            rows.append({**a, "grounded": ok, "match": round(score, 2)})
        st.dataframe(rows)

        c1, c2 = st.columns(2)
        c1.metric("Ungrounded actions", f"{sum(not r['grounded'] for r in rows)} / {len(rows)}")
        c2.metric("Schema-valid actions", f"{sum(REQ <= set(a) for a in actions)} / {len(actions)}")

        st.subheader("Open questions")
        st.write(result.get("open_questions", []))
        st.caption(f"model: {model} | prompt: {PROMPT_VERSION} | {latency:.1f}s | tokens: {tokens}")

# ---------------- Tab 2: evaluate models on the gold set ----------------
with tab_eval:
    clips = load_clips()
    st.write(f"{len(clips)} clips loaded from `data/` (transcripts paired with gold files).")
    chosen = st.multiselect("Models to compare", models,
                            help="Each model runs once per clip. Start with 1-2 models to save quota.")
    st.caption(f"{len(clips) * len(chosen)} API calls will be made. Prompt version: {PROMPT_VERSION}.")

    if st.button("Run evaluation", disabled=not (clips and chosen)):
        results = {}
        bar = st.progress(0.0)
        total, done = len(clips) * len(chosen), 0
        for m in chosen:
            rows = []
            for stem, gold, tr in clips:
                try:
                    res, raw, lat, tok = extract(m, tr)
                    err = None
                except Exception as e:  # rate limit, 404, etc.
                    res, raw, lat, tok, err = None, "", None, None, str(e)[:300]
                rows.append({"clip": stem, "result": res, "raw": raw if res is None else "",
                             "latency_s": lat, "tokens": tok, "error": err,
                             "score": score_clip(gold, res, tr)})
                done += 1
                bar.progress(done / total)
                time.sleep(1)
            results[m] = rows
        st.session_state["eval"] = {
            "prompt_version": PROMPT_VERSION,
            "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "results": results,
        }

    ev = st.session_state.get("eval")
    if ev:
        st.subheader("Scoreboard")
        board = {}
        for m, rows in ev["results"].items():
            agg = aggregate([r["score"] for r in rows],
                            [r["latency_s"] for r in rows if r["latency_s"] is not None])
            agg["api_errors"] = sum(1 for r in rows if r["error"])
            board[m] = agg
        st.dataframe(pd.DataFrame(board).T)
        st.caption("clips_clean = no missed required actions, wrong owners, trap hits, invented owners, "
                   "extra actions or false decisions. Due dates are scored separately (due_acc_%).")

        st.subheader("Clean clips (trap test)")
        matrix = {m: {r["clip"]: ("PASS" if r["score"]["clean"] else "FAIL") for r in rows}
                  for m, rows in ev["results"].items()}
        st.dataframe(pd.DataFrame(matrix).T)

        st.subheader("Inspect a clip")
        c1, c2 = st.columns(2)
        pick_m = c1.selectbox("Model", list(ev["results"]), key="inspect_model")
        pick_c = c2.selectbox("Clip", [r["clip"] for r in ev["results"][pick_m]], key="inspect_clip")
        row = next(r for r in ev["results"][pick_m] if r["clip"] == pick_c)
        if row["error"]:
            st.error(row["error"])
        elif row["result"] is None:
            st.error("Invalid JSON from model")
            st.code(row["raw"])
        else:
            st.dataframe(pd.DataFrame(row["score"]["details"]))
            with st.expander("Raw model output"):
                st.json(row["result"])

        st.download_button(
            "Download results (commit to data/results/)",
            data=json.dumps(ev, indent=2, default=str),
            file_name=f"results_{PROMPT_VERSION}_{ev['run_at'][:10]}.json",
            mime="application/json",
        )
