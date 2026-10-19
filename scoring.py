"""Deterministic scoring for Meeting Intelligence (no API calls in this file)."""
import re
import difflib

STOP = {
    "the", "an", "and", "to", "of", "for", "with", "at", "by", "on", "in", "will",
    "can", "should", "we", "is", "are", "be", "that", "this", "it", "so", "as",
    "from", "up", "out", "have", "has", "then", "also", "our", "their", "them",
    "us", "or", "but", "if", "been", "was", "were", "would", "could", "please",
}
NO_OWNER = {"", "team", "everyone", "all", "we", "group", "unassigned", "none",
            "na", "tbd", "null", "nobody", "teamwide"}
REQ = {"task", "owner", "due", "evidence_quote"}


def norm(s):
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split()


def grounded(quote, transcript, thr=0.9):
    q, t = norm(quote), norm(transcript)
    n = len(q)
    if n == 0:
        return False, 0.0
    best = 0.0
    for i in range(max(1, len(t) - n + 1)):
        best = max(best, difflib.SequenceMatcher(None, q, t[i:i + n]).ratio())
    return best >= thr, best


def toks(s):
    out = []
    for w in re.findall(r"[A-Za-z0-9]+", s or ""):
        if len(w) < 2 and not w.isupper():
            continue
        w = w.lower()
        if w not in STOP:
            out.append(w)
    return out


def tmatch(a, b):
    return a == b or (min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)))


def overlap(gold_toks, cand_toks):
    if not gold_toks:
        return 0.0
    hits = sum(any(tmatch(g, c) for c in cand_toks) for g in gold_toks)
    return hits / len(gold_toks)


def name_key(s):
    return re.sub(r"[^a-z]", "", (s or "").lower())


def owner_ok(g, m):
    g, m = name_key(g), name_key(m)
    if not g or not m:
        return False
    return g == m or (min(len(g), len(m)) >= 3 and (g.startswith(m) or m.startswith(g)))


def due_ok(g, m):
    if not g and not m:
        return True
    if not g or not m:
        return False
    return overlap(toks(g), toks(m)) >= 0.5


def _items(lst, key):
    out = []
    if not isinstance(lst, list):
        return out
    for x in lst:
        if isinstance(x, str):
            x = {key: x}
        if isinstance(x, dict):
            out.append(x)
    return out


def _text(x, key):
    return f"{x.get(key, '')} {x.get('evidence_quote', '')}"


def _assign(gold_texts, model_texts, thr=0.5):
    pairs = []
    for gi, gt in enumerate(gold_texts):
        for mi, mt in enumerate(model_texts):
            s = overlap(toks(gt), toks(mt))
            if s >= thr:
                pairs.append((s, gi, mi))
    pairs.sort(reverse=True)
    g_used, m_used, match = set(), set(), {}
    for s, gi, mi in pairs:
        if gi in g_used or mi in m_used:
            continue
        g_used.add(gi)
        m_used.add(mi)
        match[gi] = (mi, s)
    return match, m_used


def score_clip(gold, result, transcript):
    acts = _items(result.get("actions"), "task") if result else []
    decs = _items(result.get("decisions"), "text") if result else []
    d = {"clip_id": gold.get("clip_id"), "parse_ok": result is not None,
         "n_actions": len(acts), "schema_valid": sum(REQ <= set(a) for a in acts),
         "grounded": sum(grounded(a.get("evidence_quote", ""), transcript)[0] for a in acts)}

    g_acts = gold.get("actions", [])
    match, used = _assign([g["task"] for g in g_acts], [_text(a, "task") for a in acts])
    d.update(req_total=0, req_found=0, owner_ok=0, due_ok=0, opt_found=0, invented_owner=0)
    details = []
    for gi, g in enumerate(g_acts):
        opt = bool(g.get("optional"))
        row = {"gold": g["task"], "kind": "optional" if opt else "required", "found": gi in match}
        if not opt:
            d["req_total"] += 1
        if gi in match:
            mi, s = match[gi]
            a = acts[mi]
            row.update(model_task=a.get("task"), model_owner=a.get("owner"),
                       model_due=a.get("due"), match=round(s, 2))
            if opt:
                d["opt_found"] += 1
                if name_key(a.get("owner")) not in NO_OWNER:
                    d["invented_owner"] += 1
                    row["flag"] = "invented owner"
            else:
                d["req_found"] += 1
                if g.get("owner"):
                    o = owner_ok(g.get("owner"), a.get("owner"))
                else:  # gold says nobody was assigned: model must not invent an owner
                    o = name_key(a.get("owner")) in NO_OWNER
                du = due_ok(g.get("due"), a.get("due"))
                d["owner_ok"] += o
                d["due_ok"] += du
                row.update(owner_correct=o, due_correct=du)
        details.append(row)

    d["trap_hits"] = d["extra_actions"] = 0
    for mi, a in enumerate(acts):
        if mi in used:
            continue
        mt = toks(_text(a, "task"))
        if any(overlap(toks(t), mt) >= 0.5 for t in gold.get("not_actions", [])):
            d["trap_hits"] += 1
            details.append({"gold": "(trap)", "kind": "trap hit", "model_task": a.get("task"),
                            "model_owner": a.get("owner")})
        else:
            d["extra_actions"] += 1
            details.append({"gold": "(none)", "kind": "extra", "model_task": a.get("task"),
                            "model_owner": a.get("owner")})

    g_decs = gold.get("decisions", [])
    dmatch, dused = _assign([g["text"] for g in g_decs], [_text(x, "text") for x in decs])
    d["dec_total"], d["dec_found"] = len(g_decs), len(dmatch)
    d["false_decisions"] = 0
    for mi, x in enumerate(decs):
        if mi in dused:
            continue
        d["false_decisions"] += 1
        hit = any(overlap(toks(t), toks(_text(x, "text"))) >= 0.5 for t in gold.get("not_decisions", []))
        details.append({"gold": "(trap)" if hit else "(none)",
                        "kind": "decision trap hit" if hit else "false decision",
                        "model_task": x.get("text")})

    d["clean"] = bool(
        result is not None and d["req_found"] == d["req_total"] and d["owner_ok"] == d["req_total"]
        and d["trap_hits"] == 0 and d["invented_owner"] == 0 and d["extra_actions"] == 0
        and d["false_decisions"] == 0 and d["dec_found"] == d["dec_total"])
    d["details"] = details
    return d


def aggregate(clip_scores, latencies=None):
    s = lambda k: sum(c.get(k, 0) for c in clip_scores)
    pct = lambda a, b: round(100 * a / b, 1) if b else None
    return {
        "clips_clean": f"{sum(c['clean'] for c in clip_scores)}/{len(clip_scores)}",
        "parse_failures": sum(not c["parse_ok"] for c in clip_scores),
        "action_recall_%": pct(s("req_found"), s("req_total")),
        "owner_acc_%": pct(s("owner_ok"), s("req_found")),
        "due_acc_%": pct(s("due_ok"), s("req_found")),
        "decision_recall_%": pct(s("dec_found"), s("dec_total")),
        "trap_hits": s("trap_hits"),
        "invented_owners": s("invented_owner"),
        "extra_actions": s("extra_actions"),
        "false_decisions": s("false_decisions"),
        "schema_valid_%": pct(s("schema_valid"), s("n_actions")),
        "grounded_%": pct(s("grounded"), s("n_actions")),
        "avg_latency_s": round(sum(latencies) / len(latencies), 2) if latencies else None,
    }
