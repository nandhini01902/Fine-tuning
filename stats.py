import json, statistics as st
from collections import Counter
from catalog import SYMPTOMS, CHILD_SYMPTOMS, RED_FLAGS, SPECIALTIES, LOCATIONS
tools = json.load(open("data/tools.json"))
tools_chars = len(json.dumps(tools))
out = {}
for s in ("train", "val", "test"):
    rows = json.load(open(f"data/{s}.json"))
    chars = [sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls", ""))) for m in r["messages"]) + tools_chars for r in rows]
    asst_turns = sum(1 for r in rows for m in r["messages"] if m["role"] == "assistant")
    out[s] = {
        "conversations": len(rows),
        "scenario": dict(Counter(r["meta"]["scenario"] for r in rows).most_common()),
        "specialty": dict(Counter(r["meta"]["specialty"] or "none" for r in rows).most_common()),
        "urgency": dict(Counter(r["meta"]["urgency"] or "none" for r in rows).most_common()),
        "user_style": dict(Counter(r["meta"]["user_style"] for r in rows).most_common()),
        "red_flag_conversations": sum(r["meta"]["red_flag"] for r in rows),
        "assistant_turns": asst_turns,
        "tool_calls": dict(Counter(t for r in rows for t in r["meta"]["expected_tools"]).most_common()),
        "turns_mean": round(st.mean(r["meta"]["n_turns"] for r in rows), 2),
        "turns_max": max(r["meta"]["n_turns"] for r in rows),
        "approx_tokens_mean": round(st.mean(chars) / 3.8),   # rough: ~3.8 chars/token incl. tool schema
        "approx_tokens_max": round(max(chars) / 3.8),
        "approx_tokens_total": round(sum(chars) / 3.8),
    }
json.dump(out, open("data/stats.json", "w"), indent=2)
tax = {"specialties": SPECIALTIES, "locations": LOCATIONS,
       "symptoms": [{k: v for k, v in s.items()} for s in SYMPTOMS + CHILD_SYMPTOMS],
       "red_flags": RED_FLAGS,
       "urgency_rule": "urgent if symptom base urgency is urgent OR reported severity is severe; red flags -> emergency (no booking)",
       "test_holdout": "phrasing index 3 of every symptom/red flag, last 2 crisis phrasings, last 3 medication/injection/off-topic prompts, dates from 2027-04-01"}
json.dump(tax, open("data/taxonomy.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps({s: {k: v for k, v in out[s].items() if k not in ("specialty", "user_style")} for s in out}, indent=1))
print(out["train"]["specialty"])

https://chatgpt.com/c/6afebddc-fd34-83ec-ac50-f4729ef1dfb6