"""Validate the generated dataset: schema, tool-call consistency, safety invariants, split leakage."""
import json, datetime as dt, re, sys
from collections import Counter
from catalog import SYMPTOMS, CHILD_SYMPTOMS, RED_FLAGS, CRISIS
TOOLS = {t["function"]["name"]: t["function"]["parameters"] for t in json.load(open("data/tools.json"))}
errors = Counter()

def check(ex):
    msgs = ex["messages"]; meta = ex["meta"]
    assert msgs[0]["role"] == "system"
    today = dt.date.fromisoformat(re.search(r"Today is \w+, (\d{4}-\d{2}-\d{2})", msgs[0]["content"]).group(1))
    last_slots, calls = None, []
    for i, m in enumerate(msgs[1:], 1):
        if m["role"] == "assistant" and m.get("tool_calls"):
            f = m["tool_calls"][0]["function"]; name, args = f["name"], f["arguments"]
            calls.append(name)
            sch = TOOLS.get(name)
            if sch is None: errors["unknown_tool"] += 1; continue
            if set(sch["required"]) - set(args): errors["missing_required_arg"] += 1
            if set(args) - set(sch["properties"]): errors["extra_arg"] += 1
            for k, v in args.items():
                enum = sch["properties"].get(k, {}).get("enum")
                if enum and v not in enum: errors[f"bad_enum_{k}"] += 1
            if name == "check_availability":
                a, b = dt.date.fromisoformat(args["date_from"]), dt.date.fromisoformat(args["date_to"])
                if a < today or b < a: errors["bad_date_range"] += 1
                res = json.loads(msgs[i + 1]["content"])["slots"]
                for s in res:
                    d = dt.date.fromisoformat(s["start"][:10])
                    if not (a <= d <= b): errors["slot_outside_range"] += 1
                    if args["location"] != "any" and s["location"] != args["location"]: errors["slot_wrong_location"] += 1
                last_slots = res
                if res:  # next assistant text must offer exactly these slots
                    offer = msgs[i + 2]["content"]
                    if offer.count("\n") - 1 != len(res): errors["offer_count_mismatch"] += 1
                    for s in res:
                        if s["doctor"] not in offer: errors["offer_missing_slot"] += 1
            if name in ("book_appointment", "reschedule_appointment"):
                sid = args.get("slot_id") or args.get("new_slot_id")
                if not last_slots or sid not in {s["slot_id"] for s in last_slots}: errors["booked_unoffered_slot"] += 1
                if name == "book_appointment" and args["specialty"] != meta["specialty"]: errors["specialty_label_mismatch"] += 1
                if name == "book_appointment" and args["urgency"] != meta["urgency"]: errors["urgency_label_mismatch"] += 1
        elif m["role"] in ("user", "assistant") and not m["content"].strip():
            errors["empty_message"] += 1
    if calls != meta["expected_tools"]: errors["expected_tools_mismatch"] += 1
    if meta["red_flag"] and calls: errors["tool_call_in_emergency"] += 1
    if meta["red_flag"] and not re.search(r"911|988", msgs[meta["decision_turn"]]["content"]): errors["emergency_without_number"] += 1
    if meta["decision_turn"] is not None and msgs[meta["decision_turn"]]["role"] != "assistant": errors["bad_decision_turn"] += 1
    if msgs[-1]["role"] != "assistant": errors["ends_not_assistant"] += 1

data = {s: json.load(open(f"data/{s}.json")) for s in ("train", "val", "test")}
for s, rows in data.items():
    for ex in rows: check(ex)
# leakage: held-out phrasings (index 3 / last 2 crisis) must not appear in train/val; test must not reuse train dialogues
held = [x["phrasings"][3] for x in SYMPTOMS + CHILD_SYMPTOMS + RED_FLAGS] + CRISIS[6:]
for s in ("train", "val"):
    blob = "\n".join(m["content"].lower() for ex in data[s] for m in ex["messages"] if m["role"] == "user")
    leaks = [h for h in held if h.lower() in blob]
    if leaks: errors[f"heldout_phrase_in_{s}"] += len(leaks); print(s, "leaks:", leaks[:5])
fp = lambda ex: " | ".join(m["content"] for m in ex["messages"][1:] if m["role"] == "user")
tr = {fp(e) for e in data["train"]}
errors["test_dialogue_dup_of_train"] = sum(fp(e) in tr for e in data["test"])
errors["val_dialogue_dup_of_train"] = sum(fp(e) in tr for e in data["val"])
print("validation:", dict(errors) if any(errors.values()) else "ALL CHECKS PASSED")
sys.exit(1 if any(v for v in errors.values()) else 0)
