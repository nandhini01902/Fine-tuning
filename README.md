# Booking agent SFT dataset (synthetic)

- `data/train.json` 4,800 · `data/val.json` 600 · `data/test.json` 600 conversations
- Each row: `{"id", "messages", "meta"}`. Messages use the HF/OpenAI chat format with `tool_calls`;
  pass `data/tools.json` as `tools=` to `tokenizer.apply_chat_template`.
- `meta`: scenario, specialty, urgency (routine / urgent / emergency), red_flag, expected_tools,
  decision_turn (index of the routing/escalation turn in `messages`), user_style, n_turns.
- Test split is held out: unseen symptom phrasings, unseen prompts, and a later date range.
- `data/taxonomy.json`: symptom → specialty → urgency table used to generate the data. **Needs clinician review.**
- Regenerate: `python generate.py && python validate.py && python stats.py` (seeded, deterministic).


https://chatgpt.com/share/6b041ed6-a48c-83ee-8925-3eb3b8c3b228
