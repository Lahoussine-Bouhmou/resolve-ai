# ResolveAI data: TechQA

The contents of `data/raw/` and `data/processed/` are local and are not
committed to Git.

## Expected TechQA files

Place the extracted TechQA files under:

```text
data/raw/techqa/
├── training_Q_A.json
├── dev_Q_A.json
└── training_dev_technotes.json
```

## Run the audit

```bash
uv run python scripts/audit_techqa.py \
  --corpus data/raw/techqa/training_dev_technotes.json \
  --questions \
    data/raw/techqa/training_Q_A.json \
    data/raw/techqa/dev_Q_A.json
```

Generated reports are written under `data/reports/techqa/`.

## Build the reduced benchmark

```bash
uv run python scripts/build_techqa_subset.py \
  --corpus data/raw/techqa/training_dev_technotes.json \
  --train-questions data/raw/techqa/training_Q_A.json \
  --validation-questions data/raw/techqa/dev_Q_A.json \
  --train-answerable 75 \
  --train-unanswerable 25 \
  --validation-answerable 25 \
  --validation-unanswerable 25 \
  --extra-distractors 1000 \
  --seed 42
```

The generated runtime query files deliberately exclude:

- `DOC_IDS`
- answer document IDs
- answer offsets
- answerability labels

Ground-truth annotations are stored separately and must only be used by
evaluation code.

## Build offset-preserving chunks

```bash
uv run python scripts/build_techqa_chunks.py
```

The generated chunks contain:

- raw document content;
- original start and end character offsets;
- a separate normalized search representation;
- deterministic chunk identifiers.

Ground-truth answer offsets are used only to evaluate whether answer spans are
fully contained in at least one chunk. They are never used to construct chunk
boundaries.