# AGENTS.md — ML course labs (Uni/ML)

Guidance for AI agents working in this repository. Follow it unless the user says otherwise.

## 1. What this repo is

University Machine Learning course practical assignments. Each lab is a self-contained Jupyter
notebook in its own folder with its own dataset and Python virtualenv. All prose, comments,
markdown and reports are in **Ukrainian** — keep it that way.

```
ML/
├── L1/   Практичне завдання 1: лінійна регресія, оптимізація, статистичний аналіз
│   ├── ML1.ipynb              dataset: AmesHousing.csv
│   ├── AmesHousing.csv
│   └── .venv/
├── L2/   Практичне завдання 2: логістична регресія (NumPy) + федеративне навчання
    ├── main.ipynb             dataset: Dry_Bean_Dataset.xlsx
    ├── ML-2-Practice.pdf      the assignment spec
    ├── CHANGES.md             summary of changes vs requirements
    ├── THEORY_GUIDE.md        theory-only study guide (links + examples)
    ├── CODE_GUIDE.md          cell-by-cell code walkthrough
    ├── Dry_Bean_Dataset.xlsx
    └── .venv/
└── L3/   Практичне завдання 3: TF-IDF-зважені FastText-ембеддинги для детекції AI-текстів
    ├── main.ipynb                    dataset: ai_vs_human_text_2026.csv
    ├── ML-Practice-3.pdf             the assignment spec
    ├── THEORY_GUIDE.md               theory-only study guide (links + examples)
    ├── CODE_GUIDE.md                 cell-by-cell code walkthrough
    ├── ai_vs_human_text_2026.csv     Kaggle: AI vs Human Text Classification Dataset 2026
    └── .venv/                        Python 3.12 (gensim has no 3.14 wheels)
```

Note the notebook names are inconsistent (`ML1.ipynb` vs `main.ipynb`). Do not rename existing
notebooks unless asked; if you create a new lab, name the notebook `<LabName>.ipynb` or `main.ipynb`
and keep it inside that lab's folder.

## 2. Environment & running

Each lab has its **own** `.venv`. Most are Python 3.14, but **L3 uses Python 3.12** — `gensim`
has no prebuilt wheels and its C extension fails to compile against the 3.14 C API. Create it with
`uv venv --python 3.12 .venv` then `uv pip install --python .venv/bin/python ...`. Always use the
venv's binaries via the `workdir` parameter — never `cd` and never the system Python.

```bash
# execute the notebook in place (embeds all outputs/figures)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace main.ipynb \
  --ExecutePreprocessor.timeout=1800

# quick package check
.venv/bin/pip list
```

Stack: `numpy`, `pandas`, `scikit-learn` (L2), `matplotlib`, `seaborn`, `scipy`, `statsmodels` (L1),
`openpyxl` (L2, for .xlsx), `spacy` + `en_core_web_sm` and `gensim` (L3), `nbformat`/`nbconvert`,
`ipykernel`. Install per-lab if a package is missing; do not assume a package exists.

The reference PDF can't be read directly by the model — extract text first:

```bash
pdftotext -layout ML-2-Practice.pdf /tmp/opencode/ml2.txt
```

Work in `/tmp/opencode` for scratch scripts, not inside lab folders.

## 3. Notebook conventions

### 3.1 Stable repo-wide conventions (always apply)

1. **Language & comments.** All prose, markdown, comments and reports are Ukrainian. Comments are
   expected in this repo (unlike the global default) — explain intent, not what the code literally
   does.
2. **Conclusions after every figure and table** in markdown: what is shown, what pattern is
   observed, why, and how it affects model/hyperparameter choice.
3. **Reproducibility.** Fix a seed (`SEED = 42`) and use isolated `np.random.default_rng(seed)` for
   splits and per-run/per-client training so results don't depend on call order.
4. **No data leakage.** Split into train/validation/test **first**; fit scaling, imputation, encoders
   and feature selection on train only, then apply to val/test.

### 3.2 Task-specific rules — the assignment spec is authoritative

The current lab's spec (PDF) is the single source of truth for implementation constraints: which
libraries are allowed for the model, whether the core must be NumPy-only, vectorization
requirements, preferred preprocessing/encoding, metrics, and required experiments. Read the spec and
follow it; **if it contradicts anything in this file, the spec wins** — and update this file.

For reference, the L1/L2 specs *happened* to require all of the following, but do **not** assume a
future lab repeats them — confirm against its spec:

- **NumPy-only core** (model, loss, gradients, and FL aggregation). `scikit-learn` only for metrics
  (`accuracy_score`, `f1_score`, `log_loss`, ...), stratified splits and auxiliary transforms;
  never `LogisticRegression`/`SGDClassifier`/`LinearRegression`/etc. for the main results.
- **Vectorized** loss/gradients — no `for` loops over samples; loops over epochs, batches, rounds or
  clients are fine (use fancy indexing / broadcasting).
- **Stratified** classification splits with verified class proportions.

## 4. Standard per-lab deliverable set

When finishing or reviewing a lab, aim for:

- `<notebook>.ipynb` — clean, structured, **executed** (all figures/tables embedded, zero errors).
- `THEORY_GUIDE.md` — theory only (ML/Data Science, not code), each concept explained with an
  example in 1–2 paragraphs or linked to a good resource.
- `CODE_GUIDE.md` — cell-by-cell explanation of the notebook, assuming `THEORY_GUIDE.md` is known.

## 5. Workflow when given a lab task/spec

1. Extract and read the spec (`pdftotext -layout`) and map every requirement.
2. Inspect the existing notebook before editing. Use the **`nb` CLI** (nb-cli) instead of raw JSON:
   `nb read <notebook> --no-output` (AI-friendly, cell-ID-annotated view), `nb read <notebook> -i 3`
   (one cell), `nb search <notebook> "<pattern>"` (locate code/errors). Only fall back to `nbformat`
   when `nb` can't express the operation.
3. Check the coverage checklist in section 6 and note gaps.
4. Fix/implement. Both supported routes write notebooks programmatically — never hand-edit JSON:
   - **`nb` cell operations** (preferred for targeted changes): `nb create`, `nb cell add/update/delete`.
     Add cells in batches of ~3–5 per section (multi-cell `@@markdown`/`@@code` sentinels) and verify
     each batch before continuing.
   - **`nbformat` builder script** (preferred for a full rebuild from scratch): one reproducible script
     that emits every cell, as done for L1–L3.
5. Validate:
   - schema — `.venv/bin/python -c "import nbformat as nbf; nbf.validate(nbf.read('main.ipynb', as_version=4))"`
     (the `nb` CLI has no schema validator);
   - errors — `nb search <notebook> "Traceback"`, or scan error outputs:
   ```bash
   .venv/bin/python - <<'EOF'
   import nbformat as nbf
   nb = nbf.read('main.ipynb', as_version=4)
   print(sum(1 for c in nb.cells if c.cell_type=='code'
             for o in c.get('outputs',[]) if o.get('output_type')=='error'))
   EOF
   ```
6. Execute so outputs/figures are embedded. `nb execute` needs the **lab's** kernel selected
   (`nb execute <notebook> --kernel <lab-kernel>`); the guaranteed path in this repo is the venv's
   nbconvert, which uses the lab packages directly:
   ```bash
   .venv/bin/jupyter nbconvert --to notebook --execute --inplace main.ipynb \
     --ExecutePreprocessor.timeout=1800
   ```
7. Confirm figures are embedded and numbers in prose match actual outputs — inspect with
   `nb read <notebook>` (with outputs) / `nb read <notebook> --output-dir ./notebook-outputs` and
   compare against the markdown.

`nb` is suitable for inspection (step 2), cell-level editing/creation (step 4), error search (step 5)
and output inspection (step 7); it does **not** cover spec extraction (step 1) or schema validation
(step 5). Before first use, verify the binary per nb-cli's trust boundary: `command -v nb && nb --version`
(expected `nb` from nb-cli, not an unexpected binary).
