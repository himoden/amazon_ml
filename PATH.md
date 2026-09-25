# Amazon ML Challenge 2026 --- Agent Execution PRD / PATH

## 0. Mission

Build a complete, reproducible Business Entity Resolution pipeline for
the supplied Amazon ML Challenge 2026 dataset.

The agent's job is to:

1.  inspect the actual files;
2.  understand the data before designing around assumptions;
3.  build a correct working baseline;
4.  validate it using the real competition metric;
5.  produce a valid submission;
6.  only then iterate on improvements that demonstrably improve
    validation.

**Do not attempt to build the final "state-of-the-art" system in one
pass.**

The competition deadline is extremely close, so prioritize:

**correct baseline → measurable score → submission → targeted
improvements.**

------------------------------------------------------------------------

# 1. Current workspace

The user's current VS Code/OpenCode workspace resembles:

``` text
amazon_ml/
└── student_resource/
    ├── dataset/
    │   ├── train/
    │   │   ├── train_ground_truth.tsv
    │   │   ├── train_source1.tsv
    │   │   ├── train_source2.tsv
    │   │   └── train_source3.tsv
    │   └── test/
    │       ├── test_source1.tsv
    │       ├── test_source2.tsv
    │       └── test_source3.tsv
    ├── utils/
    │   └── validate_submission.py
    ├── Documentation_template.md
    ├── README.md
    ├── ER design doc.pdf
    └── other user files
```

Do not rename or destroy the supplied challenge files.

Create implementation code separately, preferably:

``` text
student_resource/
└── code/
    └── business_entity_resolution/
        ├── src/
        ├── README.md
        └── requirements.txt
```

Outputs belong in:

``` text
student_resource/output/
```

------------------------------------------------------------------------

# 2. Absolute constraints

Before writing code, read `PS.md` and treat it as the competition
contract.

### Never violate these

-   No external business/entity lookup.
-   No external databases.
-   No geocoding APIs.
-   No commercial ER APIs.
-   No internet business augmentation.
-   Use supplied TSV data and permitted pretrained model weights only.
-   Final model must satisfy MIT/Apache-2.0 and ≤8B parameter
    constraints.
-   Never hard-code the country universe to US/India.
-   Never submit invalid TSVs.
-   Never predict an ID that was not in the candidate set.
-   Never optimize only pairwise F1.
-   Never silently convert assumptions into facts.

If a potentially useful method has a compliance ambiguity, flag it
instead of silently using it.

------------------------------------------------------------------------

# 3. Operating principle for the agent

The agent must behave like an ML engineer/researcher, not a code
generator.

For every major change:

``` text
hypothesis
→ implementation
→ validation experiment
→ metric/result
→ keep or reject
```

Do not add complexity just because it sounds state-of-the-art.

Maintain a short experiment log:

``` text
experiment
change
validation F0.5
blocking recall
candidate count/reduction
decision
```

------------------------------------------------------------------------

# 4. Phase 0 --- Inspect before coding

First inspect:

``` text
train_source1.tsv
train_source2.tsv
train_source3.tsv
train_ground_truth.tsv
test_source1.tsv
test_source2.tsv
test_source3.tsv
utils/validate_submission.py
```

Report:

-   row counts
-   columns
-   null rates
-   countries
-   source sizes
-   number of ground-truth matches per S1
-   singleton rate
-   S2/S3 fan-out
-   whether S2/S3 IDs occur in multiple GT rows
-   country consistency of GT pairs
-   train/test size ratio
-   duplicate patterns
-   name/address length distributions
-   obvious formatting/noise patterns

Do not invent these statistics.

Write them to an EDA report or notebook.

------------------------------------------------------------------------

# 5. Phase 1 --- Implement the scorer FIRST

Before ML:

Create an exact local competition scorer.

It must:

-   construct true match sets from ground truth;
-   construct predicted match sets;
-   calculate per-S1 F0.5;
-   handle singleton behavior according to the verified challenge rule;
-   macro-average across S1;
-   reproduce the official worked example.

Add unit tests.

The scorer is more important than the model because every subsequent
decision depends on it.

------------------------------------------------------------------------

# 6. Phase 2 --- Build a leakage-safe validation framework

Validation must mirror the leaderboard.

Preferred structure from the design document:

``` text
Group split by Source 1 entity
```

Do not split individual pairs randomly because the same S1 entity can
have multiple matches.

Keep the full S2/S3 pool as distractors when evaluating validation S1
entities.

Report:

-   macro F0.5
-   per-country F0.5
-   singleton accuracy
-   non-singleton recall
-   empty-prediction rate for non-singletons
-   blocking recall ceiling
-   candidates per S1
-   reduction ratio

Also implement leave-one-country-out analysis where meaningful.

Do not use the public leaderboard as the primary tuning objective.

------------------------------------------------------------------------

# 7. Phase 3 --- Correct baseline

The first complete working baseline should be CPU-friendly and should
NOT require a GPU.

Pipeline:

``` text
TSV
 ↓
normalization
 ↓
candidate generation
 ↓
pairwise feature generation
 ↓
LightGBM
 ↓
threshold/set selection
 ↓
matching_results.tsv
 ↓
validator
```

The baseline must produce a real submission before advanced models are
attempted.

------------------------------------------------------------------------

# 8. Phase 4 --- Normalization

Implement multiple representations instead of destroying raw data.

Keep:

``` text
raw name/address
normalized name/address
core name
numeric/address tokens
postal-code-like tokens
```

Baseline normalization:

-   Unicode NFKC/NFKD as appropriate
-   casefold/lowercase
-   accent/diacritic normalization
-   punctuation normalization
-   whitespace normalization
-   `&` ↔ `and`
-   numeric-token extraction
-   address component/token extraction

### Important

Do NOT make a hard-coded French dictionary the backbone.

Prefer data-derived normalization:

-   inspect frequent tokens;
-   learn abbreviation substitutions from matched training records where
    justified;
-   derive high-frequency generic/legal-form tokens from the supplied
    data;
-   make any manually curated domain rules explicit and auditable.

Keep raw fields available for later models.

Do not use `libpostal` unless compliance is explicitly confirmed.

------------------------------------------------------------------------

# 9. Phase 5 --- Candidate generation / blocking

This is critical because:

``` text
true match not in candidates
→ final model can never recover it
```

Start with a strong lexical baseline.

### Recommended first blocker

Character n-gram TF-IDF / BM25-style retrieval over normalized/core name
and name+address.

Use top-k retrieval from S2/S3 records against S1.

Then add cheap exact/specialized blocks such as:

-   shared postal/numeric token + name signal
-   exact normalized/core-name keys
-   useful address-number overlap

Country may be a soft signal.

Only turn country into a hard block after the EDA proves GT consistency.

### Measure every blocker

For validation calculate:

``` text
blocking recall ceiling
average candidates per S1
reduction ratio
```

Tune k to preserve high recall without exploding candidate count.

The design target is approximately ≥98--99% candidate recall if
feasible, but do not fake this number or sacrifice correctness just to
hit the target.

------------------------------------------------------------------------

# 10. Phase 6 --- Pairwise features

For each candidate pair, create features such as:

### Name

-   RapidFuzz ratio
-   token-set similarity
-   token-sort similarity
-   partial similarity
-   Jaro-Winkler if available
-   character TF-IDF cosine
-   core-name equality
-   acronym compatibility
-   token overlap
-   length ratio

### Address

-   token Jaccard
-   character/TF-IDF cosine
-   numeric-token overlap
-   numeric-token conflict
-   postal-code equality
-   address length/coverage signals

### Contextual/collective

Where data supports them:

-   candidate rank
-   reciprocal-best indicator
-   score gap to best competitor
-   number of S1 records sharing core name
-   S2/S3 cross-source similarity

Do not add features merely because they are listed here. Test whether
they help.

------------------------------------------------------------------------

# 11. Phase 7 --- First matcher

Use:

``` text
LightGBM
```

as the first serious matcher.

Reasons:

-   tabular similarity features fit the task;
-   CPU-friendly;
-   fast to iterate;
-   interpretable;
-   suitable for probability ranking.

Generate training candidate pairs using the same blocking logic used
during validation.

Avoid label leakage.

Use hard negatives, especially:

``` text
same/similar business name
+
different address/numeric evidence
```

because false merges are heavily penalized.

------------------------------------------------------------------------

# 12. Phase 8 --- Decision layer

Do NOT simply use:

``` text
probability > 0.5
```

for every pair.

The competition predicts a SET of matches for each S1 entity.

Start with a simple validation-tuned rule:

-   first match threshold
-   higher add-on threshold for additional matches
-   allow empty prediction

Then implement the more principled expected-F0.5/top-prefix decision
from the design document if the baseline is stable.

The number of predicted matches must be chosen using validation F0.5,
not intuition.

------------------------------------------------------------------------

# 13. Phase 9 --- Verify global consistency

Only if A1 is confirmed by training data:

``` text
each S2/S3 record has at most one S1 parent
```

then test a global one-parent resolution step:

``` text
S2/S3 record
→ keep highest-probability S1 parent
→ drop competing assignments
```

Measure whether this improves macro F0.5.

Do not enforce it blindly.

------------------------------------------------------------------------

# 14. Phase 10 --- Produce the first submission

Generate:

``` text
output/candidate_pairs.tsv
output/matching_results.tsv
```

Immediately run:

``` bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

The pipeline must satisfy:

``` text
matches ⊆ candidates
```

and all formatting constraints.

This is the first milestone.

------------------------------------------------------------------------

# 15. Phase 11 --- Improvement ladder

Only after a valid baseline exists:

## Improvement A --- better lexical blocking

Try:

-   BM25/char n-grams
-   multiple retrieval fields
-   union of several top-k retrievers
-   better data-derived core names

Keep only if validation improves.

## Improvement B --- dense retrieval

If compute permits:

-   multilingual embedding model
-   FAISS/HNSW nearest-neighbor search
-   union dense candidates with lexical candidates

Candidate generation should remain auditable.

Potential permitted models from the design document include:

-   multilingual-e5 family where license is verified
-   Qwen3-Embedding-0.6B, Apache-2.0, if its exact model/license is
    verified

Do not download/use a model until its license and parameter count are
checked.

## Improvement C --- calibration

Use out-of-fold predictions and calibration such as isotonic calibration
if it improves the actual set decision.

## Improvement D --- cross-encoder/reranker

Only after the above works.

Potential direction:

``` text
Qwen3-Reranker-0.6B
```

or another verified MIT/Apache-compatible multilingual model under the
size limit.

Use it as a feature/reranking component rather than running an expensive
model over every possible pair.

## Improvement E --- optional small LLM

Only for the ambiguous band.

Potential direction from the design:

``` text
Qwen3-4B / Qwen3-Reranker-4B
```

with an explicit:

``` text
select candidate(s)
OR
none of these
```

Only keep this component if leave-one-country-out validation
demonstrates improvement.

Do NOT use an LLM on every pair.

------------------------------------------------------------------------

# 16. France/generalization strategy

France is unseen in training.

Therefore the agent must explicitly check:

-   accent handling
-   character n-grams
-   numeric/address signals
-   country handling
-   unseen vocabulary
-   legal-form/generic-token behavior
-   threshold robustness

Use US↔India leave-one-country-out experiments as a proxy for domain
shift.

Do not claim that this proves France performance.

It is only an internal robustness signal.

------------------------------------------------------------------------

# 17. GPU usage policy

GPU is OPTIONAL.

Do not block the project on GPU.

Use CPU for:

-   EDA
-   normalization
-   TF-IDF/BM25
-   RapidFuzz
-   LightGBM
-   scoring
-   validation
-   output generation

Use GPU only when it materially helps:

-   dense embedding generation
-   cross-encoder/reranker
-   optional small LLM

Do not create a large cloud bill.

------------------------------------------------------------------------

# 18. Experiment policy

Every improvement must answer:

``` text
Did macro F0.5 improve?
Did singleton accuracy change?
Did non-singleton recall change?
Did blocking recall change?
Did candidate count explode?
Did France/generalization proxy improve?
```

If a more complex component does not improve the relevant validation
metrics:

**remove it.**

Do not keep complexity for appearance.

------------------------------------------------------------------------

# 19. Compliance audit before submission

Before finalizing, automatically check:

### Data

-   no external business data
-   no API lookup
-   no geocoding
-   no registry lookup
-   no web identity resolution

### Models

-   license recorded
-   parameter count recorded
-   ≤8B
-   no Llama-family model
-   no CC-BY-NC model
-   ambiguous model licenses flagged

### Reproducibility

-   pinned dependencies
-   fixed/random seeds recorded
-   code runs from clean environment
-   model download instructions documented
-   no hidden manual step required

### Output

-   validator PASS
-   every test S1 appears once
-   candidate set is final inference set
-   matching IDs ⊆ candidate IDs
-   no invalid IDs
-   no duplicates
-   empty predictions allowed

------------------------------------------------------------------------

# 20. Required agent deliverables

At minimum:

``` text
code/business_entity_resolution/
├── src/
│   ├── data.py
│   ├── normalize.py
│   ├── blocking.py
│   ├── features.py
│   ├── model.py
│   ├── scoring.py
│   ├── decision.py
│   └── pipeline.py
├── README.md
└── requirements.txt

output/
├── matching_results.tsv
└── candidate_pairs.tsv

reports/
├── eda.md
├── experiments.md
└── validation.md
```

The exact module structure can change if the agent has a cleaner design,
but functionality must remain separated and reproducible.

------------------------------------------------------------------------

# 21. What the agent must NOT do

Do not:

-   fabricate dataset statistics before running EDA;
-   claim a model improved without measuring it;
-   claim France performance from assumption;
-   use external business information;
-   use web search to resolve entities;
-   silently install/use questionable licensed models;
-   use a GPU simply because one is available;
-   create thousands of files unnecessarily;
-   rewrite the challenge dataset;
-   overwrite supplied files;
-   submit without running the validator;
-   tune exclusively against the public leaderboard;
-   build an LLM-heavy solution before a valid baseline exists.

------------------------------------------------------------------------

# 22. FIRST EXECUTION TASK

Before implementing advanced ML, perform ONLY this sequence:

``` text
1. Inspect all supplied files.
2. Inspect validator.
3. Run EDA.
4. Resolve A1/A2/A3 where data can answer them.
5. Implement and unit-test F0.5 scorer.
6. Build leakage-safe grouped validation.
7. Implement normalization.
8. Implement lexical candidate generation.
9. Measure blocking recall/reduction.
10. Build similarity features.
11. Train LightGBM baseline.
12. Tune decision layer on macro F0.5.
13. Generate candidate_pairs.tsv.
14. Generate matching_results.tsv.
15. Run official validator.
16. Record baseline metrics.
```

**STOP and report the baseline before proceeding to dense retrieval,
rerankers, LLMs, or other advanced components.**

------------------------------------------------------------------------

# 23. Agent communication format

At the end of each phase report:

``` text
PHASE:
What was implemented:
Files changed:
Libraries added:
Validation method:
Macro F0.5:
Blocking recall:
Avg candidates/S1:
Reduction ratio:
Singleton accuracy:
Non-singleton recall:
Compliance concerns:
Next proposed experiment:
```

Never hide failures. If an assumption is unresolved, explicitly mark:

``` text
UNVERIFIED
```

------------------------------------------------------------------------

# 24. Master instruction to the coding agent

You are the lead ML engineer for this competition.

Read `PS.md`, this `PATH.md`, the supplied challenge files, and the
existing ER design document before coding.

Treat the actual challenge files as authoritative evidence.

Build the system incrementally.

Your priority order is:

``` text
CORRECTNESS
→ VALIDATION
→ WORKING SUBMISSION
→ SCORE IMPROVEMENT
→ COMPLEXITY
```

Do not hallucinate dataset facts.

Do not invent competition rules.

Do not use external business/entity data.

Do not add an advanced model until the current stage has a measured
reason for it.

If a design decision is uncertain, inspect the data first and report the
uncertainty.

Start with Phase 0 only. Do not implement the advanced stages until the
baseline is measured and reported.
