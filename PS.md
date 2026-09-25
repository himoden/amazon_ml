# Amazon ML Challenge 2026 --- Problem Specification (PS)

## 1. Purpose

This document is the **source-of-truth problem specification** for the
agent working on the Amazon ML Challenge 2026 Business Entity Resolution
task.

**Do not invent requirements. Do not relax constraints. Do not use
external business data.**

The task is reference-anchored entity resolution:

**Source 1 (clean/deduplicated reference) → matching records in Source 2
∪ Source 3**

A Source 1 entity can have **zero, one, or many** matching records.

Source-derived facts below come from the official challenge/problem
statement supplied by the user and the accompanying design document.

------------------------------------------------------------------------

## 2. Dataset

Each record contains:

-   `entity_id`
-   `business_name`
-   `business_address`
-   `country`

The three sources are identified by their `entity_id` prefixes and
files:

-   `S1-...` → Source 1
-   `S2-...` → Source 2
-   `S3-...` → Source 3

### Training

``` text
dataset/train/
├── train_source1.tsv
├── train_source2.tsv
├── train_source3.tsv
└── train_ground_truth.tsv
```

### Test

``` text
dataset/test/
├── test_source1.tsv
├── test_source2.tsv
└── test_source3.tsv
```

All files are TSV. They must be read with an explicit tab separator.

``` python
pd.read_csv(path, sep="\t")
```

The ground truth contains:

-   `source1_entity_id`
-   `matched_entity_ids`

`matched_entity_ids` is a comma-separated list of matching S2/S3 IDs and
may be empty.

------------------------------------------------------------------------

## 3. Noise to expect

The problem statement explicitly identifies:

### Business-name noise

-   abbreviations
-   legal suffix variation
-   DBA/trade names
-   punctuation differences
-   `&` vs `and`
-   word-order changes
-   typos
-   transliteration differences

Examples include:

``` text
Corp ↔ Corporation
Pvt Ltd ↔ Private Limited
```

### Address noise

-   abbreviations
-   transliteration variants
-   missing components
-   missing PIN/state
-   landmark-based references
-   municipal numbering differences
-   component reordering

Example:

``` text
Rd ↔ Road
St ↔ Street
Near SBI ATM
```

------------------------------------------------------------------------

## 4. Critical test-set shift

Training data covers:

``` text
US
India
```

The test set additionally contains:

``` text
France
```

France is unseen during training.

Therefore:

-   Do NOT hard-code `{US, India}`.
-   Do NOT filter the pipeline to those two countries.
-   Do NOT one-hot encode country in a way that cannot handle unseen
    labels.
-   Country-specific processing must not be the only normalization
    strategy.
-   Generalization to unseen country must be explicitly evaluated.

------------------------------------------------------------------------

## 5. Fair-play / prohibited external information

The solution must use the supplied challenge data and permitted
pretrained model weights only.

STRICTLY PROHIBITED:

-   commercial entity-resolution APIs
-   government/business registries
-   geocoding APIs
-   external business databases
-   internet business lookups
-   external identity resolution
-   external data augmentation

Do not search the web to identify businesses.

Do not use external business knowledge to resolve a pair.

The pipeline must work offline after any permitted one-time model-weight
download.

------------------------------------------------------------------------

## 6. Model constraints

The final model must be:

-   MIT or Apache-2.0 licensed
-   at most 8 billion parameters

Explicitly avoid:

-   Llama-family models because of their license
-   CC-BY-NC models
-   models whose actual total parameter count exceeds the limit

The design document flags Qwen3-8B as risky because its total parameter
count is 8.2B. Default to the ≤4B tier unless organizers explicitly
clarify otherwise.

Do not use `libpostal` unless organizers explicitly clear it, because
its trained resources originate from external address datasets and may
create a compliance concern.

------------------------------------------------------------------------

## 7. Evaluation metric

The leaderboard evaluates:

**F0.5**

``` text
F0.5 = (1.25 × Precision × Recall)
       / (0.25 × Precision + Recall)
```

It is calculated:

1.  separately for each Source 1 entity;
2.  then macro-averaged over Source 1 entities.

Precision is weighted more heavily than recall.

### Singleton rule

If a Source 1 entity has no true matches:

``` text
prediction = empty
score = 1.0
```

If a true singleton receives any false match:

``` text
score = 0.0
```

Therefore false merges are particularly dangerous.

### Important decision unit

The model ultimately predicts a **set of S2/S3 records for each S1
entity**.

Do not optimize only pairwise F1.

Threshold/set-selection logic must be validated against the actual
per-S1 macro F0.5 metric.

------------------------------------------------------------------------

## 8. Required outputs

The solution must create:

``` text
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

### matching_results.tsv

Columns:

``` text
source1_entity_id
matched_entity_ids
```

Requirements:

-   exactly one row for every test Source 1 entity
-   empty `matched_entity_ids` when there is no match
-   only S2/S3 IDs
-   IDs must exist in the test data
-   no duplicate IDs within an ID list
-   no duplicate Source 1 rows
-   no S1 self-matches

### candidate_pairs.tsv

Columns:

``` text
source1_entity_id
candidate_entity_ids
```

This must contain the **final candidate set actually passed to the final
matching model**.

It is NOT merely an early blocking result if later filtering is
performed.

Every final predicted match must be contained in this candidate set:

``` text
matching_results ⊆ candidate_pairs
```

The candidate set is used to evaluate blocking recall ceiling and
reduction ratio.

------------------------------------------------------------------------

## 9. Local validation

The supplied validator is:

``` text
utils/validate_submission.py
```

Run:

``` bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

A valid submission must pass this validator before submission.

------------------------------------------------------------------------

## 10. Final submission package

Expected structure:

``` text
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```

The methodology must document:

-   methodology
-   candidate generation/blocking
-   model architecture
-   feature engineering
-   other relevant implementation details

------------------------------------------------------------------------

## 11. Known assumptions that MUST be checked before enforcing

The design document identifies these as assumptions/questions, not
facts:

### A1 --- S2/S3 one-parent property

Check whether an S2/S3 record appears in more than one ground-truth
Source 1 row.

If confirmed, a global one-parent constraint may be enforced.

If not confirmed, do not blindly enforce it.

### A2 --- Country consistency

Check whether every ground-truth pair has the same country label.

Only use country as a hard block if the training data demonstrates this
property.

Otherwise country should remain a soft feature/signal.

### A3 --- Empty prediction for a non-singleton

The supplied design document flags the treatment of an empty prediction
for a non-singleton as something to confirm with organizers because it
affects exact expected-F0.5 calculations.

Do not silently assume an organizer rule that has not been verified.

------------------------------------------------------------------------

## 12. What is NOT part of the task

Do not:

-   perform symmetric deduplication of all three sources
-   cluster S2/S3 first and blindly take transitive closure
-   query external databases
-   geocode addresses
-   add internet business information
-   optimize only pairwise F1
-   hard-code US/India as the only countries
-   create candidate_pairs.tsv from an intermediate blocker and then
    score additional pairs
-   add a final model that violates the license/parameter constraints

------------------------------------------------------------------------

## 13. Source-of-truth hierarchy

When information conflicts:

1.  **Official challenge/problem statement and supplied challenge
    files**
2.  **Organizer clarifications**
3.  **ER design document assumptions/recommendations**
4.  **Agent/model suggestions**

The agent must never turn an assumption or suggestion into a confirmed
competition rule without evidence.
