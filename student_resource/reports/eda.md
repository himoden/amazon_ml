# EDA Report — Amazon ML Challenge 2026 (Business Entity Resolution)

Run date: 2026-09-25
Stage: Phase 0 (inspect + EDA only, no modeling code).

## 1. Files inspected

| File | Size (MB) | Source of truth |
|---|---|---|
| `dataset/train/train_source1.tsv` | 210 | S1 reference (clean) |
| `dataset/train/train_source2.tsv` | 489 | S2 noisy |
| `dataset/train/train_source3.tsv` | 504 | S3 noisy |
| `dataset/train/train_ground_truth.tsv` | 127 | labels |
| `dataset/test/test_source1.tsv` | 175 | S1 to predict for |
| `dataset/test/test_source2.tsv` | 509 | S2 candidate pool |
| `dataset/test/test_source3.tsv` | 506 | S3 candidate pool |
| `utils/validate_submission.py` | — | official validator (read fully) |

Columns in every source file: `entity_id`, `business_name`, `business_address`, `country` (TSV).
GT columns: `source1_entity_id`, `matched_entity_ids` (comma-separated, may be empty/absent = NaN).

## 2. Row counts and country mix

| File | Rows | US | India | France |
|---|---|---|---|---|
| train S1 | 2,206,821 | 1,323,633 | 883,188 | 0 |
| train S2 | 5,034,616 | 3,016,817 | 2,017,799 | 0 |
| train S3 | 5,285,603 | 3,170,056 | 2,115,547 | 0 |
| test S1  | 1,732,544 | 663,106 | 809,986 | 259,452 |
| test S2  | 4,887,273 | 1,871,330 | 2,312,565 | 703,378 |
| test S3  | 5,082,316 | 1,945,701 | 2,405,000 | 731,615 |

- Train S1 : test S1 ratio ≈ 1.27.
- France appears ONLY in test (15.0% of test S1; 14.4% of test S2, 14.4% of test S3). Confirms the documented train/test shift.
- Country share test S1: India 46.7%, US 38.3%, France 15.0%.

## 3. Missing data

- S1: no nulls in any column. S1 is clean by construction.
- S2/S3: `business_name` null ≈ 0.00%; `business_address` null ≈ **3.3–3.6%** (~169k/176k rows per source). Must be handled at feature time (both train and test).
- GT `matched_entity_ids` is null/empty for 5.58% of S1 rows (= singletons).

## 4. Ground-truth structure

- 2,206,821 GT rows, all unique S1 (no duplicate S1 rows).
- Total S1↔S2/S3 positive pairs: **7,638,365** (all unique).
- `n_matches` per S1: mean **3.46**, median 3, 25th pct 2, 75th pct 5, max 11. Value counts: 3→530k, 4→484k, 2→375k, 5→322k, 6→165k, 0→123k, 1→119k, 7→64k, 8→18.7k, 9→4.2k...
- **Singleton rate: 5.58%** overall; India 5.59%, US 5.58%. Near-identical by country.
- Match count per S1 is statistically identical across US/India (mean ≈3.46).
- Matched children: S3 3,944,746 vs S2 3,693,619 (S3 slightly over-represented).

## 5. Assumption checks

### A1 — S2/S3 one-parent property: **CONFIRMED** (full data)
- Scanned all 7,638,365 GT pairs; every one of the 7,638,365 distinct S2/S3 children maps to exactly **one** S1 parent.
- Zero violations. A global one-parent resolution step is legitimate to test.

### A2 — Country consistency of GT pairs: **CONFIRMED** (693,069 sampled pairs, 0 mismatches)
- Every matched S2/S3 has a country label resolvable in train S2/S3 (7,638,365/7,638,365).
- Country is 100% consistent within GT pairs. Country **can** be a hard block on train-distributed countries; France recall cannot be checked from train.

### A3 — Empty-prediction score for non-singletons: **UNVERIFIED**
- README documents the singleton rule (empty→1.0, any false match→0.0) and the worked F0.5 example, but does NOT state how the official scorer treats an empty prediction on a non-singleton (P=0/0? R=0? F₀.₅=0?).
- We will implement a documented scorer with a zero-fill convention and mark this UNVERIFIED pending organizer clarification. Reference to ED design doc required.

## 6. Entity ID sanity

- Prefixes match file source: S1-/S2-/S3- only. No prefix leaks in GT.
- No duplicate `entity_id` within any source file.
- Every GT child exists in train S2 ∪ S3 (0 missing).
- **Zero ID overlap between train and test** (disjoint).

## 7. Duplicate patterns

- S1: no exact duplicate (name,address); 667,592 duplicate `business_name` (many chains share legal names), 76,215 duplicate addresses → S1 has many same-name-but-different-address and same-address-but-different-name cases. Strong false-merge risk.
- S2: 25,891 exact (name,address) duplicates; S3: 18,881 (records duplicated *within* a source — may represent the same real business twice; not our target but relevant to candidate pool).

## 8. Character / script evidence (noise landscape)

Proportions per country+source:

| Source | Country | name accent | addr accent | name Devanagari | addr Devanagari | name digits | addr digits |
|---|---|---|---|---|---|---|---|
| S1 | US | 0.00 | 0.00 | 0.00 | 0.00 | 0.026 | 1.000 |
| S1 | India | 0.00 | 0.0006 | 0.00 | 0.00 | 0.001 | 0.913 |
| S2/S3 | US | ~0.068 | 0.00 | 0 | 0 | ~0.064 | ~0.90 |
| S2/S3 | India | 0.18–0.27 | 0.17–0.18 | 0.07–0.13 | 0.13–0.14 | ~0.03 | ~0.90 |
| T1 | France | 0.157 (name non-ascii) | — | 0 | 0 | 0.008 | 0.996 |
| T2 | France | — | 0.248 accented addr | 0 | 0 | — | — |

Reading:
- **US**: ASCII everywhere; legal forms `LLC` (355k), `INC` (238k), `&`/`and`, `CORP`, `PC`, `PLLC`, `LP`. Addresses always digit-bearing.
- **India S1 is clean Latin** (no Devanagari, no accent) while **S2/S3 India contain heavy Devanagari and accented transliteration** (e.g. `ग्लोबल इन्वेस्टमेंट प्रा. लि.`, `महाराष्ट्र`; `Sólar Limited`, `Pàrenthese`). Matches documented transliteration noise; also **Devanagari is present only in India records, so an India-tuned normalization won't harm US/France but India needs transliteration handling to bridge S1↔S2/S3.**
- India S1 legal forms: `LIMITED`, `PRIVATE`, `LTD`, `PVT`, `LLP`, `CO`. `&` is common (37.5k).
- **France**: French legal forms `SARL`, `SASU`, `EURL`, `SCI`, `Association`. Accents are real French (`Président`, `ALLÉE`) AND corruption-style (`Àmicale`, `ÂMICALE`, `Sólar`). Wildcard `<<` in names (`<< Team Ecole`). Addresses ~100% contain digits; 66% contain `RUE`; French component order varies (street first or postcode first), with `bis`/`ter` sub-numbering, `CITÉ`, `BOULEVARD`, `IMPASSE`, `CHEMIN`. 5-digit postcode-like tokens exist but not at fixed position → address-parse must be order-robust.

## 9. Implications for pipeline design (Phase 0, no code yet)

1. **Normalization must be Unicode-aware and data-derived** (NFKD/NFKC, casefold, accent folding for Latin, keep raw). France requires accent handling without assuming `{US,India}`.
2. **Blocking**: S1↔S23 pool is huge (test: 1.73M S1 × ~9.97M S23). Need lexical (char n-gram TF-IDF/BM25) top-k + exact keys (postal/numeric token + name keys). Country can be a hard block on train countries (A2 confirmed) but France must not be dropped; France numeric/street tokens must participate.
3. **Decision layer**: per-S1 set selection under F₀.₅ (precision > recall). Singleton rate 5.6% — small but worth correct handling. Strong false-merge penalty → one-parent resolution (A1 confirmed) is a promising global constraint.
4. **Validation**: test France has no labels; US↔India leave-one-country-out is the only in-train generalization proxy. France results are inherently estimated, documented as UNVERIFIED-until-submitted.
5. **GT is clean and complete**: mean 3.46 matches/S1 gives a sanity bound for predicted set sizes.

## 10. Design-reference gaps (flagged, not assumed)

- The ER design doc PDF could not be parsed by the model (binary attachment unsupported); PS.md/PATH.md sections referencing it were used as the source of truth. Flag: should be re-read by the team for A3 and model-license specifics.
- `validate_submission.py`: `--check-ids` loads all test S2/S3 IDs (~9.97M) → several GB RAM; run it once at submission time, not in every iteration.
- No external data / no geocoding / no libpostal — normalized forms must come from the supplied data only.