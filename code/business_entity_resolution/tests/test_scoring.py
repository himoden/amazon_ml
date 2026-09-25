"""Pure-python unit tests for the scorer + normalization.

These run WITHOUT the dataset (no heavy I/O). Used locally and in CI.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scoring import f05_score, per_entity_score, metrics_report, validate_outputs  # noqa: E402
from normalize import normalize_text, latin_fold, core_key, core_name_tokens  # noqa: E402


def test_f05_worked_example_from_readme():
    # README: pred [A,B,C], truth [A,B] -> P=2/3, R=1, F0.5=0.714
    assert abs(f05_score(2 / 3, 1.0) - 0.714) < 0.001


def test_singleton_rules():
    assert per_entity_score([], []) == 1.0
    assert per_entity_score([], ["S2-1"]) == 0.0


def test_empty_non_singleton_provisional():
    # A3 UNVERIFIED: provisional convention = 0.0, configurable
    assert per_entity_score(["S2-1"], []) == 0.0
    assert per_entity_score(["S2-1"], [], empty_non_singleton_score=0.5) == 0.5


def test_partial_match():
    # true {a,b}, pred {a,c} -> P=1/2, R=1/2
    s = per_entity_score({"S2-a", "S2-b"}, {"S2-a", "S2-c"})
    assert abs(s - f05_score(0.5, 0.5)) < 1e-9


def test_metrics_report():
    true = {"A": {"S2-1"}, "B": set(), "C": {"S2-2", "S2-3"}}
    pred = {"A": {"S2-1"}, "B": set(), "C": {"S2-2"}}
    r = metrics_report(true, pred, empty_non_singleton_score=0.0)
    # A: 1.0, B: 1.0, C: P=1, R=0.5 -> f05(1, .5)
    expected_c = f05_score(1.0, 0.5)
    assert abs(r["macro_f05"] - (1.0 + 1.0 + expected_c) / 3) < 1e-9
    assert r["n_singletons"] == 1
    assert r["singleton_accuracy"] == 1.0


def test_normalize_basics():
    assert normalize_text("Foot & Ankle Care Associates LLC") == \
        normalize_text("foot and ankle care associates llc")
    assert normalize_text("  MÁISON de SANTÉ ") == "maison de sante"


def test_devanagari_preserved_in_unicode_view():
    s = "ग्लोबल इन्वेस्टमेंट प्रा. लि."
    n = normalize_text(s)
    assert any("\u0900" <= ch <= "\u097F" for ch in n), "devanagari matras destroyed"


def test_latin_fold_transliteration():
    # naive Devanagari->Latin char map (auditable rule); Hindi orthography
    # drops the inherent vowel so expect a close Latin token
    n = latin_fold("ग्लोबल इन्वेस्टमेंट प्रा. लि.")
    assert not any("\u0900" <= ch <= "\u097F" for ch in n)
    tokens = n.split()
    assert any("globl" in t or "global" in t for t in tokens)


def test_core_key_word_order_robust():
    a = core_key(normalize_text("Ambernath Solar Private Limited"))
    b = core_key(normalize_text("Private Ambernath Solar Limited"))
    assert a == b


def test_validate_outputs():
    required = {"S1-1", "S1-2"}
    preds = {"S1-1": {"S2-1"}, "S1-2": set()}
    cands = {"S1-1": {"S2-1", "S3-1"}, "S1-2": set()}
    valid_ids = {"S2-1", "S3-1"}
    errs = validate_outputs(preds, cands, required, valid_ids)
    assert errs == []
    errs = validate_outputs({"S1-9": {"S2-1"}}, None, required, valid_ids)
    assert len(errs) >= 1


def run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failures}/{len(fns)} passed")
    return failures


if __name__ == "__main__":
    sys.exit(1 if run_all() else 0)