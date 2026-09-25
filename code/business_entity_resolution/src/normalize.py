"""Text normalization for the Amazon ML Challenge.

Design principles
-----------------
1. Raw text is NEVER destroyed: ``raw_name`` / ``raw_addr`` are always kept
   (see ``data.build_entity_table``).
2. We emit several *views* of the same text, because no single representation
   survives all the documented noise:

   - ``norm_name`` / ``norm_addr``  -> Unicode NFKC + casefold + punctuation /
     whitespace collapse. Accent (diacritic) folding is applied ONLY for
     Latin-script text; Indic matras are preserved because they are
     semantically load-bearing (removing them would destroy information).
   - ``latn_name`` / ``latn_addr``  -> a *latin-fold* view used ONLY as
     lexical blocking keys / retrieval corpus. For Devanagari we apply an
     explicit auditable manual transliteration table; other Indic scripts are
     left untouched in the baseline (documented limitation).
   - ``core_key``                   -> sorted content tokens after dropping
     legal-form / generic tokens. Handles word-order changes + legal-suffix
     variation for exact blocking keys.
   - ``num_tok`` / ``postal_tok``   -> digit runs and isolated 5/6-digit runs
     (PIN/ZIP/postcode lookalikes).

3. NO external data, APIs, or libraries are used. The Devanagari table below
   is a manually curated, auditable rule (flagged in the methodology doc). It
   is applied to *blocking keys only* and never replaces the Unicode view.
"""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------

# Ranges for the Indic + related scripts observed in the dataset
# (Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada,
#  Malayalam, + Sinhala as a safety margin).
_INDIC_RANGES = (
    (0x0900, 0x097F), (0x0980, 0x09FF), (0x0A00, 0x0A7F), (0x0A80, 0x0AFF),
    (0x0B00, 0x0B7F), (0x0B80, 0x0BFF), (0x0C00, 0x0C7F), (0x0C80, 0x0CFF),
    (0x0D00, 0x0D7F), (0x0D80, 0x0DFF), (0x1B00, 0x1B7F),
)


def contains_indic(text: str) -> bool:
    """True when *any* character belongs to an Indic/Sinhala script block."""
    for ch in text:
        o = ord(ch)
        for lo, hi in _INDIC_RANGES:
            if lo <= o <= hi:
                return True
    return False


def _is_latin_combining_mark(text: str) -> bool:
    """True when the text contains combining marks around Latin letters."""
    # We only strip combining marks when NO Indic script is present.
    if contains_indic(text):
        return False
    return any(unicodedata.combining(ch) for ch in unicodedata.normalize("NFD", text))


# ---------------------------------------------------------------------------
# Devanagari -> Latin transliteration (manual, auditable rule)
# ---------------------------------------------------------------------------

# Standard phoneme mapping. Only applied to blocking keys, never to the
# Unicode-preserving normalized view.
_DEVA = {
    # independent vowels
    '\u0905': 'a', '\u0906': 'aa', '\u0907': 'i', '\u0908': 'ee',
    '\u0909': 'u', '\u090A': 'oo', '\u090B': 'ri', '\u090E': 'e',
    '\u090F': 'e', '\u0910': 'ai', '\u0913': 'o', '\u0914': 'au',
    # vowel signs (matras)
    '\u093E': 'aa', '\u093F': 'i', '\u0940': 'ee', '\u0941': 'u',
    '\u0942': 'oo', '\u0943': 'ri', '\u0947': 'e', '\u0948': 'ai',
    '\u094B': 'o', '\u094C': 'au', '\u093D': 'a', '\u0902': 'n',
    '\u0903': 'h', '\u0901': 'n', '\u0900': 'a',
    # virama (halant) - dropped; the following consonant merges naturally
    '\u094D': '',
    # nukta-marked forms
    '\u095C': 'q', '\u095D': 'kh', '\u095E': 'g', '\u095F': 'z',
    '\u090C': 'ri', '\u0944': 'oo',
    # consonants
    '\u0915': 'k',  '\u0916': 'kh', '\u0917': 'g',  '\u0918': 'gh',
    '\u0919': 'n',  '\u091A': 'ch', '\u091B': 'chh', '\u091C': 'j',
    '\u091D': 'jh', '\u091E': 'ny', '\u091F': 't',  '\u0920': 'th',
    '\u0921': 'd',  '\u0922': 'dh', '\u0923': 'n',  '\u0924': 't',
    '\u0925': 'th', '\u0926': 'd',  '\u0927': 'dh', '\u0928': 'n',
    '\u092A': 'p',  '\u092B': 'ph', '\u092C': 'b',  '\u092D': 'bh',
    '\u092E': 'm',  '\u092F': 'y',  '\u0930': 'r',  '\u0932': 'l',
    '\u0935': 'v',  '\u0936': 'sh', '\u0937': 'sh', '\u0938': 's',
    '\u0939': 'h',
    # digits
    '\u0966': '0', '\u0967': '1', '\u0968': '2', '\u0969': '3',
    '\u096A': '4', '\u096B': '5', '\u096C': '6', '\u096D': '7',
    '\u096E': '8', '\u096F': '9',
}


def _deva_to_latin(text: str) -> str:
    return "".join(_DEVA.get(ch, ch) for ch in text)


# ---------------------------------------------------------------------------
# Core transforms
# ---------------------------------------------------------------------------

# Legal forms / generic tokens dropped from core names (lower-cased, no punct)
BOUNDARY = (set(" \t-/.()[]{}<>,;:|#+*'\"") | set())


def _strip_punct_keep_alnum(s: str) -> str:
    """Replace non-alphanumeric, non-space chars with a space (preserves Indic)."""
    return "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)


_WS_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d+")
_POSTAL_RE = re.compile(r"(?<!\d)(\d{5,6})(?!\d)")


def _collapse(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


# Legal / generic tokens removed when building core keys. Both English and
# French forms are listed; the set is conservative so real content tokens are
# never dropped.
LEGAL_TOKENS = frozenset(
    """
    llc inc incorporated inc corp corporation co company plc limited ltd
    private pvt llp lp pllc pc dba assoc associates corporation
    sarl sas sasu sa eurl sci snc societe association amicale comite
    groupe maison fils freres clinique ecole college ligue sportive
    an and the of to de du des la le les au aux et un une
    m/s ms mrs mr dr st saint
    """.split()
)
# Note: 'st'/'saint' removal helps French names (St === Saint) but could hurt a
# rare real name token; it stays in the list because core keys are only used
# for *exact* blocking (a missed key just costs recall, never precision).


def normalize_text(text) -> str:
    """Unicode-preserving normalized view (predicted main comparison surface).

    - NFKC compatibility decomposition
    - casefold / lower
    - ``&`` -> ``and``
    - Latin-script diacritics folded (NFD + drop combining marks)
    - Indic text left intact (vowel signs are meaning-bearing)
    - punctuation -> space; whitespace collapsed
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    s = unicodedata.normalize("NFKC", text).lower()
    s = s.replace("&", " and ")
    if contains_indic(s):
        pass  # keep matras
    else:
        s = unicodedata.normalize("NFD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _strip_punct_keep_alnum(s)
    return _collapse(s)


def latin_fold(text) -> str:
    """Latin-fold view for retrieval/blocking keys.

    Devanagari is transliterated via the explicit auditable table; everything
    is then casefolded, diacritic-stripped and punctuation-collapsed. Used
    ONLY for search indexing, never as the primary comparison text.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if contains_indic(text):
        text = _deva_to_latin(text)
    s = unicodedata.normalize("NFKC", text).lower()
    s = s.replace("&", " and ")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _strip_punct_keep_alnum(s)
    return _collapse(s)


def core_name_tokens(norm_text: str):
    """Content tokens (ordered) after dropping legal/generic tokens."""
    toks = norm_text.split()
    return [t for t in toks if t not in LEGAL_TOKENS]


def core_key(norm_name: str) -> str:
    """Sorted content tokens -> canonical blocking key (handles word order)."""
    return " ".join(sorted(set(core_name_tokens(norm_name))))


def tokens(norm_text: str):
    """Lower-cased word tokens."""
    return norm_text.split()


def unique_tokens(norm_text: str):
    return sorted(set(norm_text.split()))


def numeric_token_string(text) -> str:
    """Space-joined sorted unique digit runs (house no, plot no, PIN...)."""
    if text is None:
        return ""
    return " ".join(sorted(set(_DIGIT_RE.findall(str(text)))))


def postal_token_string(text) -> str:
    """Space-joined isolated 5/6-digit runs (PIN/ZIP/postcode lookalikes)."""
    if text is None:
        return ""
    return " ".join(sorted(set(_POSTAL_RE.findall(str(text)))))


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------

_NORM_FUNCS = {
    "norm": normalize_text,
    "latn": latin_fold,
}


def build_named_normalizations(text_series):
    """Return dict {norm: Series[str], latn: Series[str]} for one column."""
    return {k: text_series.map(f) for k, f in _NORM_FUNCS.items()}


def sanity_examples():
    """A few before/after examples used by the notebook as a sanity cell."""
    samples = [
        "Rexford LLC",
        "Foot & Ankle Care Associates",
        "Custom Wealth Services LLC",
        "ग्लोबल इन्वेस्टमेंट प्रा. लि.",
        "Maison de Santé Génération",
        "SAINT-HERBLAIN Societe SARL",
        "OZT ÂMICALE SAS",
        "5 bis Rue Pierre Dignac",
        "H.No.16-11-23/37/A, 2Nd Floor",
    ]
    out = []
    for s in samples:
        out.append((s, normalize_text(s), latin_fold(s), core_key(normalize_text(s))))
    return out