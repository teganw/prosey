"""Prosey corpus pipeline — rebuilds words_data.json from books/.

Implements CLAUDECODE_BUILD.md. Run from the repo root:

    python build_pipeline.py

Populate books/ and books/metadata.json first (see CLAUDECODE_BUILD.md Step 0).
The pipeline stops and reports if books/ is empty.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from wordfreq import zipf_frequency

BOOKS_DIR = Path("books")

# Example-sentence bounds (words). Short fragments ("He extricated.") carry no
# prose value; very long ones are usually OCR-mangled run-ons.
SENT_MIN_WORDS = 8
SENT_MAX_WORDS = 55

# Cap examples per word: keep the most diverse few, not hundreds. The app shows
# 3 and reveals the rest up to this cap; it also keeps words_data.json lean.
MAX_EXAMPLES_PER_WORD = 8


# --- Step 0 helpers: Gutenberg boilerplate --------------------------------

def strip_gutenberg_boilerplate(text):
    lines = text.split("\n")
    start = next((i for i, l in enumerate(lines) if "*** START OF" in l), 0) + 1
    end = next((i for i, l in enumerate(lines) if "*** END OF" in l), len(lines))
    return "\n".join(lines[start:end])


# --- Step 2: Load corpus ---------------------------------------------------

def load_books():
    metadata = json.loads((BOOKS_DIR / "metadata.json").read_text())
    books = []
    for txt_file in sorted(BOOKS_DIR.glob("*.txt")):
        stem = txt_file.stem
        if stem not in metadata:
            raise ValueError(f"No metadata for {stem} — add to books/metadata.json first")
        meta = metadata[stem]
        text = strip_gutenberg_boilerplate(
            txt_file.read_text(encoding="utf-8", errors="replace")
        )
        sentences = nltk.sent_tokenize(text)
        books.append({"meta": meta, "sentences": sentences, "stem": stem})
        print(f"  Loaded {stem}: {len(sentences)} sentences")
    return books


# --- Step 3: Lemmatization -------------------------------------------------

LEMMATIZER = WordNetLemmatizer()
_LEMMA_CACHE = {}
_FREQ_CACHE = {}


def derivational_roots(word):
    """Generate plausible base forms by stripping derivational suffixes."""
    cands = []
    if word.endswith("bly"):    cands.append(word[:-3] + "ble")   # flexibly → flexible
    if word.endswith("ily"):    cands.append(word[:-3] + "y")     # happily → happy
    if word.endswith("ly"):
        cands.append(word[:-2])                                    # sternly → stern
        cands.append(word[:-2] + "e")
    if word.endswith("iness"): cands.append(word[:-5] + "y")      # happiness → happy
    if word.endswith("ness"):  cands.append(word[:-4])            # remoteness → remote
    if word.endswith("iest"):  cands.append(word[:-4] + "y")      # easiest → easy
    if word.endswith("est"):
        cands.append(word[:-3])                                    # soonest → soon
        cands.append(word[:-3] + "e")
    if word.endswith("ier"):   cands.append(word[:-3] + "y")
    if word.endswith("er"):
        cands.append(word[:-2])
        cands.append(word[:-2] + "e")
    return cands


def lemmatize(word):
    cached = _LEMMA_CACHE.get(word)
    if cached is not None:
        return cached
    result = _lemmatize_uncached(word)
    _LEMMA_CACHE[word] = result
    return result


def _lemmatize_uncached(word):
    # 1. Verb inflection
    v = LEMMATIZER.lemmatize(word, pos="v")
    if v != word and wn.synsets(v):
        return v
    # 2. Noun plural
    n = LEMMATIZER.lemmatize(word, pos="n")
    if n != word and wn.synsets(n):
        return n
    # 3. Derivational suffix stripping
    for root in derivational_roots(word):
        if len(root) >= 3 and root != word and wn.synsets(root):
            return root
    return word


def freq_of(lemma):
    cached = _FREQ_CACHE.get(lemma)
    if cached is not None:
        return cached
    f = zipf_frequency(lemma, "en")
    _FREQ_CACHE[lemma] = f
    return f


# --- Step 4: Filters and blocklists ---------------------------------------

BLOCKLIST = {
    # Number words (spelled out)
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety", "hundred", "thousand", "million", "billion",
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    "ninth", "tenth", "eleventh", "twelfth",
    # Roman numerals (common OCR artifacts; most are also caught by is_number_word)
    "iii", "vii", "viii", "xii", "xiii", "xiv", "xvi", "xvii", "xviii", "xix",
    "xxi", "xxii", "xxiii", "xxiv", "xxv", "xxvi", "xxvii", "xxviii", "xxix", "xxx",
    "xlviii",  # documented escapee — has a WordNet synset ("being eight more than forty")
    # Archaic / legal boilerplate
    "hath", "doth", "shalt", "canst", "wouldst", "shouldst", "whilst", "amongst",
    "thereof", "hereof", "whereof", "wherein", "herein", "whereby", "hereby",
    "aforesaid", "heretofore", "notwithstanding",
    # Project Gutenberg artifacts
    "gutenberg", "ebook", "epub", "utf",
}


def precompute_cap_counts(books):
    """Single pass over the corpus: lowercased-token -> [lower_count, upper_count].

    Replaces the per-occurrence full-corpus rescan in the original spec, which
    made the pipeline O(words * sentences). Here it is O(words).
    """
    counts = defaultdict(lambda: [0, 0])
    for book in books:
        for sent in book["sentences"]:
            for tok in re.findall(r"[a-zA-Z]+", sent):
                entry = counts[tok.lower()]
                if tok[0].isupper():
                    entry[1] += 1
                else:
                    entry[0] += 1
    return counts


def capitalized_ratio(word, cap_counts):
    lower, upper = cap_counts.get(word, (0, 0))
    total = lower + upper
    if total == 0:
        return 0.0
    return upper / total


def is_proper_noun(word):
    """True if WordNet classifies this word as a named instance."""
    syns = wn.synsets(word)
    if not syns:
        return False
    return len(syns[0].instance_hypernyms()) > 0


# Number words (cardinals, ordinals, roman numerals) have WordNet entries, so a
# frequency/synset gate won't catch them. Their primary definition follows a few
# distinctive shapes — "the cardinal number...", "position N in a countable
# series", "being X more than Y" — which a static blocklist can't keep up with.
# This is the LOG's definition-pattern filter, replacing the whack-a-mole list.
_NUMBER_DEF_PATTERNS = [
    re.compile(p, re.I) for p in (
        r"cardinal number",
        r"ordinal number",
        r"in a countable series",
        r"\bbeing\b.*\bmore than\b",
        r"in an ordering or series",
    )
]


def is_number_word(word):
    syns = wn.synsets(word)
    if not syns:
        return False
    definition = syns[0].definition()
    return any(p.search(definition) for p in _NUMBER_DEF_PATTERNS)


def passes_filters(lemma, freq, cap_counts):
    if lemma in BLOCKLIST:
        return False
    if len(lemma) < 3:
        return False
    if not lemma.isalpha():
        return False
    if not (1.8 <= freq <= 4.1):
        return False
    if not wn.synsets(lemma):
        return False
    if is_proper_noun(lemma):
        return False
    if is_number_word(lemma):
        return False
    if capitalized_ratio(lemma, cap_counts) > 0.40:
        return False
    return True


# --- Step 6: Band assignment ----------------------------------------------

def assign_band(freq):
    if freq >= 3.2:
        return "everyday"
    elif freq >= 2.6:
        return "uncommon"
    elif freq >= 2.35:
        return "rare"
    else:
        return "morerare"  # 1.8–2.35 — Tier 2 deep vocabulary


# Bands that survive on a single source. Deep/rare vocabulary often appears in
# only one author, so the >=2-source cross-validation rule would delete exactly
# the words this tool exists to surface (e.g. louche, opprobrium, numinous).
SINGLE_SOURCE_BANDS = {"rare", "morerare"}


# --- Step 5: Candidate extraction -----------------------------------------

def extract_candidates(books, cap_counts):
    """Returns {lemma: [{"text": sent, "meta": meta}]}.

    everyday/uncommon lemmas require >= 2 distinct source files; rare/morerare
    lemmas are kept from a single source.
    """
    occurrences = defaultdict(lambda: defaultdict(list))  # lemma -> stem -> [items]

    for book in books:
        meta = book["meta"]
        for sent in book["sentences"]:
            if not (SENT_MIN_WORDS <= len(sent.split()) <= SENT_MAX_WORDS):
                continue  # skip fragments and OCR-mangled run-ons
            seen_in_sent = set()
            for word in re.findall(r"[a-zA-Z]+", sent):
                lemma = lemmatize(word.lower())
                if lemma in seen_in_sent:
                    continue  # one sentence contributes a lemma at most once
                seen_in_sent.add(lemma)
                freq = freq_of(lemma)
                if passes_filters(lemma, freq, cap_counts):
                    occurrences[lemma][book["stem"]].append({"text": sent, "meta": meta})

    candidates = {}
    for lemma, by_stem in occurrences.items():
        min_sources = 1 if assign_band(freq_of(lemma)) in SINGLE_SOURCE_BANDS else 2
        if len(by_stem) >= min_sources:
            candidates[lemma] = [item for items in by_stem.values() for item in items]

    print(f"Candidates after filters: {len(candidates)}")
    return candidates


# --- Step 7: Definition and example extraction ----------------------------

def get_definition_and_pos(lemma):
    synsets = wn.synsets(lemma)
    if not synsets:
        return "", "", []
    pos_map = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}
    pos = pos_map.get(synsets[0].pos(), "")
    definition = synsets[0].definition()
    alt_definitions = [s.definition() for s in synsets[1:5]]
    return pos, definition, alt_definitions


def century_of(meta):
    return (meta.get("year", 0) // 100) * 100


def rank_examples(occurrences):
    """Rank by diversity across author, genre, century, nationality, gender."""
    seen = {"author": set(), "genre": set(), "century": set(),
            "nationality": set(), "gender": set()}
    scored = []
    for item in occurrences:
        m = item["meta"]
        vals = {"author": m.get("author"), "genre": m.get("genre"),
                "century": century_of(m), "nationality": m.get("nationality"),
                "gender": m.get("gender")}
        score = 0
        for key, pts in [("author", 4), ("genre", 3), ("century", 2),
                         ("nationality", 2), ("gender", 1)]:
            val = vals[key]
            if val and val not in seen[key]:
                score += pts
                seen[key].add(val)
        scored.append((score, item))
    scored.sort(key=lambda x: -x[0])
    return [item for _, item in scored]


def build_example(item):
    m = item["meta"]
    return {
        "text": item["text"].strip(),
        "title": m.get("title", ""),
        "author": m.get("author", ""),
        "year": m.get("year", 0),
        "genre": m.get("genre", ""),
        "nationality": m.get("nationality", ""),
        "century": century_of(m),
        "gender": m.get("gender", "-"),
    }


# --- Step 8: Build word objects -------------------------------------------

def build_words(candidates):
    words = []
    for lemma, occurrences in sorted(candidates.items()):
        freq = freq_of(lemma)
        band = assign_band(freq)
        pos, definition, alt_definitions = get_definition_and_pos(lemma)

        ranked = rank_examples(occurrences)[:MAX_EXAMPLES_PER_WORD]
        examples = [build_example(item) for item in ranked]

        words.append({
            "word": lemma,
            "pos": pos,
            "definition": definition,
            "freq": round(freq, 2),
            "examples": examples,
            "band_auto": band,
            "band": band,
            "alt_definitions": alt_definitions,
        })

    print(f"Built {len(words)} word objects")
    return words


# --- Step 9: Concreteness tagging -----------------------------------------

def load_brysbaert():
    import urllib.request

    url = ("https://raw.githubusercontent.com/desmond-ong/colorMeText"
           "/master/lexicons/Concreteness_ratings_Brysbaert_et_al_BRM_parsed.csv")
    with urllib.request.urlopen(url, timeout=15) as r:
        raw = r.read().decode("utf-8", "replace")

    lines = [l for l in raw.replace("\r", "\n").split("\n") if l.strip()]
    if not lines:
        raise ValueError("Brysbaert download was empty")

    # Locate the concreteness-mean column by header rather than assuming index 1.
    header = [h.strip().lower() for h in lines[0].split(",")]
    word_col = 0
    conc_col = next((i for i, h in enumerate(header) if h in ("conc.m", "conc_m", "concreteness")), None)
    if conc_col is None:
        # No recognizable header — fall back to the documented layout (Word, Conc.M).
        conc_col = 1
        data_lines = lines
        print("  WARNING: no Brysbaert header found; assuming column 1 is Conc.M")
    else:
        data_lines = lines[1:]

    conc = {}
    for line in data_lines:
        parts = line.split(",")
        if len(parts) > conc_col:
            try:
                conc[parts[word_col].strip().lower()] = float(parts[conc_col].strip())
            except ValueError:
                pass
    print(f"Brysbaert loaded: {len(conc):,} words (conc column {conc_col})")
    return conc


def apply_concreteness(words, conc, threshold=3.5):
    tagged = 0
    for w in words:
        score = conc.get(w["word"].lower())
        if score is not None and score >= threshold:
            w["concrete"] = True
            tagged += 1
        elif "concrete" in w:
            del w["concrete"]
    pct = 100 * tagged / len(words) if words else 0
    print(f"Concrete tagged: {tagged} ({pct:.1f}%)")
    return words


# --- Step 10: Apply tier_overrides.json -----------------------------------

def apply_overrides(words):
    overrides = json.loads(Path("tier_overrides.json").read_text())
    final = []
    for w in words:
        ov = overrides.get(w["word"])
        if not ov:
            final.append(w)
        elif ov["action"] == "exclude":
            pass
        else:
            w["band"] = ov["action"]
            w["band_overridden"] = True
            final.append(w)
    excluded = len(words) - len(final)
    rebanded = sum(1 for w in final if w.get("band_overridden"))
    print(f"Overrides applied: {excluded} excluded, {rebanded} re-banded")
    return final


# --- Step 11: Write output and verify -------------------------------------

def write_and_verify(words):
    Path("words_data.json").write_text(
        json.dumps(words, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"\nwrote words_data.json ({Path('words_data.json').stat().st_size // 1024} KB)")

    d = json.loads(Path("words_data.json").read_text())
    print(f"Total words: {len(d)}")
    bands = Counter(w["band"] for w in d)
    for band in ("everyday", "uncommon", "rare", "morerare"):
        print(f"  {band}: {bands.get(band, 0)}")
    print(f"Concrete tagged: {sum(1 for w in d if w.get('concrete'))}")
    print(f"Has examples: {sum(1 for w in d if w.get('examples'))}")
    no_examples = [w["word"] for w in d if not w.get("examples")]
    if no_examples:
        print(f"WARNING: {len(no_examples)} words with no examples: {no_examples[:10]}")

    wmap = {w["word"]: w for w in d}
    checks = [
        ("soonest" not in wmap,    "soonest filtered (lemma=soon)"),
        ("remoteness" not in wmap, "remoteness filtered (lemma=remote)"),
        ("flexibly" not in wmap,   "flexibly filtered (lemma=flexible)"),
        ("extricate" in wmap,      "extricate present"),
        ("morose" in wmap,         "morose present"),
        (wmap.get("freckle", {}).get("concrete"), "freckle tagged concrete"),
        (bands.get("morerare", 0) > 0, "morerare band has words"),
    ]
    all_pass = True
    for result, label in checks:
        status = "✓" if result else "✗"
        print(f"  {status}  {label}")
        if not result:
            all_pass = False
    print("\nAll checks passed." if all_pass else "\nSome checks failed — review before deploying.")


# --- Step 12: Main entry point --------------------------------------------

def main():
    print("=== Prosey pipeline ===\n")

    if not BOOKS_DIR.exists() or not any(BOOKS_DIR.glob("*.txt")):
        print(f"books/ is empty (no .txt files in {BOOKS_DIR.resolve()}).")
        print("Populate the corpus and books/metadata.json first — see CLAUDECODE_BUILD.md Step 0.")
        print("Stopping.")
        return

    print("Loading corpus...")
    books = load_books()
    print(f"Loaded {len(books)} books\n")

    print("Precomputing capitalization stats...")
    cap_counts = precompute_cap_counts(books)

    print("Extracting candidates...")
    candidates = extract_candidates(books, cap_counts)

    print("\nBuilding word objects...")
    words = build_words(candidates)

    print("\nTagging concreteness...")
    conc = load_brysbaert()
    words = apply_concreteness(words, conc)

    print("\nApplying overrides...")
    words = apply_overrides(words)

    print("\nWriting output...")
    write_and_verify(words)


if __name__ == "__main__":
    main()
