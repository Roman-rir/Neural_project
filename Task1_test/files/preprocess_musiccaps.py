"""
Task 1 Preprocessing: MusicCaps caption -> tag proxy task
CSE425/EEE474/CSE715 - GNN-BERT Music Context Project

Turns MusicCaps (caption text) into the CSV format train_bert_tags.py expects:
    text, tag_1, tag_2, ..., tag_K   (multi-hot 0/1 columns)

Why this exists: MusicCaps ships free-text captions, not tag labels, so tags
have to be *derived* from the caption text via keyword matching against a
fixed vocabulary. This keeps caption (input) and tags (label) genuinely
distinct — unlike using MagnaTagATune tags as both input and label, which is
a leaky, near-trivial setup.

Input options:
  1. --hf_dataset  Load "google/MusicCaps" directly via the `datasets` library
                    (requires internet access + `pip install datasets`).
  2. --input_csv   A local CSV with at least a caption column (e.g. if you've
                    already downloaded MusicCaps as CSV). Point --caption_col
                    at the right column name.

Output: a single CSV (default: data/processed/musiccaps_tags.csv) containing
        `text` + one 0/1 column per surviving tag. Rows with zero matched
        tags are dropped by default (--keep_untagged to keep them).

Usage:
    python preprocess_musiccaps.py --hf_dataset --top_k 50 \
        --output data/processed/musiccaps_tags.csv
"""

import argparse
import re
from collections import Counter

import pandas as pd


# --------------------------------------------------------------------------- #
# Fixed tag vocabulary
# --------------------------------------------------------------------------- #
# Spans genre / mood / instrument, mirroring the kind of tags MagnaTagATune /
# typical music-tagging setups use. Extend freely — this is a starting point,
# not a fixed spec. Keys are the canonical tag name; values are regex
# alternatives (word-boundary matched, case-insensitive) that count as a hit.

TAG_VOCAB = {
    # Genre
    "rock": [r"rock"],
    "pop": [r"pop"],
    "jazz": [r"jazz"],
    "classical": [r"classical", r"orchestra(l)?"],
    "electronic": [r"electronic", r"electro"],
    "hip hop": [r"hip[\s-]?hop", r"rap"],
    "blues": [r"blues"],
    "country": [r"country"],
    "reggae": [r"reggae"],
    "metal": [r"metal"],
    "folk": [r"folk"],
    "funk": [r"funk"],
    "soul": [r"soul"],
    "ambient": [r"ambient"],
    "techno": [r"techno"],
    "house": [r"house music", r"\bhouse\b"],
    "disco": [r"disco"],
    "punk": [r"punk"],
    "latin": [r"latin"],
    "r&b": [r"r&b", r"r and b", r"rhythm and blues"],
    # Mood
    "happy": [r"happy", r"cheerful", r"joyful"],
    "sad": [r"sad", r"melancholic", r"melancholy", r"somber"],
    "energetic": [r"energetic", r"upbeat", r"lively"],
    "calm": [r"calm", r"peaceful", r"soothing", r"relax(ing|ed)?"],
    "romantic": [r"romantic"],
    "angry": [r"angry", r"aggressive"],
    "dark": [r"dark", r"eerie", r"haunting"],
    "uplifting": [r"uplifting", r"triumphant"],
    "dreamy": [r"dreamy", r"ethereal"],
    "tense": [r"tense", r"suspenseful"],
    # Instrument / production
    "piano": [r"piano"],
    "guitar": [r"guitar"],
    "drums": [r"drum(s)?", r"percussion"],
    "violin": [r"violin", r"strings"],
    "saxophone": [r"saxophone", r"\bsax\b"],
    "synthesizer": [r"synthesizer", r"synth"],
    "bass": [r"bass"],
    "vocals": [r"vocal(s)?", r"singing", r"singer"],
    "flute": [r"flute"],
    "trumpet": [r"trumpet", r"brass"],
    "acoustic": [r"acoustic"],
    "male vocals": [r"\bmale vocal", r"\bmale singer"],
    "female vocals": [r"female vocal", r"female singer"],
    "instrumental": [r"instrumental"],
    "choir": [r"choir", r"chorus"],
    "electric guitar": [r"electric guitar"],
    "live": [r"live recording", r"\blive\b"],
    "slow": [r"\bslow\b", r"slow tempo"],
    "fast": [r"\bfast\b", r"fast tempo", r"up[\s-]?tempo"],
    "loop": [r"\bloop(ing|ed)?\b"],
}

# Precompile
_COMPILED_VOCAB = {
    tag: re.compile("|".join(patterns), re.IGNORECASE)
    for tag, patterns in TAG_VOCAB.items()
}


def extract_tags(caption: str) -> list:
    """Returns the list of vocabulary tags whose pattern matches this caption."""
    hits = []
    for tag, pattern in _COMPILED_VOCAB.items():
        if pattern.search(caption):
            hits.append(tag)
    return hits


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_musiccaps_from_hf():
    from datasets import load_dataset
    ds = load_dataset("google/MusicCaps", split="train")
    df = ds.to_pandas()
    # MusicCaps columns include: ytid, caption, aspect_list, start_s, end_s, ...
    return df[["ytid", "caption"]].rename(columns={"ytid": "clip_id"})


def load_musiccaps_from_csv(path: str, caption_col: str, id_col: str = None):
    df = pd.read_csv(path)
    assert caption_col in df.columns, f"'{caption_col}' not found in {path}"
    out = pd.DataFrame()
    out["caption"] = df[caption_col]
    out["clip_id"] = df[id_col] if id_col and id_col in df.columns else range(len(df))
    return out


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #

def build_tag_csv(df: pd.DataFrame, top_k: int, keep_untagged: bool):
    df = df.dropna(subset=["caption"]).copy()
    df["caption"] = df["caption"].astype(str).str.strip()
    df = df[df["caption"].str.len() > 0]

    print(f"Extracting tags from {len(df)} captions using a {len(TAG_VOCAB)}-tag vocabulary...")
    df["matched_tags"] = df["caption"].apply(extract_tags)

    if not keep_untagged:
        before = len(df)
        df = df[df["matched_tags"].map(len) > 0]
        print(f"Dropped {before - len(df)} captions with zero matched tags "
              f"(pass --keep_untagged to retain them).")

    # Rank tags by frequency, keep top_k
    tag_counts = Counter(t for tags in df["matched_tags"] for t in tags)
    top_tags = [t for t, _ in tag_counts.most_common(top_k)]
    print(f"Top {len(top_tags)} tags by frequency:")
    for t in top_tags:
        print(f"  {t:20s} {tag_counts[t]}")

    if len(top_tags) < top_k:
        print(f"NOTE: only {len(top_tags)} distinct tags matched at least once — "
              f"consider expanding TAG_VOCAB in this script if you need more.")

    # Build multi-hot columns
    for tag in top_tags:
        df[tag] = df["matched_tags"].apply(lambda tags, t=tag: int(t in tags))

    # Drop rows that end up with zero labels after restricting to top_k tags
    label_sum = df[top_tags].sum(axis=1)
    before = len(df)
    df = df[label_sum > 0]
    if len(df) < before:
        print(f"Dropped {before - len(df)} additional rows with zero labels "
              f"after restricting to top-{top_k} tags.")

    out_cols = ["text"] + top_tags
    df = df.rename(columns={"caption": "text"})
    return df[out_cols].reset_index(drop=True), top_tags


def main():
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--hf_dataset", action="store_true",
                      help="Load google/MusicCaps via the `datasets` library.")
    src.add_argument("--input_csv", type=str,
                      help="Path to a local CSV containing MusicCaps captions.")
    parser.add_argument("--caption_col", type=str, default="caption",
                         help="Caption column name when using --input_csv.")
    parser.add_argument("--id_col", type=str, default=None,
                         help="Optional clip-id column name when using --input_csv.")
    parser.add_argument("--top_k", type=int, default=50,
                         help="Number of most frequent tags to keep (matches the 'top-50 tags' spec).")
    parser.add_argument("--keep_untagged", action="store_true",
                         help="Keep captions with zero keyword matches (adds a hard, ambiguous 'no tag' case).")
    parser.add_argument("--output", type=str, default="data/processed/musiccaps_tags.csv")
    args = parser.parse_args()

    if args.hf_dataset:
        raw_df = load_musiccaps_from_hf()
    else:
        raw_df = load_musiccaps_from_csv(args.input_csv, args.caption_col, args.id_col)

    tag_df, top_tags = build_tag_csv(raw_df, args.top_k, args.keep_untagged)

    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    tag_df.to_csv(args.output, index=False)
    print(f"\nSaved {len(tag_df)} labeled examples with {len(top_tags)} tags to {args.output}")
    print("This CSV is ready for train_bert_tags.py --data_path "
          f"{args.output} --text_col text")


if __name__ == "__main__":
    main()
