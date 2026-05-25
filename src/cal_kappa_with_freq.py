#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate contextual diversity  v = 1/κ  from a vertical-format corpus,
following Nagata & Tanaka-Ishii (ACL 2025), adapted for Czech.

Input
-----
Vertical (CONLL-like) file with TAB-separated columns
    surface  lemma  morphological-tag  [...]
Sentences are bounded by <s ...> ... </s>. Other markup (<doc>, <text>,
<p>, <block>, ...) is ignored. Only columns 1–3 are read; any number of
trailing columns is tolerated.

Pipeline
--------
1. Parse vertical to sentences of (surface, lemma, POS) triples.
   POS = first character of the morphological tag (PDT positional tagset).
2. Encode each sentence with a RoBERTa-style model using the SURFACE forms,
   so the encoder sees grammatical Czech.
3. Mean-pool subword hidden states back to word level via tokenizer.word_ids().
4. L2-normalise each per-occurrence word vector (direction only).
5. Accumulate sums and counts under the LEMMA (or LEMMA/POS with --pos-key).
6. Estimate concentration  κ ≈ l (d − l²) / (1 − l²)   (Banerjee et al. 2005),
   where l is the norm of the mean of unit vectors and d their dimension.
   Contextual diversity  v = 1/κ.

Output
------
TAB-separated rows on stdout:  key  κ  v  freq  (sorted by freq desc).
"""

import argparse
import sys
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

try:
    from tqdm import tqdm
except ImportError:                                            # pragma: no cover
    def tqdm(x, **kw): return x


# ---------- vertical parsing ------------------------------------------------

def parse_vertical(path, drop_punct=True):
    """Yield sentences as list[(surface, lemma, pos)]."""
    sent, in_sent = [], False
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            if line.startswith("<s"):
                sent, in_sent = [], True
                continue
            if line.startswith("</s"):
                if sent:
                    yield sent
                sent, in_sent = [], False
                continue
            if line.startswith("<"):           # <doc>, <text>, <p>, <block>, ...
                continue
            if not in_sent:
                continue
            cols = line.split("\t")
            if len(cols) < 3:
                continue
            surface, lemma, tag = cols[0], cols[1], cols[2]
            pos = tag[0] if tag else "X"
            if drop_punct and pos == "Z":
                continue
            sent.append((surface, lemma, pos))
    if sent:                                   # flush if file ends without </s>
        yield sent


# ---------- batching --------------------------------------------------------

def batched(iterable, n):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


# ---------- accumulation ----------------------------------------------------

def accumulate(model, tokenizer, sentences, key_fn,
               layer=-1, batch_size=32, max_length=512, device="cpu"):
    """Encode surface forms; accumulate unit vectors under key_fn(surf, lem, pos)."""
    model.eval().to(device)
    sum_vecs = {}
    freqs    = defaultdict(int)

    for batch in tqdm(batched(sentences, batch_size), desc="Encoding", unit="batch"):
        surfaces = [[s for s, _, _ in sent] for sent in batch]
        keys     = [[key_fn(*t)  for t in sent] for sent in batch]

        enc = tokenizer(
            surfaces,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc_dev = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            H = model(**enc_dev, output_hidden_states=True).hidden_states[layer]

        for b, key_list in enumerate(keys):
            word_ids = enc.word_ids(batch_index=b)
            # group subword positions by source word index
            groups = defaultdict(list)
            for tok_pos, wid in enumerate(word_ids):
                if wid is not None:
                    groups[wid].append(tok_pos)

            for wid, positions in groups.items():
                if wid >= len(key_list):        # safety for truncated tails
                    continue
                k = key_list[wid]
                idx = torch.tensor(positions, device=device, dtype=torch.long)
                vec = H[b].index_select(0, idx).mean(dim=0)
                vec = vec.cpu().numpy().astype(np.float64)
                n = np.linalg.norm(vec)
                if n == 0.0 or not np.isfinite(n):
                    continue
                vec /= n                        # unit-normalise: direction only
                if k in sum_vecs:
                    sum_vecs[k] += vec
                else:
                    sum_vecs[k] = vec.copy()
                freqs[k] += 1

    return sum_vecs, freqs


# ---------- κ and v ---------------------------------------------------------

def cal_concentration(mean_norm, vector_size):
    """κ ≈ l(d − l²) / (1 − l²),  Banerjee et al. 2005."""
    l, d = float(mean_norm), float(vector_size)
    if l >= 1.0:
        return float("inf")
    return l * (d - l * l) / (1.0 - l * l)


def compute_rows(sum_vecs, freqs, freq_threshold):
    rows = []
    for k, vsum in sum_vecs.items():
        f = freqs[k]
        if f < freq_threshold:
            continue
        l = float(np.linalg.norm(vsum / f))
        kappa = cal_concentration(l, vsum.size)
        v = 1.0 / kappa if np.isfinite(kappa) and kappa > 0 else float("nan")
        rows.append((k, kappa, v, f))
    return rows


# ---------- CLI -------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Contextual diversity from a vertical Czech corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("corpus", help="Path to vertical-format corpus file.")
    p.add_argument("-m", "--model", default="ufal/robeczech-base",
                   help="HuggingFace model id.")
    p.add_argument("-b", "--batch-size", type=int, default=32)
    p.add_argument("-l", "--layer", type=int, default=-1,
                   help="Hidden-state layer to use (-1 = final).")
    p.add_argument("-f", "--freq-threshold", type=int, default=100,
                   help="Minimum frequency to include a key in the output.")
    p.add_argument("-L", "--max-length", type=int, default=512,
                   help="Maximum subword sequence length per sentence.")
    p.add_argument("--pos-key", action="store_true",
                   help="Key by lemma/POS (e.g. stát/N vs stát/V) instead of lemma alone.")
    p.add_argument("--keep-punct", action="store_true",
                   help="Keep punctuation tokens (PDT tag starts with Z).")
    p.add_argument("--digit", type=int, default=4,
                   help="Decimal places for κ and v in output.")
    p.add_argument("--header", action="store_true",
                   help="Print a header line.")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"# device: {device}", file=sys.stderr)
    print(f"# model:  {args.model}", file=sys.stderr)
    print(f"# layer:  {args.layer}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if not tokenizer.is_fast:
        sys.exit("Error: a fast tokenizer is required (needs word_ids()).")
    model = AutoModel.from_pretrained(args.model)

    if args.pos_key:
        def key_fn(surf, lem, pos): return f"{lem}/{pos}"
    else:
        def key_fn(surf, lem, pos): return lem

    sentences = parse_vertical(args.corpus, drop_punct=not args.keep_punct)
    sum_vecs, freqs = accumulate(
        model, tokenizer, sentences, key_fn,
        layer=args.layer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    rows = compute_rows(sum_vecs, freqs, args.freq_threshold)
    rows.sort(key=lambda r: -r[3])

    print(f"# kept {len(rows)} keys (freq ≥ {args.freq_threshold}); "
          f"discarded {len(sum_vecs) - len(rows)}", file=sys.stderr)

    if args.header:
        print("key\tkappa\tv\tfreq")
    for key, kappa, v, freq in rows:
        kappa_s = f"{kappa:.{args.digit}f}" if np.isfinite(kappa) else "inf"
        v_s     = f"{v:.{args.digit}f}"     if np.isfinite(v)     else "nan"
        print(f"{key}\t{kappa_s}\t{v_s}\t{freq}")


if __name__ == "__main__":
    main()