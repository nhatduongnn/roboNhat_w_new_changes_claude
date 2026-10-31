#!/usr/bin/env python3
"""Flagged positions (reads disagree with the reference) for decode plots.

User decision 2026-10-28: we always decode the reference genome, and every decode plot marks
positions where the reads say the reference may be wrong, so a TF call there reads as less
reliable. The genome-wide scan (singlefiber/consensus_scan_genome.py, rule in
singlefiber/consensus_seq.py:1-22) writes one TSV per chromosome:

    <flags_dir>/per_chrom/<chrom>.flags.tsv
    chrom pos coverage ref top_alt alt_count del_count ins_top_count ins_top_seq
    cov_fwd cov_rev alt_fwd alt_rev action

`pos` is 1-based. `action` is one of flagged_alt / flagged_indel / flagged_low_cov, or several
joined with ';' (the genome scan has none joined). For an indel-only row top_alt is not
meaningful (alt_count is 0 and argmax falls on A), so callers show del/ins counts instead.

Shared by make_posterior_viewer.py and the locus-slide template so the styles stay the same.
"""
import os

import pandas as pd

FLAGS_DIR = "/usr/project/xtmp/nd141/consensus_scan"

# kind code -> (action name, short legend label). Order is also draw priority (alt on top).
KINDS = [("flagged_alt", "alt base"), ("flagged_indel", "indel"),
         ("flagged_low_cov", "low coverage")]
KIND_CODE = {name: i for i, (name, _) in enumerate(KINDS)}


def flags_path(chrom, flags_dir=FLAGS_DIR):
    return os.path.join(flags_dir, "per_chrom", "%s.flags.tsv" % chrom)


def load_flags(chrom, start, end, flags_dir=FLAGS_DIR):
    """-> DataFrame of flagged rows with start <= pos <= end (1-based inclusive), or None if
    the chromosome has no flags file (so a caller can say so instead of drawing "no flags")."""
    p = flags_path(chrom, flags_dir)
    if not os.path.isfile(p):
        return None
    df = pd.read_csv(p, sep="\t", dtype={"ins_top_seq": str}, keep_default_na=False)
    df = df[(df["pos"] >= start) & (df["pos"] <= end)].copy()
    return df.sort_values("pos").reset_index(drop=True)


def flags_payload(df):
    """Compact columnar dict for JSON (one entry per flagged position and kind).

    p   position (1-based)          k   kind code (KIND_CODE)
    r   ref base                    a   top alt base ("" for a non-alt row)
    ac  alt count                   dc  deletion count
    ic  top insertion count         is  top inserted sequence
    c   coverage (reads with a base or a deletion)
    """
    out = {"p": [], "k": [], "r": [], "a": [], "ac": [], "dc": [], "ic": [], "is": [], "c": []}
    if df is None:
        return out
    for row in df.itertuples(index=False):
        for act in str(row.action).split(";"):
            if act not in KIND_CODE:
                continue
            out["p"].append(int(row.pos))
            out["k"].append(KIND_CODE[act])
            out["r"].append(str(row.ref))
            out["a"].append(str(row.top_alt) if act == "flagged_alt" else "")
            out["ac"].append(int(row.alt_count))
            out["dc"].append(int(row.del_count))
            out["ic"].append(int(row.ins_top_count))
            out["is"].append(str(row.ins_top_seq))
            out["c"].append(int(row.coverage))
    return out


def counts(payload):
    """-> {action name: n positions} for a flags_payload."""
    n = {name: 0 for name, _ in KINDS}
    for k in payload["k"]:
        n[KINDS[k][0]] += 1
    return n
