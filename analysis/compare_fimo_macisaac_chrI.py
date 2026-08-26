"""Whole-chrI FIMO scan for Abf1_murphy vs the 5 MacIsaac ABF1 sites.

Answers a narrow question: if you scan all of chrI with the SAME PWM RoboCOP uses
and keep FIMO's default p < 1e-4, where do the hits land, and do they land on the
sites MacIsaac called?

Everything here is external to RoboCOP -- FIMO is a motif scanner, MacIsaac is a
published site list. Neither is decode output. The decode columns are added only as
a third reference and are read from the h5 posterior table.

Coordinates: FIMO reports 1-based inclusive; the bed is 0-based half-open. Converted
to the bed 0-based frame on entry (start-1), matching the rest of analysis/.

    python compare_fimo_macisaac_chrI.py
"""
import os, sys, glob, subprocess, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pkg"))
import numpy as np
import pandas as pd
import score_robocop as S
import plot_abf1_5sites_decoded as PD

MEME_ENV = "/home/users/nd141/miniconda3/envs/meme/bin"
MEME_FILE = "inputs/motifs_meme.txt"
GENOME = "inputs/SacCer3.fa"
MACISAAC = "inputs/MacIsaac_sacCer3_liftOver_Abf1_Reb1_match_PWM.bed"
FIBER_DIR = "robocop_chrI_maskon_revfix"
SEQ_DIR = "robocop_chrI_seq_maskon_revfix"


def run_fimo(workdir, thresh):
    """FIMO over all of chrI at the given threshold, RoboCOP's background."""
    os.makedirs(workdir, exist_ok=True)
    fa = os.path.join(workdir, "chrI.fa")
    bg = os.path.join(workdir, "robocop_bg.txt")
    oc = os.path.join(workdir, "fimo_out")
    if not os.path.isfile(fa):
        with open(fa, "w") as fh:
            subprocess.run(["samtools", "faidx", GENOME, "chrI"], check=True, stdout=fh)
    if not os.path.isfile(bg):
        import pickle
        b = np.ravel(pickle.load(open(os.path.join(FIBER_DIR, "pwm.p"), "rb"))["background"])[:4]
        with open(bg, "w") as fh:
            fh.write("# order 0\n")
            for letter, v in zip("ACGT", b):
                fh.write("%s %.8f\n" % (letter, v))
    subprocess.run([os.path.join(MEME_ENV, "fimo"), "--motif", "Abf1_murphy",
                    "--bfile", bg, "--thresh", str(thresh), "--oc", oc,
                    MEME_FILE, fa], check=True, capture_output=True)
    df = pd.read_csv(os.path.join(oc, "fimo.tsv"), sep="\t", comment="#")
    df = df[df["motif_id"] == "Abf1_murphy"].copy()
    df["bed_start"] = df["start"].astype(int) - 1      # 1-based -> bed 0-based
    df["bed_stop"] = df["stop"].astype(int) - 1
    df["mid"] = (df["bed_start"] + df["bed_stop"]) // 2
    return df.sort_values("p-value").reset_index(drop=True), oc


def decode_peak(outDir, lo, hi):
    """Max ABF1 fwd/rev posterior in [lo,hi] -- decode output, for reference."""
    dec = S.load_decode(outDir)
    comp, pos, _ = PD.state_composition(dec, "chrI", lo, hi)
    if comp is None:
        return None
    tot = comp["abf1_fwd"] + comp["abf1_rev"]
    j = int(np.argmax(tot))
    return dict(peak=float(tot[j]), at=int(pos[j]),
                fwd=float(comp["abf1_fwd"][j]), rev=float(comp["abf1_rev"][j]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="chrI_fimo")
    ap.add_argument("--thresh", type=float, default=1e-4, help="FIMO default")
    ap.add_argument("--pad", type=int, default=20,
                    help="bp of slack when testing overlap with a MacIsaac site")
    args = ap.parse_args()

    hits, oc = run_fimo(args.workdir, args.thresh)
    print("FIMO whole-chrI scan, Abf1_murphy, RoboCOP background, p < %g" % args.thresh)
    print("  %d hits  (+ %d, - %d)   output: %s"
          % (len(hits), int((hits["strand"] == "+").sum()),
             int((hits["strand"] == "-").sum()), os.path.join(oc, "fimo.tsv")))
    if "q-value" in hits.columns:
        print("  best q-value %.3g  ->  %d hits with q < 0.05"
              % (hits["q-value"].min(), int((hits["q-value"] < 0.05).sum())))

    bed = pd.read_csv(MACISAAC, sep="\t", header=None,
                      names=["chr", "start", "end", "name", "score", "strand"])
    mac = bed[(bed.chr == "chrI") & (bed.name.str.upper() == "ABF1")].sort_values("start")
    mac = mac.reset_index(drop=True)
    print("\nMacIsaac ABF1 sites on chrI: %d" % len(mac))

    print("\n%-5s %-17s %-4s | %-28s | %s"
          % ("site", "MacIsaac span", "str", "nearest FIMO hit (p<1e-4)", "decode ABF1 peak (fwd/rev)"))
    print("-" * 132)
    recovered = 0
    for i, r in mac.iterrows():
        m0, m1, mstr = int(r.start), int(r.end), r.strand
        mmid = (m0 + m1) // 2
        d = (hits["mid"] - mmid).abs()
        j = int(d.idxmin()) if len(hits) else None
        if j is None:
            cell = "none"
        else:
            h = hits.loc[j]
            dist = int(h["mid"] - mmid)
            ov = not (h["bed_stop"] < m0 - args.pad or h["bed_start"] > m1 + args.pad)
            if ov:
                recovered += 1
            cell = "%s %d-%d %s p=%.2g %s" % ("HIT " if ov else "far ", h["bed_start"],
                                              h["bed_stop"], h["strand"], h["p-value"],
                                              "(%+d bp)" % dist)
        fib = decode_peak(FIBER_DIR, mmid - 200, mmid + 200)
        seq = decode_peak(SEQ_DIR, mmid - 200, mmid + 200)
        dc = "fiber %.3f@%d (%.2f/%.2f)  seq %.3f@%d (%.2f/%.2f)" % (
            fib["peak"], fib["at"], fib["fwd"], fib["rev"],
            seq["peak"], seq["at"], seq["fwd"], seq["rev"]) if fib and seq else "-"
        print("#%-4d %-17s %-4s | %-28s | %s"
              % (i + 1, "%d-%d" % (m0, m1), mstr, cell, dc))

    print("\n%d of %d MacIsaac sites have a FIMO p<1e-4 hit within +/-%d bp"
          % (recovered, len(mac), args.pad))

    # the converse: how many FIMO hits are NOT MacIsaac sites
    onmac = 0
    for _, h in hits.iterrows():
        if ((mac.start - args.pad <= h["bed_stop"]) & (mac.end + args.pad >= h["bed_start"])).any():
            onmac += 1
    print("%d of %d FIMO hits fall on a MacIsaac site; %d do not"
          % (onmac, len(hits), len(hits) - onmac))

    print("\nTop 15 FIMO hits on chrI by p-value")
    print("%-5s %-17s %-4s %9s %11s  %-16s %s"
          % ("rank", "span (bed)", "str", "score", "p-value", "sequence", "MacIsaac?"))
    for k in range(min(15, len(hits))):
        h = hits.loc[k]
        on = ((mac.start - args.pad <= h["bed_stop"]) &
              (mac.end + args.pad >= h["bed_start"]))
        which = ("site #%d" % (int(np.flatnonzero(on)[0]) + 1)) if on.any() else "-"
        print("%-5d %-17s %-4s %9.3f %11.3g  %-16s %s"
              % (k + 1, "%d-%d" % (h["bed_start"], h["bed_stop"]), h["strand"],
                 h["score"], h["p-value"], h["matched_sequence"], which))


if __name__ == "__main__":
    main()
