"""Assemble three motif databases in one MEME file each, for TOMTOM comparison.

  shipped  inputs/motifs_meme.txt              -- what RoboCOP actually decodes with
  jaspar   JASPAR CORE / fungi / latest        -- fetched from the JASPAR REST API
  rossi    Rossi ChExMix MEME motifs           -- pre-computed, read off disk

Nothing here modifies an existing input. Everything lands in analysis/motifdb/.

WHY A SHARED BACKGROUND FILE. Each source ships its own header background: motifs_meme.txt
claims uniform 0.25, JASPAR's .meme export claims 0.25, Rossi's carries the real yeast
composition. RoboCOP ignores all three and counts the genome itself
(parameterize.computeBackground, called from getDBFconc.py:76). So the comparison is run
with -bfile yeast_bg.txt, i.e. the background the MODEL assumes, not the one the file
claims. computeBackground is replicated here rather than imported so this script does not
drag in rpy2; the values are asserted against the known RoboCOP numbers.

ROSSI MOTIF SELECTION. A sample yields up to 3 motifs and most TFs have 2 samples, so a TF
can carry ~6 candidate matrices -- but most are noise. For ABF1 sample 15254, Motif 1 is
E=1.7e-484 while Motifs 2 and 3 are E=2.1e+03 and 2.8e+03. Everything above --rossi-evalue
is dropped; every survivor is kept and named <GENE>_rossi_<sample>_m<k>, so the comparison
can report the best match AND how many candidates there were.

The sample -> TF map comes from our own rossi_peak_w_strand_all_TFs.bed (sample_id and TF
columns), which is 1:1 over 718 samples. No external metadata sheet is needed.

    conda activate pyranges_env3
    python build_motif_dbs.py
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

MEME_BIN = "/home/users/nd141/miniconda3/envs/meme/bin"
SHIPPED = "inputs/motifs_meme.txt"
GENOME = "inputs/SacCer3.fa"
ROSSI_BED = "inputs/rossi_peak_w_strand_all_TFs.bed"
ROSSI_ROOT = "/usr/xtmp/nd141/projects/data/rossi_strand"
OUTDIR = "motifdb"
JASPAR_LIST = ("https://jaspar.elixir.no/api/v1/matrix/"
               "?collection=CORE&tax_group=fungi&version=latest&page_size=500")
JASPAR_MEME = "https://jaspar.elixir.no/api/v1/matrix/%s.meme"

# RoboCOP's own genome background, for the assertion below (parameterize.computeBackground
# over inputs/SacCer3.fa). Hardcoded ONLY as a check that the reimplementation agrees.
ROBOCOP_BG = {"A": 0.30980641, "C": 0.19088229, "G": 0.19059636, "T": 0.30871494}

MEME_HEADER = """MEME version 4

ALPHABET= ACGT

strands: + -

Background letter frequencies
A %.6f C %.6f G %.6f T %.6f

"""


def gene_of(motif_name):
    """Shipped/Rossi motif name -> gene symbol. 'Abf1_murphy'->ABF1, 'Rap1_telomeric'->RAP1."""
    return motif_name.rsplit("_", 1)[0].upper()


# ---------------------------------------------------------------- background

def compute_background(fasta):
    """Byte-for-byte equivalent of parameterize.computeBackground, without the Bio import."""
    n = collections.Counter()
    with open(fasta) as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            n.update(line.strip().upper())
    tot = sum(n[b] for b in "ACGT")
    return {b: n[b] / tot for b in "ACGT"}


def write_background(bg, path):
    with open(path, "w") as fh:
        fh.write("# order 0\n")
        for b in "ACGT":
            fh.write("%s %.8f\n" % (b, bg[b]))


# ---------------------------------------------------------------- meme parsing

def parse_meme(text):
    """Minimal-MEME parser -> [(name, header_line, [matrix rows as text])].

    Kept deliberately dumb: it copies the letter-probability rows through verbatim rather
    than round-tripping through floats, so no precision is lost relative to the source.
    """
    out, lines, i = [], text.split("\n"), 0
    while i < len(lines):
        if lines[i].startswith("MOTIF"):
            name_parts = lines[i].split()
            name = name_parts[1] if len(name_parts) > 1 else "unnamed"
            alt = name_parts[2] if len(name_parts) > 2 else ""
            j = i + 1
            while j < len(lines) and not lines[j].startswith("letter-probability"):
                if lines[j].startswith("MOTIF"):
                    break
                j += 1
            if j >= len(lines) or not lines[j].startswith("letter-probability"):
                i += 1
                continue
            hdr = lines[j]
            w = int(re.search(r"w=\s*(\d+)", hdr).group(1))
            rows = lines[j + 1:j + 1 + w]
            out.append({"name": name, "alt": alt, "header": hdr, "rows": rows})
            i = j + 1 + w
        else:
            i += 1
    return out


def evalue_of(header):
    m = re.search(r"E=\s*(\S+)", header)
    try:
        return float(m.group(1)) if m else float("nan")
    except ValueError:
        return float("nan")


def emit(motifs, bg, path):
    with open(path, "w") as fh:
        fh.write(MEME_HEADER % (bg["A"], bg["C"], bg["G"], bg["T"]))
        for m in motifs:
            fh.write("MOTIF %s %s\n" % (m["name"], m.get("alt") or m["name"]))
            fh.write(m["header"] + "\n")
            fh.write("\n".join(m["rows"]) + "\n\n")
    return len(motifs)


# ---------------------------------------------------------------- sources

def fetch(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read().decode()
        except (urllib.error.URLError, TimeoutError) as e:
            if k == tries - 1:
                raise
            print("    retry %d/%d (%s)" % (k + 1, tries - 1, e), flush=True)
            time.sleep(2 * (k + 1))


def build_jaspar(cache):
    """Fetch CORE/fungi/latest. Cached, because it is ~193 HTTP round trips."""
    if os.path.isfile(cache):
        print("  using cached %s" % cache)
        return parse_meme(open(cache).read())
    listing = json.loads(fetch(JASPAR_LIST))
    recs = listing["results"]
    print("  JASPAR listing: %d matrices (API count %d)" % (len(recs), listing["count"]))
    if listing.get("next"):
        sys.exit("JASPAR listing is paginated -- raise page_size")
    motifs = []
    for k, r in enumerate(recs, 1):
        mid, name = r["matrix_id"], r["name"].upper()
        got = parse_meme(fetch(JASPAR_MEME % mid))
        if len(got) != 1:
            print("    SKIP %s: %d motifs in export" % (mid, len(got)))
            continue
        m = got[0]
        m["name"], m["alt"] = "%s_jaspar_%s" % (name, mid), name
        motifs.append(m)
        if k % 40 == 0:
            print("    %d/%d" % (k, len(recs)), flush=True)
    return motifs


def rossi_sample_map(bed):
    """sample_id -> gene symbol, from our own peak bed."""
    m = {}
    with open(bed) as fh:
        hdr = fh.readline().rstrip("\n").split("\t")
        si, ti = hdr.index("sample_id"), hdr.index("TF")
        for line in fh:
            f = line.rstrip("\n").split("\t")
            m[f[si]] = f[ti].upper()
    return m


def build_rossi(samp2tf, evalue_max):
    motifs, stats = [], collections.Counter()
    samples = sorted(samp2tf)
    for k, s in enumerate(samples, 1):
        path = "%s/%s/%s_YEP/%s_MEME_Motifs.txt" % (ROSSI_ROOT, s, s, s)
        if not os.path.isfile(path):
            stats["no_meme_file"] += 1
            continue
        try:
            txt = subprocess.run([os.path.join(MEME_BIN, "meme2meme"), path],
                                 capture_output=True, text=True, check=True).stdout
        except subprocess.CalledProcessError as e:
            print("    meme2meme FAILED on %s: %s" % (s, e.stderr.strip()[:120]))
            stats["meme2meme_failed"] += 1
            continue
        kept = 0
        for m in parse_meme(txt):
            ev = evalue_of(m["header"])
            stats["motifs_seen"] += 1
            if not (ev <= evalue_max):
                stats["dropped_evalue"] += 1
                continue
            idx = re.sub(r"\D", "", m["name"]) or str(kept + 1)
            m["name"] = "%s_rossi_%s_m%s" % (samp2tf[s], s, idx)
            m["alt"] = samp2tf[s]
            motifs.append(m)
            kept += 1
        stats["samples_used" if kept else "samples_all_dropped"] += 1
        if k % 150 == 0:
            print("    %d/%d samples" % (k, len(samples)), flush=True)
    return motifs, stats


# ---------------------------------------------------------------- coverage

def write_coverage(shipped, jaspar, rossi, path):
    ship_by_gene = collections.defaultdict(list)
    for m in shipped:
        ship_by_gene[gene_of(m["name"])].append(m["name"])
    jas_by_gene = collections.defaultdict(list)
    for m in jaspar:
        jas_by_gene[m["alt"]].append(m["name"])
    ros_by_gene = collections.defaultdict(list)
    for m in rossi:
        ros_by_gene[m["alt"]].append(m["name"])

    buckets = collections.Counter()
    with open(path, "w") as fh:
        fh.write("gene\tn_shipped\tn_jaspar\tn_rossi\tbucket\tsources\tshipped_names\n")
        for g in sorted(ship_by_gene):
            nj, nr = len(jas_by_gene.get(g, [])), len(ros_by_gene.get(g, []))
            if nj and nr:
                b = "3way"
            elif nj:
                b = "2way_jaspar"
            elif nr:
                b = "2way_rossi"
            else:
                b = "nothing_to_compare"
            buckets[b] += 1
            srcs = ",".join(sorted({n.rsplit("_", 1)[1] for n in ship_by_gene[g]}))
            fh.write("%s\t%d\t%d\t%d\t%s\t%s\t%s\n"
                     % (g, len(ship_by_gene[g]), nj, nr, b, srcs,
                        ";".join(sorted(ship_by_gene[g]))))
    return buckets, len(ship_by_gene), ros_by_gene, jas_by_gene


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--rossi-evalue", type=float, default=0.05,
                    help="drop Rossi motifs with E above this (default 0.05)")
    ap.add_argument("--refetch-jaspar", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    P = lambda f: os.path.join(args.outdir, f)

    print("== background ==")
    bg = compute_background(GENOME)
    for b in "ACGT":
        d = abs(bg[b] - ROBOCOP_BG[b])
        print("  %s %.8f   (RoboCOP %.8f, diff %.2e)" % (b, bg[b], ROBOCOP_BG[b], d))
        assert d < 1e-6, "background disagrees with RoboCOP for %s" % b
    write_background(bg, P("yeast_bg.txt"))
    print("  wrote %s  -- ASSERTION PASSED, matches parameterize.computeBackground" % P("yeast_bg.txt"))

    print("\n== shipped ==")
    shipped = parse_meme(open(SHIPPED).read())
    emit(shipped, bg, P("shipped.meme"))
    src = collections.Counter(m["name"].rsplit("_", 1)[1] for m in shipped)
    print("  %d motifs, %d genes, sources: %s"
          % (len(shipped), len({gene_of(m["name"]) for m in shipped}), dict(src.most_common())))

    print("\n== JASPAR ==")
    cache = P(".jaspar_raw.meme")
    if args.refetch_jaspar and os.path.isfile(cache):
        os.remove(cache)
    jaspar = build_jaspar(cache)
    emit(jaspar, bg, cache)
    emit(jaspar, bg, P("jaspar.meme"))
    print("  %d matrices, %d gene names" % (len(jaspar), len({m["alt"] for m in jaspar})))

    print("\n== Rossi ==")
    samp2tf = rossi_sample_map(ROSSI_BED)
    print("  sample->TF map: %d samples, %d distinct TFs"
          % (len(samp2tf), len(set(samp2tf.values()))))
    rossi, stats = build_rossi(samp2tf, args.rossi_evalue)
    emit(rossi, bg, P("rossi.meme"))
    print("  motifs seen %d | kept %d | dropped on E>%g %d"
          % (stats["motifs_seen"], len(rossi), args.rossi_evalue, stats["dropped_evalue"]))
    print("  samples with >=1 kept %d | all dropped %d | no meme file %d | meme2meme failed %d"
          % (stats["samples_used"], stats["samples_all_dropped"],
             stats["no_meme_file"], stats["meme2meme_failed"]))
    print("  %d distinct TFs have a Rossi motif" % len({m["alt"] for m in rossi}))

    print("\n== coverage ==")
    buckets, ngenes, ros_by_gene, jas_by_gene = write_coverage(
        shipped, jaspar, rossi, P("coverage.tsv"))
    for b in ("3way", "2way_jaspar", "2way_rossi", "nothing_to_compare"):
        print("  %-20s %d" % (b, buckets[b]))
    print("  %-20s %d" % ("TOTAL genes", sum(buckets.values())))
    assert sum(buckets.values()) == ngenes, "buckets do not sum to shipped genes"

    fitted = ["ABF1", "CIN5", "FHL1", "FKH1", "MCM1", "NHP6A",
              "RAP1", "REB1", "SKO1", "SPT15", "TBF1", "UME6"]
    print("\n  the 12 fitted TFs:")
    print("  %-8s %-8s %-8s %s" % ("gene", "jaspar", "rossi", "note"))
    for g in fitted:
        nj, nr = len(jas_by_gene.get(g, [])), len(ros_by_gene.get(g, []))
        note = "" if (nj and nr) else "NOT 3-WAY"
        print("  %-8s %-8d %-8d %s" % (g, nj, nr, note))
    print("\nwrote %s/{shipped,jaspar,rossi}.meme, yeast_bg.txt, coverage.tsv" % args.outdir)


if __name__ == "__main__":
    main()
