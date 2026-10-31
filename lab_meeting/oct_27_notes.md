# Lab meeting notes — 2026-10-27

Deck: https://claude.ai/artifact/2TCJW7cRfv87hivaFSchu9 (`presentation/lab_oct27/`).
Notes as taken by the user (typos fixed only). Open questions for each are listed below; answers to be added.

- Nucleosome model with Brogaard: needs to be symmetric; we can average them and see.
- Apply start and end anywhere, and still trim.
- Flag reads that differ from the reference genome; don't adopt them. We move on with the reference genome.
- For fiberCOP: the TF probability should not be so crazy, and the probability should be better than aggregate.
- Start a spreadsheet to keep track of all the choices we made for the model and why we made them
  (I need to understand why I made those choices; some may be arbitrary).

## Clarified with the user (2026-10-28)

1. **Nucleosome model.** Retrain the sequence model on the Brogaard top-2000 well-positioned nucleosomes
   (`/usr/project/xtmp/nd141/projects/Fiber_seq/process_nanopore_sequencing/Mar_19_24/Brogaard_nuc_positions.sacCer3.top2000.1bp_window.bed`)
   instead of the 9,271 Chereji +1/−1 nucleosomes (`analysis/inputs/Chereji_2018_+1_-1_nucs.bed`, 5,000 drawn at random).
   - Make it symmetric (user idea: train Watson, train Crick, average).
   - Refit the m6A profile with the same, correct centre (no 1 bp mismatch; keep sequence and m6A coordinates consistent).
   - Retest on the same 100 reads (both-directions test).
2. **Start/end anywhere + trim.** Wanted for both aggregate and single-fiber, but the user signs off only after a simple
   explanation of start anywhere, end anywhere and trim, and how they differ. Trim = drop the unreliable start and end of each
   fiber's decode (exists already or to be added).
3. **Reference vs read sequence.** Stick with the reference genome at ALL times (indels are hard). The consensus rule
   (≥ 10 reads, > 50% agree on one different base) now only FLAGS the position in a file; later maybe on the decode plot.
   Count how often it triggers per chromosome.
4. **TF probability.** The transition (start) probability is the same in aggregate and single-fiber, set by concentration.
   User's question: does the aggregate really carry more evidence than one fiber (binomial vs Bernoulli both give a 0–1
   probability)? Find why a TF is never entered on a single read and what to change (concentration or otherwise).
5. **Choices spreadsheet.** A shared artifact. Every choice in the whole project, from raw data to the end.
   Columns: choice · value · where in code · why · who decided / when · arbitrary? · evidence · revisit?
2b. **Trimming (user, 2026-10-28):** if ever done, trimming is AFTER decoding only (drop the decoded start/end of a fiber).
   In the aggregate decode, averaging the 1,000 bp segment overlaps already stands in for trimming.
   **Not implemented for now; kept as a note.**
3b. **Flags on plots (user, 2026-10-28):** flagged positions must be drawn on every decode plot, the posterior viewer first,
   so a TF call there reads as less reliable. Agent launched 2026-10-28.
5b. **Outcome (2026-10-30):**
   - Start anywhere adopted.
   - Brogaard symmetric nucleosome model + all-7 m6A adopted as the single-fiber default.
   - Per-site fraction-bound model chosen for per-read TF calls; the ×138 prior raise was rejected.
   - Details are in SINGLEFIBER_PLAN.md "Update 2026-10-26 → 10-30".
5c. **Raised 2026-10-31:** background m6A rose 0.138 → 0.155 when refit on all 7 barcodes. Check this is not caused by hypermethylated outlier reads, one barcode, or the threshold, and review read filtering. Planned for the next session (SINGLEFIBER_PLAN.md "Next session").
6. **Priority:** spreadsheet and symmetric nucleosome model first. Agents launched 2026-10-28 for 1, 3 (scan), 4 and 5.
