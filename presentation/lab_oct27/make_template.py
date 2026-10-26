"""Build deck_template.html for the Oct 27 lab deck with the look and navigation of the Oct 6 deck.

Same recipe as presentation/lab_oct6/make_template.py (copied, not imported): copies from
presentation/talk.html (read only, never edited) the Google Fonts links, the whole <style> block, the
notes/overview/HUD markup and the main <script> (deck state, fit-to-stage, HUD section bar, overview,
keyboard/wheel/touch navigation, locus drawing). Changes to the copy: NOTES filled from notes.json,
SEC_NAMES replaced, the talk's chart code dropped, document.title and HUD brand renamed, storage keys
"oct27.*", and the Oct 6 hooks on drawLocus (chip display labels, A/T-run track). Slides come from
slides_fragment.html, extra CSS from oct27_extra.css. The four locus payload placeholders
%%LOC_RPC19%% %%LOC_XIV440%% %%LOC_XIV724%% %%LOC_IV1200%% are filled by build_deck.py.
"""
import json, os, re
H = os.path.dirname(os.path.abspath(__file__))
talk = open(os.path.join(H, "..", "talk.html"), encoding="utf-8").read()
TITLE = "Oct_27th_lab_meeting"


def between(s, a, b, inclusive=False):
    i = s.index(a); j = s.index(b, i + len(a))
    return s[i:j + len(b)] if inclusive else s[i + len(a):j]


fonts = "\n".join(l for l in talk.splitlines()[:6] if l.startswith("<link"))
css = between(talk, "<style>\n", "\n</style>")
chrome = between(talk, '<div id="notes" hidden>', "</nav>", inclusive=True)
old = '<span class="brand">RoboCOP × Fiber-seq</span>'
assert old in chrome; chrome = chrome.replace(old, '<span class="brand">Lab meeting · 27 Oct 2026</span>')
js = talk[talk.index("<script>\n(function(){"):]
js = js[:js.index("</script>") + len("</script>")]

notes = json.load(open(os.path.join(H, "notes.json"), encoding="utf-8"))
js = re.sub(r"var NOTES = \{.*?\n\};", lambda m: "var NOTES = " + json.dumps(notes, ensure_ascii=False) + ";", js, count=1, flags=re.S)
js = re.sub(r"var SEC_NAMES = \[.*?\];",
            'var SEC_NAMES = ["Title","What changed","Four loci, before and after","Summary"];', js, count=1)
a = js.index("/* ---------------- charts"); b = js.index("/* ---------------- locus slides")
js = js[:a] + js[b:]
old = 'document.title = "RoboCOP Meets Fiber-seq";'
assert old in js; js = js.replace(old, 'document.title = "%s";' % TITLE)
js = js.replace('"rfs.showAll"', '"oct27.showAll"').replace('"rfs.notes"', '"oct27.notes"')
old = 'var txt = "\\u00b7 " + call.run + " "'
assert old in js; js = js.replace(old, 'var txt = "\\u00b7 " + (call.label || call.run) + " "')
old = 'txt += " \\u00b7 " + ar + " "'
assert old in js; js = js.replace(old, 'txt += " \\u00b7 " + ((call.alsoLabel || {})[ar] || ar) + " "')
old = "  function drawAxis(c){"
tract = '''  function drawTract(c){                /* Oct 6 addition: A/T homopolymer runs */
    var s = setup(c), ctx = s.ctx, h = s.h;
    gridX(ctx, h);
    ctx.font = "14px " + mono; ctx.textAlign = "left";
    (T.tracts || []).forEach(function(t){
      var x0 = xOf(t[0]) - 0.5, x1 = xOf(t[1]) + 0.5;
      ctx.fillStyle = ink2; ctx.fillRect(x0, 8, Math.max(3, x1 - x0), 10);
      ctx.fillText("poly-" + t[2] + " " + (t[1]-t[0]+1) + " bp", x1 + 6, 18);
    });
    frame(ctx, h);
  }
'''
assert old in js; js = js.replace(old, tract + old, 1)
old = 'else if (k === "gene") drawGene(c);'
assert old in js; js = js.replace(old, 'else if (k === "tract") drawTract(c); else if (k === "gene") drawGene(c);', 1)

# Oct 27: short rows (44 px) only label 0 and 1 on the y axis, so the three labels do not collide
old = "yGrid(ctx, y, [0,0.5,1]); gridX(ctx, h);"
assert js.count(old) == 2; js = js.replace(old, "yGrid(ctx, y, h < 60 ? [0,1] : [0,0.5,1]); gridX(ctx, h);")

# Oct 27: optional run.diff = [[start, end], ...] spans where before and after differ; drawn as a bar along the row's bottom edge
old = "ctx.strokeStyle = T.colors.tf; ctx.lineWidth = 2.2; pathSeries(ctx, run.tf, y); ctx.stroke();"
assert js.count(old) == 1
js = js.replace(old, old + """
    if (run.diff){ ctx.fillStyle = tok("--pending-ink");
      run.diff.forEach(function(d){ var x0 = xOf(Math.max(d[0], T.s)) - 0.5, x1 = xOf(Math.min(d[1], T.e)) + 0.5; ctx.fillRect(x0, h - 5, Math.max(2, x1 - x0), 5); }); }""")

slides = open(os.path.join(H, "slides_fragment.html"), encoding="utf-8").read()
extra = open(os.path.join(H, "oct27_extra.css"), encoding="utf-8").read()
out = ('<meta charset="utf-8">\n<title>%s</title>\n' % TITLE + fonts + "\n<style>\n" + css + "\n" + extra + "</style>\n\n"
       '<div id="viewport">\n<div id="stage">\n\n' + slides + "\n</div>\n</div>\n\n" + chrome + "\n\n"
       '<script type="application/json" id="locus-rpc19">%%LOC_RPC19%%</script>\n'
       '<script type="application/json" id="locus-xiv440">%%LOC_XIV440%%</script>\n'
       '<script type="application/json" id="locus-xiv724">%%LOC_XIV724%%</script>\n'
       '<script type="application/json" id="locus-iv1200">%%LOC_IV1200%%</script>\n' + js + "\n")
open(os.path.join(H, "deck_template.html"), "w", encoding="utf-8").write(out)
print("wrote deck_template.html", len(out))
