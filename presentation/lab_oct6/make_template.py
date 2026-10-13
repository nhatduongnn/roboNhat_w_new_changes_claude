"""Build deck_template.html for the Oct 6 lab deck with the look and navigation of the talk deck.
Copies from presentation/talk.html (read only, never edited): the Google Fonts links, the whole
<style> block, the notes/overview/HUD markup, and the main <script> (deck state, fit-to-stage,
HUD section bar, overview, keyboard/wheel/touch navigation, locus drawing). Changes to the copy:
NOTES emptied, SEC_NAMES replaced, the talk's chart code dropped (its hosts are not in this deck),
document.title and HUD brand renamed, and two small hooks added to drawLocus (an A/T-run track,
and display labels on site chips). Slides come from slides_fragment.html, extra CSS from
oct6_extra.css, locus data placeholders %%RPC19UP%% %%RPC19DOWN%% %%FIXXIVUP%% %%FIXXIVDOWN%% %%FIXXII%% are filled by build_deck.py.
"""
import os, re
H = os.path.dirname(os.path.abspath(__file__))
talk = open(os.path.join(H, "..", "talk.html"), encoding="utf-8").read()

def between(s, a, b, inclusive=False):
    i = s.index(a); j = s.index(b, i + len(a))
    return s[i:j + len(b)] if inclusive else s[i + len(a):j]

fonts = "\n".join(l for l in talk.splitlines()[:6] if l.startswith("<link"))
css = between(talk, "<style>\n", "\n</style>")
chrome = between(talk, '<div id="notes" hidden>', "</nav>", inclusive=True)
chrome = chrome.replace('<span class="brand">RoboCOP × Fiber-seq</span>', '<span class="brand">Lab meeting · 6 Oct 2026</span>')
js = talk[talk.index("<script>\n(function(){"):]
js = js[:js.index("</script>") + len("</script>")]

# NOTES -> empty; section names for this deck
js = re.sub(r"var NOTES = \{.*?\n\};", "var NOTES = {};", js, count=1, flags=re.S)
js = re.sub(r"var SEC_NAMES = \[.*?\];",
            'var SEC_NAMES = ["Title","Tuning runs","The suspicious locus","The posterior bug","The fix"];', js, count=1)
# drop the talk's chart code (hosts ch9a/ch9b/ch10/chB1 do not exist here)
a = js.index("/* ---------------- charts"); b = js.index("/* ---------------- locus slides")
js = js[:a] + js[b:]
js = js.replace('document.title = "RoboCOP Meets Fiber-seq";', 'document.title = "Oct_6th_2026_lab_presentation";')
js = js.replace('"rfs.showAll"', '"oct6.showAll"').replace('"rfs.notes"', '"oct6.notes"')
# locus hooks: chip display labels, A/T-run track
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

slides = open(os.path.join(H, "slides_fragment.html"), encoding="utf-8").read()
extra = open(os.path.join(H, "oct6_extra.css"), encoding="utf-8").read()
out = ("<title>Oct_6th_2026_lab_presentation</title>\n" + fonts + "\n<style>\n" + css + "\n" + extra + "</style>\n\n"
       '<div id="viewport">\n<div id="stage">\n\n' + slides + "\n</div>\n</div>\n\n" + chrome + "\n\n"
       '<script type="application/json" id="locus-rpc19-up">%%RPC19UP%%</script>\n'
       '<script type="application/json" id="locus-rpc19-down">%%RPC19DOWN%%</script>\n'
       '<script type="application/json" id="locus-fixXIVup">%%FIXXIVUP%%</script>\n'
       '<script type="application/json" id="locus-fixXIVdown">%%FIXXIVDOWN%%</script>\n'
       '<script type="application/json" id="locus-fixXII">%%FIXXII%%</script>\n' + js + "\n")
open(os.path.join(H, "deck_template.html"), "w", encoding="utf-8").write(out)
print("wrote deck_template.html", len(out))
