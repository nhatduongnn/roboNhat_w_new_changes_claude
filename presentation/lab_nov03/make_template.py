"""Build deck_template.html for the Nov 3 lab deck with the look and navigation of the Oct 27 deck.

Same recipe as presentation/lab_oct27/make_template.py (copied, not imported): copies from
presentation/talk.html (read only) the fonts, the <style> block, the notes/overview/HUD markup and the main
<script> (deck state, fit-to-stage, HUD, overview, keyboard/wheel/touch navigation). Changes to the copy:
NOTES from notes.json, SEC_NAMES, the talk's chart code dropped, title and HUD brand renamed, storage keys
"nov03.*". Keys are unchanged from Oct 27: h = hidden slides (and, new here, hidden content: any element
with class "hid-only" shows only while h is on), o = overview, n = notes, f = fullscreen.
Slides come from slides_fragment.html (written by make_fragment.py), extra CSS from ../lab_oct27/oct27_extra.css
plus nov03_extra.css.

    python make_fragment.py && python make_template.py
"""
import json, os, re
H = os.path.dirname(os.path.abspath(__file__))
talk = open(os.path.join(H, "..", "talk.html"), encoding="utf-8").read()
TITLE = "Nov 3 lab meeting"


def between(s, a, b, inclusive=False):
    i = s.index(a); j = s.index(b, i + len(a))
    return s[i:j + len(b)] if inclusive else s[i + len(a):j]


fonts = "\n".join(l for l in talk.splitlines()[:6] if l.startswith("<link"))
css = between(talk, "<style>\n", "\n</style>")
chrome = between(talk, '<div id="notes" hidden>', "</nav>", inclusive=True)
old = '<span class="brand">RoboCOP × Fiber-seq</span>'
assert old in chrome; chrome = chrome.replace(old, '<span class="brand">Lab meeting · 3 Nov 2026</span>')
old = '<span><kbd>h</kbd> include hidden + backup</span>'
assert old in chrome; chrome = chrome.replace(old, '<span><kbd>h</kbd> include hidden slides + content</span>')
js = talk[talk.index("<script>\n(function(){"):]
js = js[:js.index("</script>") + len("</script>")]

notes = json.load(open(os.path.join(H, "notes.json"), encoding="utf-8"))
js = re.sub(r"var NOTES = \{.*?\n\};", lambda m: "var NOTES = " + json.dumps(notes, ensure_ascii=False) + ";", js, count=1, flags=re.S)
js = re.sub(r"var SEC_NAMES = \[.*?\];", 'var SEC_NAMES = ["Title","Barcodes and read filters","Model decisions","Single-fiber update"];', js, count=1)
a = js.index("/* ---------------- charts"); b = js.index("/* ---------------- locus slides")
js = js[:a] + js[b:]
old = 'document.title = "RoboCOP Meets Fiber-seq";'
assert old in js; js = js.replace(old, 'document.title = "%s";' % TITLE)
js = js.replace('"rfs.showAll"', '"nov03.showAll"').replace('"rfs.notes"', '"nov03.notes"')
# h also reveals hidden content inside slides (.hid-only)
old = 'document.getElementById("btn-h").setAttribute("aria-pressed", showAll?"true":"false");'
assert js.count(old) == 1
js = js.replace(old, old + '\n  document.documentElement.classList.toggle("show-hidden", showAll);')

slides = open(os.path.join(H, "slides_fragment.html"), encoding="utf-8").read()
extra = open(os.path.join(H, "..", "lab_oct27", "oct27_extra.css"), encoding="utf-8").read()
extra += "\n" + open(os.path.join(H, "nov03_extra.css"), encoding="utf-8").read()
# click-to-enlarge for figure slides (.zoom-btn data-src): overlay outside #viewport, so the deck's click/wheel
# handlers never see it; a capturing keydown listener keeps arrows/space from paging the deck while it is open.
ZOOM_HTML = ('<div id="zoom" hidden role="dialog" aria-modal="true" aria-labelledby="zoom-t">'
             '<div class="zoom-bar"><span id="zoom-t"></span><button type="button" id="zoom-x">Close (Esc)</button></div>'
             '<div class="zoom-sc"><img id="zoom-img" alt=""></div></div>')
ZOOM_JS = """<script>
(function(){
  var z=document.getElementById("zoom"), im=document.getElementById("zoom-img"), t=document.getElementById("zoom-t"), last=null;
  function open(b){ last=b; im.src=b.getAttribute("data-src"); im.alt=b.getAttribute("data-title")||""; t.textContent=b.getAttribute("data-title")||""; z.hidden=false; document.getElementById("zoom-x").focus(); }
  function close(){ z.hidden=true; im.removeAttribute("src"); if(last) try{ last.focus(); }catch(e){} }
  document.addEventListener("click", function(e){ var b=e.target.closest(".zoom-btn"); if(b){ e.stopPropagation(); open(b); } }, true);
  document.getElementById("zoom-x").addEventListener("click", close);
  window.addEventListener("keydown", function(e){ if(z.hidden) return; e.stopPropagation(); if(e.key==="Escape"){ e.preventDefault(); close(); } }, true);
})();
</script>"""
out = ('<meta charset="utf-8">\n<title>%s</title>\n' % TITLE + fonts + "\n<style>\n" + css + "\n" + extra + "</style>\n\n"
       '<div id="viewport">\n<div id="stage">\n\n' + slides + "\n</div>\n</div>\n\n" + chrome + "\n" + ZOOM_HTML + "\n\n" + js + "\n" + ZOOM_JS + "\n")
assert "%%" not in out
open(os.path.join(H, "lab_nov03_deck.html"), "w", encoding="utf-8").write(out)
print("wrote lab_nov03_deck.html", len(out), "slides", out.count('<section class="slide'))
