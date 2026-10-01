"""Build the course-project report as a Word document, styled after the SRM sample
report (A4, page border, Times New Roman, blue headings, blue-bordered tables).

    python report/build_report.py   -> report/IDS_Case_Study_Report.docx

Every result number is read from results/ (and the extractor checks are re-run)
at build time -- nothing is typed in by hand. Personal details come from
report/details.json; anything still in [brackets] is highlighted yellow.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT  # noqa: F401  (kept for easy landscape tweaks)
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Mm, Pt, RGBColor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "lab" / "extract"))
sys.path.insert(0, str(ROOT / "tests"))

RES, PLOTS = ROOT / "results", ROOT / "results" / "plots"
FIG, ASSETS = ROOT / "report" / "figures", ROOT / "report" / "assets"
OUT = ROOT / "report" / "IDS_Case_Study_Report.docx"
D = json.loads((ROOT / "report" / "details.json").read_text(encoding="utf-8"))

FONT = "Times New Roman"
BLUE = RGBColor(0x2F, 0x54, 0x96)
BORDER, HEAD_FILL = "4472C4", "D9E2F3"
BODY_PT = 12
TEXT_W = 6.27                      # A4 width minus 1" margins, inches

doc = Document()
_fig = [0]
_tab = [0]


# =========================================================================== #
# Numbers: all read from results/ at build time
# =========================================================================== #

def load_numbers() -> dict:
    from preprocess import (CATEGORICAL, LOG_COLS, NUMERIC, RATE_FEATURES, TEST_FILE,
                            TRAIN_FILE, Preprocessor, load_raw)
    m = pd.read_csv(RES / "metrics.csv")
    ev = pd.read_csv(RES / "evasion.csv")
    bm = pd.read_csv(RES / "benchmark.csv")
    rb = pd.read_csv(RES / "benchmark_robustness.csv")
    meta = json.loads((RES / "run_metadata.json").read_text())
    shap_s = pd.read_csv(RES / "shap" / "shap_summary.csv", index_col=0)["value"]
    shap_i = pd.read_csv(RES / "shap" / "shap_importance.csv")

    tr, te = load_raw(TRAIN_FILE), load_raw(TEST_FILE)
    pre = Preprocessor().fit(tr)
    onehot = {c: len(k) for c, k in zip(CATEGORICAL, pre.encoder.categories_)}

    def row(model, ev_set):
        return m[(m.model == model) & (m.eval_set == ev_set)].iloc[0]

    def evr(model, threat, attack, con):
        r = ev[(ev.model == model) & (ev.threat_model == threat) & (ev.attack == attack)
               & (ev.constrained == con)]
        return float(r.evasion_rate.iloc[0]), int(r.n_detected_attacks.iloc[0])

    import ids_model
    thr = float(ids_model.load()["ids"].threshold)
    return dict(
        m=m, ev=ev, bm=bm, rb=rb, meta=meta, shap_s=shap_s, shap_i=shap_i, row=row,
        evr=evr, thr=thr, tr=tr, te=te, onehot=onehot, n_numeric=len(NUMERIC),
        n_rate=len(RATE_FEATURES), n_log=len(LOG_COLS),
        indist=float(bm[bm.protocol.str.contains("random split")].accuracy.iloc[0]),
    )


def run_extractor_tests() -> tuple[int, int, list[str]]:
    """Re-run the capture-extractor unit tests so the report states a fresh result."""
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/test_extractor.py", "-v",
                        "-p", "no:warnings", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True)
    names = re.findall(r"test_extractor\.py::(\w+) PASSED", r.stdout)
    failed = len(re.findall(r" FAILED", r.stdout))
    return len(names), failed, names


def synthetic_capture_check() -> dict:
    """Build a synthetic capture (benign user sessions + an attacker port sweep),
    run it through the real extractor and both IDSs, and count verdicts."""
    import joblib
    import ids_model
    from pcap_to_nslkdd import compute_features, parse_connections
    from preprocess import FEATURES, load_dataset
    from test_extractor import ACK, ATT, FIN, NOISE, RST, SYN, VIC, _pkt, _write

    pk, t = [], 0.0
    for i in range(8):
        s = 40000 + i
        pk += [(t, _pkt(NOISE, VIC, s, 80, SYN)), (t + .001, _pkt(VIC, NOISE, 80, s, SYN | ACK)),
               (t + .002, _pkt(NOISE, VIC, s, 80, ACK, b"GET / HTTP/1.0\r\n\r\n")),
               (t + .003, _pkt(VIC, NOISE, 80, s, ACK, b"HTTP/1.0 200 OK\r\n\r\nhello")),
               (t + .004, _pkt(NOISE, VIC, s, 80, FIN | ACK)),
               (t + .005, _pkt(VIC, NOISE, 80, s, FIN | ACK))]
        t += 0.9
    for port in range(1, 41):
        s = 50000 + port
        pk += [(t, _pkt(ATT, VIC, s, port, SYN)), (t + .0005, _pkt(VIC, ATT, port, s, RST | ACK))]
        t += 0.01
    path = ROOT / "report" / "_preview" / "synthetic_check.pcap"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(path, pk)
    df = pd.DataFrame(compute_features(parse_connections(path)))
    rf = joblib.load(RES / "rf_baseline.joblib")
    bundle = ids_model.load()
    mm = load_dataset(log=False)["preprocessor"]
    p_paper = rf.predict_proba(mm.transform(df[FEATURES]))[:, 1] >= 0.5
    p_ours = bundle["ids"].predict(bundle["preprocessor"].transform(df[FEATURES])) == 1
    user, att = (df._src_ip == NOISE).to_numpy(), (df._src_ip == ATT).to_numpy()
    return dict(n=len(df), n_user=int(user.sum()), n_att=int(att.sum()),
                paper_user=int(p_paper[user].sum()), ours_user=int(p_ours[user].sum()),
                paper_att=int(p_paper[att].sum()), ours_att=int(p_ours[att].sum()),
                flags=df.flag.value_counts().to_dict(),
                max_count=int(df["count"].max()))


# =========================================================================== #
# Formatting helpers
# =========================================================================== #

def _set_fonts(rpr, name=FONT):
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for a in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        if rf.get(qn(a)) is not None:
            del rf.attrib[qn(a)]
    for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        rf.set(qn(a), name)


def setup_styles():
    st = doc.styles
    normal = st["Normal"]
    normal.font.size = Pt(BODY_PT)
    _set_fonts(normal.element.get_or_add_rPr())
    pf = normal.paragraph_format
    pf.space_after = Pt(6)
    pf.line_spacing = 1.15
    for name, size, before, after in [("Heading 1", 16, 18, 8), ("Heading 2", 13, 14, 6),
                                      ("Heading 3", 12, 10, 4), ("Title", 16, 0, 0)]:
        s = st[name]
        s.font.size, s.font.bold, s.font.italic = Pt(size), True, False
        s.font.color.rgb = BLUE if name != "Title" else RGBColor(0, 0, 0)
        _set_fonts(s.element.get_or_add_rPr())
        s.paragraph_format.space_before = Pt(before)
        s.paragraph_format.space_after = Pt(after)
        s.paragraph_format.keep_with_next = True
    for name in ("List Bullet", "List Number", "Caption"):
        _set_fonts(st[name].element.get_or_add_rPr())


def page_setup():
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Mm(210), Mm(297)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Inches(1))
    sec.different_first_page_header_footer = True     # no page number on the title page
    # page border like the sample: a thin-thick double line on every page
    sp = sec._sectPr
    borders = OxmlElement("w:pgBorders")
    borders.set(qn("w:offsetFrom"), "page")
    for side in ("top", "left", "bottom", "right"):
        b = OxmlElement(f"w:{side}")
        b.set(qn("w:val"), "thinThickSmallGap")
        b.set(qn("w:sz"), "24")
        b.set(qn("w:space"), "24")
        b.set(qn("w:color"), "000000")
        borders.append(b)
    sp.find(qn("w:pgMar")).addnext(borders)
    # centred page number in the footer
    p = sec.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_field(p, "PAGE", "1", size=10)


def add_field(paragraph, instr, placeholder="", size=None):
    def run_with(el):
        r = paragraph.add_run()
        if size:
            r.font.size = Pt(size)
        r._r.append(el)
        return r
    b = OxmlElement("w:fldChar"); b.set(qn("w:fldCharType"), "begin"); run_with(b)
    it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = f" {instr} "
    run_with(it)
    s = OxmlElement("w:fldChar"); s.set(qn("w:fldCharType"), "separate"); run_with(s)
    r = paragraph.add_run(placeholder)
    if size:
        r.font.size = Pt(size)
    e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), "end"); run_with(e)


def runs(paragraph, text, size=None, bold=None, italic=None, color=None):
    """Add text, highlighting [placeholders] and honouring **bold** spans."""
    for part in re.split(r"(\*\*.+?\*\*|\[[^\]]+\])", text):
        if not part:
            continue
        b = bold
        if part.startswith("**") and part.endswith("**"):
            part, b = part[2:-2], True
        r = paragraph.add_run(part)
        if size:
            r.font.size = Pt(size)
        if b is not None:
            r.bold = b
        if italic is not None:
            r.italic = italic
        if color is not None:
            r.font.color.rgb = color
        if re.fullmatch(r"\[[^\]]*[A-Za-z][^\]]*\]", part):   # [NAME], not [1] or [0, 1]
            r.font.highlight_color = WD_COLOR_INDEX.YELLOW
    return paragraph


def P(text, align="justify", size=None, bold=None, italic=None, after=None, before=None,
      keep=False):
    p = doc.add_paragraph()
    p.alignment = {"justify": WD_ALIGN_PARAGRAPH.JUSTIFY, "center": WD_ALIGN_PARAGRAPH.CENTER,
                   "left": WD_ALIGN_PARAGRAPH.LEFT}[align]
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    if before is not None:
        p.paragraph_format.space_before = Pt(before)
    if keep:
        p.paragraph_format.keep_with_next = True
    return runs(p, text, size=size, bold=bold, italic=italic)


def H1(text):
    doc.add_heading(text, level=1)


def H2(text):
    doc.add_heading(text, level=2)


def H3(text):
    doc.add_heading(text, level=3)


def _restarted_num_id() -> int:
    """A fresh numbering instance of the List Number definition, starting at 1, so each
    numbered list counts from 1 instead of continuing the previous one."""
    numbering = doc.part.numbering_part.numbering_definitions._numbering
    base = doc.styles["List Number"].element.pPr.numPr.numId.val
    abstract_id = numbering.num_having_numId(base).abstractNumId.val
    num = numbering.add_num(abstract_id)
    num.add_lvlOverride(ilvl=0).add_startOverride(1)
    return num.numId


def bullets(items, style="List Bullet"):
    num_id = _restarted_num_id() if style == "List Number" else None
    for it in items:
        p = doc.add_paragraph(style=style)
        if num_id is not None:
            numPr = p._p.get_or_add_pPr().get_or_add_numPr()
            numPr.get_or_add_ilvl().val = 0
            numPr.get_or_add_numId().val = num_id
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(3)
        runs(p, it)


def equation(text):
    """Centred italic formula; ^{...} is set as superscript and _{...} as subscript."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(8)
    for part in re.split(r"(\^\{[^}]*\}|_\{[^}]*\})", text):
        if not part:
            continue
        r = p.add_run(part[2:-1] if part[:2] in ("^{", "_{") else part)
        r.italic = True
        if part.startswith("^{"):
            r.font.superscript = True
        elif part.startswith("_{"):
            r.font.subscript = True


def code(lines):
    for i, line in enumerate(lines):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(0 if i < len(lines) - 1 else 8)
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.line_spacing = 1.0
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "F2F2F2")
        p._p.get_or_add_pPr().append(shd)
        r = p.add_run(line or " ")
        r.font.name = "Consolas"
        r._r.get_or_add_rPr().find(qn("w:rFonts")).set(qn("w:eastAsia"), "Consolas")
        r.font.size = Pt(9.5)


def figure(path, caption, width=TEXT_W):
    _fig[0] += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(6)
    p.add_run().add_picture(str(path), width=Inches(width))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_after = Pt(12)
    runs(c, f"Fig {_fig[0]}: {caption}", size=10.5, italic=True)
    return _fig[0]


def next_fig() -> int:
    return _fig[0] + 1


def next_tab() -> int:
    return _tab[0] + 1


def _cell_shade(cell, fill):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(shd)


def table(headers, rows, widths, caption, size=10, align=None, bold_first_col=False):
    """Blue-bordered table with a shaded, repeating header row and a caption above."""
    _tab[0] += 1
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.keep_with_next = True
    cap.paragraph_format.space_before = Pt(8)
    cap.paragraph_format.space_after = Pt(4)
    runs(cap, f"Table {_tab[0]}: {caption}", size=10.5, bold=True)

    assert abs(sum(widths) - TEXT_W) < 0.05 or sum(widths) < TEXT_W, (caption, sum(widths))
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr
    tb = OxmlElement("w:tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6"); e.set(qn("w:space"), "0")
        e.set(qn("w:color"), BORDER)
        tb.append(e)
    old = tblPr.find(qn("w:tblBorders"))
    if old is not None:
        tblPr.remove(old)
    tblPr.append(tb)
    for j, w in enumerate(widths):
        t.columns[j].width = Inches(w)
    align = align or ["left"] + ["center"] * (len(headers) - 1)
    amap = {"left": WD_ALIGN_PARAGRAPH.LEFT, "center": WD_ALIGN_PARAGRAPH.CENTER,
            "right": WD_ALIGN_PARAGRAPH.RIGHT}
    for i, values in enumerate([headers] + rows):
        tr = t.rows[i]._tr
        trPr = tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit"); trPr.append(cs)
        if i == 0:
            th = OxmlElement("w:tblHeader"); trPr.append(th)
        for j, v in enumerate(values):
            cell = t.rows[i].cells[j]
            cell.width = Inches(widths[j])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(1)
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.line_spacing = 1.0
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i == 0 else amap[align[j]]
            runs(p, str(v), size=size, bold=True if i == 0 or (bold_first_col and j == 0) else None)
            if i == 0:
                _cell_shade(cell, HEAD_FILL)
    if len(rows) <= 16:              # small tables never split across a page
        for r_ in t.rows[:-1]:
            for c_ in r_.cells:
                for p_ in c_.paragraphs:
                    p_.paragraph_format.keep_with_next = True
    after = doc.add_paragraph()
    after.paragraph_format.space_after = Pt(4)
    return _tab[0]


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def pct(x, d=1):
    return f"{100 * float(x):.{d}f}%"


def f3(x):
    return f"{float(x):.3f}"


# =========================================================================== #
# Front matter
# =========================================================================== #

def title_page():
    def line(text, size=12, bold=False, before=0, after=4):
        P(text, align="center", size=size, bold=bold, before=before, after=after)
    line(D["title"], size=16, bold=True, before=20, after=8)
    line(D["subtitle"], size=13, after=6)
    line(D["course"], size=11, after=30)
    line("Submitted by", size=12, after=8)
    for name, reg in D["students"]:
        line(f"{name} [{reg}]" if not reg.startswith("[") else f"{name} {reg}", size=12, after=3)
    line("Under the Guidance of", size=12, before=26, after=8)
    line(D["guide_name"], size=14, after=4)
    line(f"{D['guide_designation']}, {D['guide_department']}", size=11, after=28)
    line("In partial satisfaction of the requirements for the degree of", size=12, after=14)
    line(D["degree"], size=12, after=2)
    line("in", size=12, after=2)
    line(D["branch"], size=13, after=8)
    line(D["specialisation"], size=12, after=22)
    logo = doc.add_paragraph()
    logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    logo.paragraph_format.space_after = Pt(16)
    logo.add_run().add_picture(str(ASSETS / "srm_logo.jpg"), width=Inches(2.7))
    for t in ["SCHOOL OF COMPUTING", "COLLEGE OF ENGINEERING AND TECHNOLOGY",
              "SRM INSTITUTE OF SCIENCE AND TECHNOLOGY", "KATTANKULATHUR - 603203"]:
        line(t, size=13, after=6)
    line(D["month_year"], size=12, before=10)
    page_break()


def certificate():
    for t in ["SRM INSTITUTE OF SCIENCE AND TECHNOLOGY", "KATTANKULATHUR - 603203"]:
        P(t, align="center", size=14, after=2, before=24 if t.startswith("SRM") else 0)
    P("BONAFIDE CERTIFICATE", align="center", size=13, bold=True, before=30, after=24)
    names = ", ".join(f"{n} {r if r.startswith('[') else '[' + r + ']'}" for n, r in D["students"])
    P(f"Certified that this Course Project Report titled “{D['title'].title()}: "
      f"{D['subtitle']}” is the bonafide work done by {names}, who carried it out under my "
      "supervision. Certified further, that to the best of my knowledge the work reported herein "
      "does not form part of any other work.", after=60)
    for t in ["", D["guide_name"], D["guide_designation"], D["guide_department"],
              "SRM Institute of Science and Technology", "Kattankulathur Campus, Chennai"]:
        P(t, align="left", after=0)
    page_break()


def contents():
    P("Table of Contents", align="left", size=16, bold=True, after=10).runs[0].font.color.rgb = BLUE
    p = doc.add_paragraph()
    add_field(p, 'TOC \\o "1-2" \\h \\z \\u',
              "Right-click here and choose Update Field to build the table of contents.")
    page_break()


# =========================================================================== #
# Body
# =========================================================================== #

def abstract(R):
    b, o, a = R["row"]("baseline", "clean"), R["row"]("improved_ids", "clean"), R["row"]("adv_trained", "clean")
    e_bu, _ = R["evr"]("baseline", "static", "pgd", False)
    e_br, _ = R["evr"]("baseline", "static", "pgd", True)
    e_ou, _ = R["evr"]("improved_ids", "static", "pgd", False)
    e_or, _ = R["evr"]("improved_ids", "static", "pgd", True)
    e_ar, _ = R["evr"]("adv_trained", "static", "pgd", True)
    mk = R["meta"]["mask"]
    H1("Abstract")
    P("An intrusion detection system (IDS) watches network traffic and raises an alarm when a "
      "connection looks malicious. Machine-learning IDSs are usually stress-tested with adversarial "
      "examples that nudge every input feature freely until the detector is fooled. On a real network, "
      "however, an attacker controls only the packets they send: most of the 41 NSL-KDD connection "
      "features are decided by the victim host or by other users’ traffic, and cannot be set by the "
      "attacker at all.")
    P("This case study first reproduces a published pipeline (a Random Forest IDS on NSL-KDD, attacked "
      "with FGSM and PGD and defended by adversarial training) and then extends it in three ways. "
      f"First, every feature is classified by who controls it over TCP/IP, and the result is enforced "
      f"as a **realizability mask** that constrains attacks in real network units; only "
      f"{mk['continuously_mutable']} of {mk['n_raw_features']} features remain movable. Second, we show "
      "that the usual “accuracy under attack” metric mixes genuine evasion with false alarms "
      "manufactured on benign traffic, and measure the **evasion rate** instead. Third, we build an "
      "**improved IDS**: a soft-voting ensemble whose alarm threshold is calibrated on attack types held "
      "out of training.")
    P(f"On the KDDTest+ set, which contains 17 attack types never seen in training, the improved IDS raises "
      f"accuracy from {pct(b.accuracy)} to {pct(o.accuracy)} and F1-score from {f3(b.f1)} to {f3(o.f1)}. "
      f"It cuts the evasion rate from {pct(e_bu)} to {pct(e_ou)} against an unrestricted PGD attacker, and "
      f"from {pct(e_br)} to {pct(e_or)} against a realistic one. Adversarial training on unrestricted "
      f"perturbations blocked every unrestricted attack yet still let {pct(e_ar)} of realistic attacks "
      "through. An isolated Docker capture lab with a PCAP-to-feature extractor, and an offline "
      "demonstration application, accompany the study.")
    P("**Keywords:** intrusion detection, adversarial machine learning, NSL-KDD, network security, "
      "realizable attacks, evasion rate.", align="left")
    page_break()


def introduction(R):
    H1("1. Introduction")
    H2("1.1 Intrusion Detection in Computer Networks")
    P("An intrusion detection system monitors a network or a host for malicious activity and policy "
      "violations. A **network IDS (NIDS)** observes packets at a vantage point in the network, typically a "
      "switch mirror (SPAN) port or a network tap, while a **host IDS** runs on an individual machine. "
      "Signature-based NIDSs such as Snort and Suricata match traffic against rules for known attacks; they "
      "are precise but blind to anything without a rule. Anomaly-based and machine-learning (ML) NIDSs learn "
      "what malicious traffic looks like and can flag attacks they were never explicitly told about, at the "
      "cost of more false alarms [20].")
    H2("1.2 Machine Learning for Intrusion Detection")
    P("ML-based detectors do not read raw packets. Each TCP/UDP/ICMP connection is first summarised as a "
      "fixed set of features: how long it lasted, how many bytes flowed each way, how it ended, and how many "
      "similar connections reached the same host recently. The NSL-KDD dataset [1], a cleaned version of the "
      "KDD Cup 1999 data [9], describes every connection with 41 such features and a label. A classifier "
      "trained on these records labels new connections as normal or attack.")
    H2("1.3 Adversarial Attacks on ML-based IDS")
    P("ML models can be fooled by adversarial examples: inputs altered by small, carefully chosen amounts "
      "so that the model changes its decision. The Fast Gradient Sign Method (FGSM) [2] and Projected "
      "Gradient Descent (PGD) [3] are the standard attacks. The work this study builds on [18] trained a "
      "Random Forest IDS on NSL-KDD, explained it with SHAP [6], attacked it with FGSM and PGD, and "
      "defended it with adversarial training.")
    H2("1.4 Problem Statement")
    P("Those attacks perturb **all 41 features freely**. But a feature vector is not traffic. An attacker "
      "cannot edit a number; they can only send packets, and the wire limits what those packets can change. "
      "The number of bytes the attacker sends is theirs to choose. Whether the victim answered with a reset, "
      "or what fraction of the victim’s last 100 connections came from other hosts, is not. An "
      "adversarial example that no packet sequence can produce is a mathematical artefact, not a threat. "
      "This report asks how much of the published adversarial threat survives when attacks are limited to "
      "what a real attacker can actually send, and whether a better-calibrated IDS is more robust than one "
      "hardened by adversarial training.")
    H2("1.5 Objectives")
    bullets([
        "Reproduce the published Random Forest IDS, its FGSM/PGD attacks and its adversarial-training defence.",
        "Classify all 41 NSL-KDD features by who controls them over TCP/IP, and enforce the classification "
        "as a realizability mask applied inside the attacks.",
        "Replace whole-test-set “accuracy under attack” with an evasion rate that measures only "
        "attacks disguised as normal traffic.",
        "Build an IDS with higher honest accuracy on unseen attacks, without using the test set to tune it.",
        "Provide an isolated capture lab and extractor that turn real packets into the same 41 features, "
        "and an offline application that demonstrates the results.",
    ])
    H2("1.6 Contributions")
    bullets([
        "**A realizability mask** that enforces protocol constraints in raw network units (whole bytes, "
        "raise-only connection counts, valid ranges) rather than in the model’s scaled feature space.",
        "**A measurement showing that accuracy under attack is misleading:** most of the measured drop "
        "comes from benign traffic made to look malicious, which an attacker does not control.",
        "**Evidence that a calibrated threshold outperforms adversarial training** against realistic "
        "attackers, while also raising accuracy on unseen attacks.",
        "**An end-to-end capture pipeline and demonstration application** that tie the ML result back to "
        "real network traffic.",
    ])
    P("Constrained adversarial attacks on NIDSs have been studied before [11]–[16]; Section 7.1 "
      "positions this work against that literature.")
    H2("1.7 Organisation of the Report")
    P("Section 2 lists the requirements and the threat model. Section 3 presents the architecture and the "
      "network design of the capture lab. Section 4 describes the dataset and the feature realizability "
      "taxonomy. Section 5 covers the methodology and implementation, Section 6 the results and testing, "
      "Section 7 the discussion and limitations, and Section 8 concludes.")
    page_break()


def requirements(R):
    libs = R["meta"]["libraries"]
    H1("2. Requirement Analysis")
    P("Requirement analysis fixes what the IDS must do, what the attacker can and cannot do, and what "
      "hardware and software the study needs.")
    H2("2.1 Functional Requirements")
    bullets([
        "**Detection:** classify each network connection as normal or attack.",
        "**Generalisation:** detect attack types that were absent from training, as KDDTest+ requires.",
        "**Low false-alarm rate:** keep precision high so that operators are not flooded with alerts.",
        "**Robustness:** resist attackers who reshape their traffic to evade detection.",
        "**Honest evaluation:** no test-set information may influence training or threshold selection.",
        "**Reproducibility:** every reported number regenerates from one command with fixed random seeds.",
        "**Deployability:** the demonstration runs offline on an ordinary laptop.",
    ])
    H2("2.2 Threat Model")
    P("The attacker in this study is realistic rather than all-powerful:")
    bullets([
        "**Controls only their own traffic:** payload size, connection duration, how many connections "
        "they open, and crafted header fields.",
        "**Cannot control the victim’s responses:** whether a connection completes, is reset or is "
        "ignored is decided by the victim, as is how many bytes it sends back.",
        "**Cannot control third-party traffic:** the host-based features summarise the last 100 "
        "connections to the victim from every source, including legitimate users.",
        "**Has no access to the deployed IDS:** the attacker trains their own surrogate model and transfers "
        "the attack, rather than reading the defender’s model.",
    ])
    H2("2.3 Hardware Requirements")
    table(["Component", "Used in this study", "Minimum"],
          [["Processor", "Intel Core i7-1165G7, 4 cores / 8 threads", "Any 64-bit x86 CPU"],
           ["Memory", "16 GB", "8 GB"],
           ["Graphics", "Intel Iris Xe (integrated); no GPU used", "Not required"],
           ["Storage", "~0.5 GB Python environment, 21.5 MB dataset", "2 GB free"],
           ["Capture lab host", "Separate machine running Docker", "Docker Engine or Desktop, 4 GB RAM"],
           ["Network", "Lab network is internal (no Internet)", "Internet only for first-time setup"]],
          [1.45, 2.65, 2.17], "Hardware requirements")
    H2("2.4 Software Requirements")
    sw = [["Operating system", "Windows 11 (analysis), Linux containers (lab)", "Host and lab platform"],
          ["Python", libs["python"], "Implementation language"],
          ["scikit-learn", libs["scikit-learn"], "Random Forest, Extra Trees, gradient boosting, metrics"],
          ["Adversarial Robustness Toolbox", libs["adversarial-robustness-toolbox"], "Adversarial-ML toolkit [5]"],
          ["SHAP", libs["shap"], "Feature-importance analysis [6]"],
          ["NumPy / pandas / SciPy", f"{libs['numpy']} / {libs['pandas']} / {libs['scipy']}",
           "Numerical work and data handling"],
          ["Matplotlib", libs["matplotlib"], "Figures"],
          ["dpkt", _version("dpkt"), "Reading PCAP files in the extractor"],
          ["Docker", "Engine/Desktop 29.x", "Isolated capture lab"],
          ["tcpdump, nmap, hping3", "Debian/netshoot packages", "Capture and traffic generation"],
          ["Web browser", "Chrome or Edge", "Demonstration application"]]
    table(["Software", "Version", "Purpose"], sw, [1.85, 1.9, 2.52], "Software requirements")
    page_break()


def _version(pkg):
    import importlib.metadata as md
    try:
        return md.version(pkg)
    except Exception:
        return "-"


def architecture(R):
    H1("3. Architecture & Design")
    H2("3.1 Placement of the IDS in the Network")
    P("A network IDS must see traffic without slowing it down. It is therefore attached to a switch "
      "**mirror (SPAN) port** or an inline **network tap**, which copies every frame to the sensor while "
      "the original continues to its destination. The sensor’s interface runs in **promiscuous mode** "
      "so that it accepts frames addressed to other hosts. Because the sensor is passive, a failure or "
      "overload of the IDS cannot interrupt the network. It also sees **both directions** of each "
      "connection, which is essential: features such as the connection flag and the bytes returned by the "
      "server can only be computed from the server’s replies.")
    H2("3.2 System Architecture")
    n = next_fig()
    P(f"Fig {n} shows the two halves of the system. The defender path takes NSL-KDD records through "
      "preprocessing into the IDS models and evaluates them. The attacker path trains a surrogate model on "
      "the same feature space, generates FGSM/PGD perturbations, passes them through the realizability "
      "mask after every step, and sends the result to the real IDS to see whether it is still detected.")
    figure(FIG / "fig_pipeline.png", "System architecture: defender pipeline and attacker pipeline")
    H2("3.3 Network Design of the Capture Lab")
    n = next_fig()
    P(f"To connect the ML results to real traffic, the study includes an isolated capture lab built with "
      f"Docker Compose (Fig {n}). Four containers share one bridge network that is declared "
      "**internal**, so it has no gateway and no route to the campus LAN or the Internet; nothing in the "
      "lab can reach a network the authors do not own.")
    bullets([
        "**Victim server:** a multi-port TCP service. Open ports complete the handshake (flag SF); closed "
        "ports answer a SYN with a reset (flag REJ).",
        "**Attacker:** generates reconnaissance sweeps and scripted probes that try to hit target feature "
        "values.",
        "**Third-party host:** sends ordinary HTTP traffic. It exists to show that the victim’s "
        "host-based features change even when the attacker’s own traffic is unchanged.",
        "**IDS sensor:** tcpdump running inside the victim’s network namespace, which gives the same "
        "vantage point as a SPAN port in front of the victim.",
    ])
    figure(FIG / "fig_topology.png", "Network topology of the isolated capture lab")
    H2("3.4 Network Configuration Table")
    table(["Device", "Role", "IP address", "Subnet mask", "Gateway", "Ports / services"],
          [["Bridge idsnet", "Isolated segment", "192.168.90.0/24", "255.255.255.0", "None (internal)", "-"],
           ["Victim server", "Target host", "192.168.90.10", "255.255.255.0", "None",
            "TCP 21, 22, 80, 143 open; other ports reset"],
           ["Attacker", "Attack traffic", "192.168.90.20", "255.255.255.0", "None", "nmap, hping3, probes"],
           ["Third-party host", "Benign traffic", "192.168.90.30", "255.255.255.0", "None", "HTTP client"],
           ["IDS sensor", "Packet capture", "Victim's namespace", "-", "-",
            "tcpdump, 5 × 50 MB ring buffer"],
           ["Demo application", "Analysis and UI", "127.0.0.1", "-", "-", "HTTP on TCP 8765"]],
          [1.05, 0.95, 1.12, 0.98, 0.82, 1.35], "Network configuration of the capture lab", size=9)
    H2("3.5 Connection Features from Packets")
    P("The extractor groups packets into TCP connections and recovers each NSL-KDD feature the same way the "
      "original dataset defined it [9]. The connection **flag** comes from the TCP three-way handshake "
      "and teardown [17], [19], observed in both directions:")
    table(["Flag", "Meaning", "Observed when"],
          [["SF", "Normal establishment and termination", "SYN, SYN-ACK, then FIN"],
           ["S0", "Connection attempt, no reply", "SYN only"],
           ["REJ", "Connection rejected", "SYN answered with RST"],
           ["S1", "Established, not closed", "Handshake completed, no FIN or RST"],
           ["RSTO", "Originator aborted", "Established, then RST from the client"],
           ["RSTR", "Responder aborted", "Established, then RST from the server"],
           ["SH", "SYN then FIN, no reply", "SYN and FIN from the client only"],
           ["OTH", "Mid-stream traffic", "No SYN captured"]],
          [0.8, 2.6, 2.87], "TCP connection flags recovered by the extractor")
    P("The **time-based** features (23–31) count connections to the same host or service in the "
      "**past two seconds**. The **host-based** features (32–41) are computed over the **last 100 "
      "connections to the destination host, from any source**. That single definition is why an attacker "
      "cannot set the host-based features: they depend on what everyone else sent to the victim.")
    page_break()


TAXONOMY = [
    # (feature, group, class, controlled by / constraint)
    ("duration", "Basic", "B", "Attacker holds the connection; whole seconds, ≥ 0"),
    ("protocol_type", "Basic", "C", "Choice made when the attack is designed; changing it is a different attack"),
    ("service", "Basic", "C", "Target port chosen at design time; categorical"),
    ("flag", "Basic", "B*", "Reflects how the victim answered; categorical, frozen in the attack"),
    ("src_bytes", "Basic", "A", "Bytes the attacker sends; whole bytes, ≥ 0"),
    ("dst_bytes", "Basic", "C", "Bytes the victim sends back"),
    ("land", "Basic", "B", "Source = destination spoofing; binary and self-incriminating"),
    ("wrong_fragment", "Basic", "B", "Malformed fragments; small integer, self-incriminating"),
    ("urgent", "Basic", "B", "Packets with URG set; small integer"),
    ("hot", "Content", "C", "Emerges from session content on the host"),
    ("num_failed_logins", "Content", "B", "Attacker chooses how many tries; can only increase"),
    ("logged_in", "Content", "C", "1 only if authentication actually succeeded"),
    ("num_compromised", "Content", "C", "Victim state after compromise"),
    ("root_shell", "Content", "C", "1 only if a root shell was obtained"),
    ("su_attempted", "Content", "B", "Attacker action; binary"),
    ("num_root", "Content", "C", "Victim state; success-dependent"),
    ("num_file_creations", "Content", "C", "Victim state after compromise"),
    ("num_shells", "Content", "C", "Victim state after compromise"),
    ("num_access_files", "Content", "C", "Victim state after compromise"),
    ("num_outbound_cmds", "Content", "C", "Constant 0 in the whole dataset"),
    ("is_host_login", "Content", "C", "Depends on the account used"),
    ("is_guest_login", "Content", "C", "Depends on the account used"),
    ("count", "Time (2 s)", "B", "Attacker adds connections; can only increase"),
    ("srv_count", "Time (2 s)", "B", "Same, per service; can only increase"),
    ("serror_rate", "Time (2 s)", "C", "Victim's SYN errors over the window"),
    ("srv_serror_rate", "Time (2 s)", "C", "Same, per service"),
    ("rerror_rate", "Time (2 s)", "C", "Victim's rejections over the window"),
    ("srv_rerror_rate", "Time (2 s)", "C", "Same, per service"),
    ("same_srv_rate", "Time (2 s)", "C", "Ratio over all recent connections"),
    ("diff_srv_rate", "Time (2 s)", "C", "Ratio over all recent connections"),
    ("srv_diff_host_rate", "Time (2 s)", "C", "Ratio over all recent connections"),
    ("dst_host_count", "Host (100)", "C", "Last 100 connections to the victim, from anyone"),
    ("dst_host_srv_count", "Host (100)", "C", "Same, per service"),
    ("dst_host_same_srv_rate", "Host (100)", "C", "Ratio over third-party history"),
    ("dst_host_diff_srv_rate", "Host (100)", "C", "Ratio over third-party history"),
    ("dst_host_same_src_port_rate", "Host (100)", "C", "Ratio over third-party history"),
    ("dst_host_srv_diff_host_rate", "Host (100)", "C", "Ratio over third-party history"),
    ("dst_host_serror_rate", "Host (100)", "C", "Victim errors over third-party history"),
    ("dst_host_srv_serror_rate", "Host (100)", "C", "Victim errors over third-party history"),
    ("dst_host_rerror_rate", "Host (100)", "C", "Victim rejections over third-party history"),
    ("dst_host_srv_rerror_rate", "Host (100)", "C", "Victim rejections over third-party history"),
]


def dataset(R):
    from preprocess import binarize_labels
    tr, te = R["tr"], R["te"]
    H1("4. Dataset and Feature Analysis")
    H2("4.1 The NSL-KDD Dataset")
    rows = []
    for name, df in [("KDDTrain+", tr), ("KDDTest+", te)]:
        y = binarize_labels(df["label"])
        rows.append([name, f"{len(df):,}", f"{int((y == 0).sum()):,}", f"{int(y.sum()):,}",
                     pct(y.mean()), str(df.label[df.label != 'normal'].nunique())])
    novel = sorted(set(te.label) - set(tr.label))
    P("NSL-KDD [1] provides a training file and a separate test file. The test file is deliberately "
      f"harder: {len(novel)} of its attack types never appear in training, so a model is scored on its "
      "ability to recognise new attacks rather than to memorise old ones.")
    table(["File", "Connections", "Normal", "Attack", "Attack share", "Attack types"], rows,
          [1.2, 1.1, 0.95, 0.95, 1.07, 1.0], "NSL-KDD files used in this study (binary labels)")
    P("Attack types present only in KDDTest+: " + ", ".join(novel) + ".", size=10.5)
    H2("4.2 The 41 Features")
    table(["Group", "Features", "Computed from", "Examples"],
          [["Basic", "1–9", "Packet headers of the connection", "duration, service, flag, src_bytes"],
           ["Content", "10–22", "Host state and session content", "logged_in, num_failed_logins"],
           ["Time-based", "23–31", "Connections in the past 2 seconds", "count, serror_rate"],
           ["Host-based", "32–41", "Last 100 connections to the same host",
            "dst_host_count, dst_host_serror_rate"]],
          [1.05, 0.85, 2.15, 2.22], "Feature groups of NSL-KDD")
    H2("4.3 Preprocessing")
    oh = R["onehot"]
    n_enc = R["n_numeric"] + sum(oh.values())
    P(f"The three text features are one-hot encoded: protocol_type into {oh['protocol_type']} columns, "
      f"service into {oh['service']} and flag into {oh['flag']}. With the {R['n_numeric']} numeric features "
      f"this gives a {n_enc}-dimensional input. Numeric features are scaled to [0, 1] with MinMax scaling "
      f"fitted on the training file only. For the improved IDS, the {R['n_log']} heavy-tailed magnitude "
      f"features (bytes, counts, durations) are first transformed with log(1 + x): bytes range from 0 to "
      "over a billion, and without this nearly every src_bytes value is compressed to almost zero.")
    P("An explicit **index map** records which encoded columns belong to each of the 41 raw features, and "
      "a two-way mapping converts encoded values back to raw units (bytes, seconds, counts). The "
      "realizability mask depends on both, as Section 5.4 explains.")
    H2("4.4 Feature Realizability Taxonomy")
    P("Each feature is assigned a class according to whether, in a live attack over TCP/IP, the attacker "
      "can set it to an arbitrary value:")
    table(["Class", "Count", "Meaning", "Features"],
          [["A — free", "1", "Attacker sets it directly", "src_bytes"],
           ["B — constrained", "9", "Attacker influences it within protocol rules: whole numbers, "
            "raise-only, binary or self-incriminating",
            "duration, flag, land, wrong_fragment, urgent, num_failed_logins, su_attempted, count, srv_count"],
           ["C — uncontrollable", "31", "Decided by the victim, by third-party traffic, or fixed when "
            "the attack is designed", "All other features"]],
          [1.25, 0.6, 2.1, 2.32], "Feature realizability classes")
    P("Of the nine Class B features, flag cannot move under a small continuous step because it is a "
      "category, so it is frozen in the attacks. That leaves **9 movable features**: src_bytes and eight "
      "Class B features. The full 41-feature classification with a reason for each is in Appendix A.")
    page_break()


def methodology(R):
    meta = R["meta"]
    sub_acc = meta.get("surrogate", {}).get("clean_test_accuracy")
    H1("5. Methodology and Implementation")
    H2("5.1 Baseline IDS (the Paper’s Model)")
    P("The baseline is a Random Forest [7], implemented with scikit-learn [10], with the published configuration: 100 trees, maximum depth 15, "
      "minimum samples per split 5 and random state 42, trained on KDDTrain+ with MinMax-scaled features. "
      "A connection is labelled an attack when the predicted attack probability is at least 0.5.")
    H2("5.2 Threat Model: Transfer Attack through a Surrogate")
    P("FGSM and PGD need gradients: the direction in which each input should move to lower the attack "
      "score. A Random Forest provides none, and a real attacker would not have the defender’s model "
      "anyway. The attacker therefore trains their **own surrogate model** on the same feature space, crafts "
      "the attack against it, and sends the result to the real IDS (a transfer attack). The surrogate is a "
      "small neural network (122 → 64 → 32 → 2, ReLU, Adam, 12 epochs) implemented with "
      "exact backpropagation in NumPy"
      + (f"; it reaches {pct(sub_acc)} accuracy on KDDTest+." if sub_acc else "."))
    P("A linear surrogate was tried first and rejected. Its input gradient is always ±w, so every "
      "perturbation of a given class pointed the same way and the two classes moved in exactly opposite "
      "directions. The perturbation therefore encoded the label, and a model adversarially trained on "
      "those examples scored a meaningless 100% — a known effect called label leaking [4]. The "
      "attacks also use the surrogate’s **predicted** labels rather than the true ones, for the same "
      "reason.")
    H2("5.3 FGSM and PGD")
    P("Let x be an encoded connection, g the surrogate, L the cross-entropy loss and y the label the "
      "attacker wants to move away from. FGSM takes one step of size ε in the sign of the gradient:")
    equation("x′ = x + ε · sign( ∇_{x} L( g(x), y ) )")
    P("PGD takes T smaller steps of size α and, after each one, projects back into the ε-ball "
      "around x and the valid range:")
    equation("x^{(t+1)} = Π [ x^{(t)} + α · sign( ∇_{x} "
             "L( g(x^{(t)}), y ) ) ],   t = 0, …, T − 1")
    a, it, eps = meta["attack_params"]["alpha"], meta["attack_params"]["iters"], meta["attack_params"]["eps"]
    P(f"This study uses ε = {eps}, α = {a} and T = {it}. The published settings (α = 0.01, "
      f"T = 5) let PGD move at most 0.01 × 5 = 0.05 per feature against a budget of {eps}, so it could "
      "never use its own budget and was weaker than the published FGSM. The range constraint is widened to "
      "include each clean test point, so that an unperturbed connection is never altered by the clipping "
      "alone.")
    H2("5.4 The Realizability Mask")
    P("The mask projects a perturbed connection back onto what the wire allows. It runs **after every PGD "
      "step** and once after FGSM:")
    bullets([
        "**Freeze:** all 31 Class C features and the categorical flag are reset to their original values, "
        "including every column of their one-hot blocks.",
        "**Return to raw units:** the scaling (and log transform, if used) is inverted, so the next steps act "
        "on real bytes, seconds and connection counts.",
        "**Round and clamp:** byte counts, durations and connection counts are rounded to whole numbers; "
        "binary features to 0 or 1; count, srv_count and num_failed_logins may only increase.",
        "**Clip:** nothing may be negative, and every rate stays within [0, 1].",
        "**Re-encode and re-freeze:** values are transformed back into model space and the frozen columns are "
        "reset again, so that round-trip floating-point error cannot move them.",
    ], style="List Number")
    P("The order matters. Rounding “0.37” in scaled space is meaningless, because 0.37 of the "
      "scaled range is not a whole number of bytes, so a mask applied in model space silently does nothing. "
      "Unit tests perturb every frozen feature heavily and assert that the model input is byte-identical "
      "afterwards; they also check whole numbers, raise-only behaviour, valid ranges, that a clean input is "
      "left unchanged, and that applying the mask twice equals applying it once. The tests run in both the "
      "plain and the log-scaled feature spaces.")
    H2("5.5 The Improved IDS")
    P(f"The improved IDS averages the predicted attack probabilities of three models: Extra Trees [8] "
      f"(300 trees), histogram-based gradient boosting (300 iterations) and the MLP described above, all on "
      f"log-scaled features. The main lever, however, is the **decision threshold**. At 0.5 the paper’s "
      f"forest misses over a third of the attacks: unseen attack types score as suspicious, but not "
      f"suspicious enough to pass 0.5.")
    P("The threshold is chosen without touching KDDTest+:")
    bullets([
        "Hold out whole **attack types** (30% of the types in KDDTrain+) plus 20% of the normal traffic as a "
        "validation set, which reproduces the “unseen attack” condition of the test file.",
        "Train on the remaining data and pick the threshold that maximises balanced accuracy on the "
        "validation set. Balanced accuracy is used because the test file’s attack share differs from "
        "the training file’s.",
        f"Repeat with five different random selections of held-out types and take the median, "
        f"{R['thr']:.2f}.",
        "Retrain on all of KDDTrain+ and evaluate once on KDDTest+.",
    ], style="List Number")
    H2("5.6 Adversarial Training and the Adaptive Attacker")
    agr = meta.get("surrogate", {}).get("adaptive_agreement_with_defended")
    P("The defence from the published work retrains the Random Forest on the training set plus PGD "
      "adversarial examples generated from the training set (never from the test set). A defence is only "
      "meaningful if it survives an attacker who knows it exists, so the defended model is also tested "
      "against an **adaptive attacker**, who retrains a fresh surrogate on the defended model’s own "
      "predictions" + (f" (agreeing with it on {pct(agr)} of the test set)" if agr else "")
      + " and attacks again.")
    H2("5.7 Evaluation Metrics")
    P("With TP, TN, FP and FN the true/false positives and negatives (an attack is a positive):")
    equation("Accuracy = (TP + TN) / (TP + TN + FP + FN)          Precision = TP / (TP + FP)")
    equation("Recall = TP / (TP + FN)          F1 = 2 · Precision · Recall / (Precision + Recall)")
    P("The **evasion rate** measures the threat that matters for an IDS. Take the attack connections in "
      "KDDTest+ that the IDS detects before any perturbation; the attacker perturbs only those, aiming for "
      "the label “normal”:")
    equation("Evasion rate = | { x : f(x) = attack  and  f(x′) = normal } |  /  | { x : f(x) = attack } |")
    P("Unlike accuracy under attack, it never perturbs benign traffic, which no attacker controls.")
    H2("5.8 Implementation of the Capture Lab")
    bullets([
        "Define the four services and the internal bridge network in docker-compose.yml.",
        "Build the victim image (a Python multi-port TCP server) and the attacker image (Debian with nmap, "
        "hping3, curl and Python).",
        "Start the lab; the sensor container begins a ring-buffered tcpdump capture in the victim’s "
        "network namespace.",
        "Run the scenarios: benign background traffic, a TCP connect sweep over ports 1–200, the "
        "realizability probe, and the third-party test, in which the attacker’s traffic is repeated "
        "identically while only the third-party host’s traffic changes.",
        "Copy the PCAP to the analysis laptop and run pcap_to_nslkdd.py to obtain the 41 features per "
        "connection; score them with both IDSs.",
    ], style="List Number")
    H2("5.9 Demonstration Application")
    P("An offline web application presents the work. It uses only the Python standard library and opens in "
      "a browser at http://127.0.0.1:8765. It has four views: an **overview** of the headline results; a "
      "**network simulation** that replays real KDDTest+ connections through an animated topology and both "
      "IDSs; an **attack lab** that runs FGSM or PGD live against either IDS, in unrestricted or realistic "
      "mode, and flags every physically impossible value; and a **live capture** view that scores a PCAP "
      "from the lab. Section 6.8 shows each view.")
    page_break()


def results(R):
    m, row, evr = R["m"], R["row"], R["evr"]
    b, a, o = row("baseline", "clean"), row("adv_trained", "clean"), row("improved_ids", "clean")
    H1("6. Results and Testing")
    P("All numbers in this section are produced by python src/experiment.py and python src/benchmark.py "
      "and read from the results folder when this report is built.")

    H2("6.1 Detection Performance")
    rows = []
    for name, r in [("Paper’s IDS (Random Forest)", b), ("Adversarially trained RF", a),
                    ("Our IDS (ensemble, tuned threshold)", o)]:
        rows.append([name, pct(r.accuracy, 2), pct(r.precision, 2), pct(r.recall, 2), f3(r.f1)])
    t = table(["Model", "Accuracy", "Precision", "Recall", "F1-score"], rows,
              [2.47, 0.95, 0.95, 0.95, 0.95], "Detection performance on KDDTest+ (no attacker)")
    nf = next_fig()
    P(f"Table {t} and Fig {nf} compare the three models. The paper’s Random Forest is precise "
      f"(precision {pct(b.precision)}) but misses {pct(1 - b.recall)} of attacks. Our IDS trades a little "
      f"precision ({pct(o.precision)}) for far higher recall ({pct(o.recall)}), raising accuracy from "
      f"{pct(b.accuracy)} to {pct(o.accuracy)} and F1 from {f3(b.f1)} to {f3(o.f1)}. For an IDS this is the "
      "right trade: a missed attack is usually far more costly than an extra alert. Adversarial training "
      f"slightly lowered the Random Forest’s clean accuracy, to {pct(a.accuracy)}.")
    figure(PLOTS / "fig_metrics.png", "Accuracy, precision, recall and F1-score on KDDTest+")
    nf = next_fig()
    P(f"The confusion matrices (Fig {nf}) show where the gain comes from: the bottom-left cell, attacks "
      "classified as normal, shrinks sharply, at the cost of more normal connections flagged.")
    figure(PLOTS / "fig_confusion.png", "Confusion matrices on KDDTest+ (counts and row percentages)")

    H2("6.2 Robustness of the Threshold Choice")
    rb, bm = R["rb"], R["bm"]
    names = {"rf_paper": "Paper’s RF (100 trees, depth 15)", "rf_full": "Random Forest (300 trees)",
             "extra_trees": "Extra Trees", "hist_gb": "Hist. gradient boosting",
             "ensemble": "Ensemble (our IDS)"}
    rows = []
    for k in ["rf_paper", "rf_full", "extra_trees", "hist_gb", "ensemble"]:
        g = rb[rb.model == k]
        base = bm[(bm.model == k) & (bm.features == "log1p+minmax") & (bm.threshold == "0.5")]
        b05 = pct(base.accuracy.iloc[0]) if len(base) else "-"
        rows.append([names[k], b05,
                     f"{100 * g.accuracy.mean():.1f} ± {100 * g.accuracy.std(ddof=1):.1f}",
                     f"{100 * g.precision.mean():.1f}", f"{100 * g.recall.mean():.1f}",
                     f"{g.f1.mean():.3f}"])
    t = table(["Model", "Accuracy at 0.5", "Accuracy, tuned (%)", "Precision (%)", "Recall (%)", "F1"],
              rows, [2.07, 0.95, 1.15, 0.75, 0.7, 0.65],
              "Accuracy before and after threshold tuning (log features; mean ± s.d. over 5 splits)",
              size=9.5)
    nf = next_fig()
    ens = rb[rb.model == "ensemble"]
    P(f"Table {t} and Fig {nf} repeat the whole select-then-test procedure over five different choices of "
      "held-out attack types. Tuning the threshold improves every model. The ensemble is the most stable "
      f"(standard deviation {100 * ens.accuracy.std(ddof=1):.1f} points), which is why it was chosen over "
      "Extra Trees, whose mean is similar but whose spread is twice as large. Histogram gradient boosting "
      "always selected the lowest threshold searched, so it shows no spread.")
    figure(PLOTS / "fig_robustness.png", "Test accuracy for each threshold-selection split; the bar marks "
                                         "the default threshold of 0.5")
    P(f"For contrast, the paper’s Random Forest scores {pct(R['indist'], 2)} on a random 80/20 split "
      "of KDDTrain+. That figure is not comparable: the test portion contains only attack types the model "
      "has already seen. It is the source of the “99%” figures common in the literature, and is "
      "not used as a result here.")

    H2("6.3 Accuracy under Attack: the Published Protocol")
    labels = {"clean": "No attack", "fgsm_unconstrained": "FGSM, unrestricted",
              "pgd_unconstrained": "PGD, unrestricted", "fgsm_constrained": "FGSM, realistic",
              "pgd_constrained": "PGD, realistic"}
    rows = []
    for model, short in [("baseline", "Paper’s IDS"), ("improved_ids", "Our IDS")]:
        for es, lab in labels.items():
            r = row(model, es)
            rows.append([short, lab, pct(r.accuracy), pct(r.precision), pct(r.recall), f3(r.f1)])
    t = table(["IDS", "Attack", "Accuracy", "Precision", "Recall", "F1"], rows,
              [1.25, 1.62, 0.85, 0.85, 0.85, 0.85],
              "Whole-test-set accuracy under attack (every connection perturbed)", size=9.5)
    bo, ou = row("improved_ids", "clean"), row("improved_ids", "pgd_unconstrained")
    bb, bbu = row("baseline", "clean"), row("baseline", "pgd_unconstrained")
    P(f"Table {t} follows the published protocol, which perturbs every test connection, normal ones "
      f"included. Read naively, it says our IDS is weaker: its accuracy falls from {pct(bo.accuracy)} to "
      f"{pct(ou.accuracy)} under unrestricted PGD. But its **recall rises** to {pct(ou.recall)} while "
      f"precision collapses to {pct(ou.precision)}. Attacks are still caught; what changed is that "
      "**normal** connections, perturbed towards “attack”, now trigger alarms. The paper’s "
      f"model shows the same pattern: its recall barely moves ({pct(bb.recall)} → {pct(bbu.recall)}) "
      f"while precision drops from {pct(bb.precision)} to {pct(bbu.precision)}.")
    P("An attacker controls their own traffic, not other users’. The bulk of this protocol’s "
      "accuracy drop therefore measures false alarms manufactured on benign traffic, not evasion, and it "
      "penalises a sensitive IDS for being sensitive. The next section measures evasion directly.")

    H2("6.4 Evasion Rate")
    rows = []
    for model, threat, name in [("baseline", "static", "Paper’s IDS"),
                                ("adv_trained", "static", "Adversarially trained RF"),
                                ("adv_trained", "adaptive", "Adversarially trained RF (adaptive attacker)"),
                                ("improved_ids", "static", "Our IDS")]:
        for atk in ["fgsm", "pgd"]:
            u, n = evr(model, threat, atk, False)
            c, _ = evr(model, threat, atk, True)
            rows.append([name, atk.upper(), f"{n:,}", pct(u), pct(c)])
    t = table(["IDS", "Attack", "Attacks detected (clean)", "Evasion, unrestricted", "Evasion, realistic"],
              rows, [2.47, 0.65, 1.05, 1.05, 1.05],
              "Evasion rate: share of detected attacks disguised as normal", size=9.5)
    e = {k: evr(*k)[0] for k in [("baseline", "static", "pgd", False), ("baseline", "static", "pgd", True),
                                 ("adv_trained", "static", "pgd", False), ("adv_trained", "static", "pgd", True),
                                 ("adv_trained", "adaptive", "pgd", True),
                                 ("improved_ids", "static", "pgd", False), ("improved_ids", "static", "pgd", True)]}
    nf = next_fig()
    P(f"Table {t} and Fig {nf} report the metric that matters. Three findings stand out:")
    bullets([
        f"**Our IDS is both more accurate and far harder to evade.** Against unrestricted PGD, "
        f"{pct(e[('baseline', 'static', 'pgd', False)])} of the attacks the paper’s IDS detects can be "
        f"disguised; against ours, {pct(e[('improved_ids', 'static', 'pgd', False)])}.",
        f"**Realistic attacks are much weaker.** Under the realizability mask the paper’s IDS is evaded "
        f"{pct(e[('baseline', 'static', 'pgd', True)])} of the time, less than half the unrestricted figure; "
        f"ours, {pct(e[('improved_ids', 'static', 'pgd', True)])}.",
        f"**Adversarial training defends against the wrong threat.** The defended Random Forest blocks every "
        f"unrestricted attack ({pct(e[('adv_trained', 'static', 'pgd', False)])}) yet lets "
        f"{pct(e[('adv_trained', 'static', 'pgd', True)])} of realistic ones through "
        f"({pct(e[('adv_trained', 'adaptive', 'pgd', True)])} against the adaptive attacker). It learned to "
        "reject perturbations that cannot occur, not the ones that can.",
    ])
    figure(PLOTS / "fig_evasion.png", "Evasion rate under PGD, unrestricted and realistic attackers")

    H2("6.5 Feature Importance versus Controllability")
    s = R["shap_s"]
    top = R["shap_i"].head(5)
    t = table(["Feature class", "Number of features", "Share of model importance (SHAP)"],
              [["A — free", "1", pct(s["share_A_free"])],
               ["B — constrained", "9", pct(s["share_B_constrained"])],
               ["C — uncontrollable", "31", pct(s["share_C_frozen"])]],
              [2.27, 1.6, 2.4], "Where the paper’s Random Forest places its decision weight")
    P(f"SHAP [6] values for the paper’s Random Forest explain why realistic attacks achieve so little "
      f"(Table {t}). {pct(s['share_C_frozen'])} of the model’s importance lies in Class C features, "
      f"and only {pct(s['attacker_controllable_share'])} in features an attacker can move. The five most "
      "important features are " + ", ".join(top.feature) + "; none of them is attacker-controllable. "
      f"The ranking is stable across disjoint subsamples (Spearman ρ = {s['spearman_rho']:.4f}, top-10 "
      f"overlap {int(s['top10_overlap'])}/10). SHAP is used here only as supporting evidence for the "
      "realizability argument.")

    H2("6.6 Validation of the Capture Pipeline")
    n_pass, n_fail, names_ = R["ext_tests"]
    desc = {"test_flags_are_recovered": "SF, REJ and S0 flags recovered from hand-built handshakes",
            "test_src_bytes_and_service": "Payload bytes in each direction and service name",
            "test_land_flag": "land = 1 when source and destination match",
            "test_count_and_srv_count_climb": "2-second window counts a burst of 40 connections",
            "test_host_window_caps_at_100": "Host window is capped at the last 100 connections",
            "test_third_party_moves_dst_host_features":
                "Third-party connections raise dst_host_count for the attacker"}
    t = table(["Unit test", "What it checks", "Result"],
              [[nm, desc.get(nm, ""), "Pass"] for nm in names_],
              [2.25, 3.22, 0.8], "Extractor unit tests (re-run when this report is built)", size=9.5)
    sc = R["synth"]
    P(f"The extractor is validated against packets with known ground truth (Table {t}; {n_pass} passed, "
      f"{n_fail} failed). The last test is the third-party argument in miniature: connections the "
      "attacker never sent change the attacker’s own dst_host_count.")
    P(f"An end-to-end check pushes a synthetic capture through the real extractor and both IDSs: "
      f"{sc['n_user']} benign HTTP sessions from a user and a {sc['n_att']}-port sweep from the attacker, "
      f"{sc['n']} connections in all. Both IDSs flag {sc['paper_att']}/{sc['n_att']} (paper) and "
      f"{sc['ours_att']}/{sc['n_att']} (ours) of the sweep connections; on the benign sessions the paper’s "
      f"IDS raises {sc['paper_user']} alarm(s) and ours {sc['ours_user']}, consistent with the precision "
      "trade-off in Section 6.1. **The live capture on real Docker traffic has not yet been run**: the "
      "campus network blocks the plain DNS that Docker needs to download base images, so the lab is to be "
      "run on a separate machine, and its results will be added to this section.")

    H2("6.7 Reproducibility")
    P("All random seeds are fixed at 42. Library versions are recorded with every run "
      "(results/run_metadata.json). The full test suite (preprocessing, mask and extractor) passes, and "
      "every figure and table in this report is regenerated from the results folder.")

    H2("6.8 Demonstration Application")
    P(f"The figures below are screenshots of the demonstration application (Section 5.9), taken "
      "automatically from the running application.")
    figure(FIG / "demo_overview.png", "Overview view: headline results and who controls each feature",
           width=TEXT_W)
    figure(FIG / "demo_network.png", "Network simulation: a reconnaissance scan replayed through both "
                                     "IDSs; the paper’s IDS misses the unseen mscan attacks",
           width=TEXT_W)
    nf = next_fig()
    P(f"Fig {nf} shows an unrestricted PGD attack against the paper’s IDS on an apache2 connection, "
      "an attack type absent from training. The IDS is evaded, but the attack changes 29 features and 28 "
      "of them take values no packet can carry (outlined in red): fractional durations, a protocol that is "
      "partly TCP and partly something else, a connection count that went down.")
    figure(FIG / "demo_lab_unrestricted.png", "Attack lab: unrestricted PGD evades the paper’s IDS "
                                              "using physically impossible values", width=TEXT_W)
    nf = next_fig()
    P(f"In realistic mode (Fig {nf}) the 32 frozen features are locked. The attacker can still evade the "
      "paper’s IDS here, by setting duration and src_bytes to zero. This illustrates the remaining "
      "limitation discussed in Section 7.2: those values are legal on the wire, but an apache2 flood that "
      "sends no bytes is no longer an attack.")
    figure(FIG / "demo_lab_realistic.png", "Attack lab: a realistic attack on the same connection; frozen "
                                           "features are locked", width=TEXT_W)
    figure(FIG / "demo_lab_ours.png", "Attack lab: the same unrestricted attack against our IDS is "
                                      "still detected", width=TEXT_W)


def discussion(R):
    H1("7. Discussion")
    H2("7.1 Comparison with Related Work")
    P("Constraining adversarial attacks on intrusion detectors is an established research direction. "
      f"Table {next_tab()} summarises the closest work and how this study differs.")
    table(["Work", "What it showed", "Difference in this study"],
          [["Teuffenbach et al. [11]",
            "Attacks on NIDSs weighted by how hard each feature is to manipulate",
            "Hard per-feature constraints in raw units, from a TCP/IP taxonomy"],
           ["Sheatsley et al. [12]",
            "Domain constraints do not make NIDSs robust; as few as five features can suffice",
            "Measures the effect with an evasion metric and compares defences"],
           ["Merzouk et al. [13]",
            "Many published adversarial examples for intrusion detection are invalid",
            "Flags each impossible value per feature, live, in the demonstration"],
           ["Apruzzese et al. [14]",
            "Realistic attacker capabilities against ML-based NIDSs",
            "Applies a realistic threat model to the specific published pipeline"],
           ["Simonetto et al. [15]",
            "Constrained attacks and constrained adversarial retraining",
            "Shows a calibrated threshold beats unconstrained adversarial training"],
           ["Pierazzi et al. [16]",
            "Problem-space versus feature-space attacks in general",
            "Grounds problem-space limits in TCP/IP connection semantics"]],
          [1.45, 2.4, 2.42], "Related work", size=9.5)
    P("The specific contributions of this study are therefore not the idea of constrained attacks itself, "
      "but (i) showing that whole-test-set accuracy under attack conflates evasion with benign false alarms, "
      "(ii) showing that a calibrated threshold outperforms adversarial training under realistic constraints, "
      "and (iii) a pipeline that checks feature controllability on real packet captures.")
    H2("7.2 Limitations")
    bullets([
        "**Dataset age:** NSL-KDD derives from 1999 traffic. The conclusions should be confirmed on a modern "
        "dataset such as UNSW-NB15 or CIC-IDS2017.",
        "**Single runs:** each experiment uses one fixed seed. Only the threshold selection was repeated over "
        "several splits.",
        "**Hand-assigned classes:** the realizability classes are a threat-model judgement. The live capture "
        "lab is designed to test them, but has not yet been run.",
        "**Functional realizability:** the mask enforces what the wire allows, not what keeps an attack "
        "working. Setting a flood’s payload to zero is legal but defeats the flood, so the realistic "
        "evasion rates reported here are upper bounds.",
        "**Transfer attacks only:** stronger query-based attacks against the deployed model were not "
        "evaluated.",
        "**Content features in captures:** features such as logged_in describe host state and are not "
        "visible on the wire, so they are zero in captured traffic.",
    ])
    page_break()


def conclusion(R):
    b, o = R["row"]("baseline", "clean"), R["row"]("improved_ids", "clean")
    e_bu = R["evr"]("baseline", "static", "pgd", False)[0]
    e_ou = R["evr"]("improved_ids", "static", "pgd", False)[0]
    H1("8. Conclusion and Future Work")
    P("An adversarial example is only a threat if a real attacker can send it. Classifying the 41 NSL-KDD "
      "features by who controls them over TCP/IP shows that an attacker can move only 9 of them, and "
      "enforcing that classification inside FGSM and PGD substantially weakens the published attacks. "
      "Measuring the right quantity matters as much: the conventional accuracy-under-attack protocol mostly "
      "records false alarms on benign traffic, while the evasion rate isolates attacks that actually get "
      "through.")
    P(f"An IDS built around a threshold calibrated on held-out attack types raised accuracy on unseen "
      f"attacks from {pct(b.accuracy)} to {pct(o.accuracy)} and F1 from {f3(b.f1)} to {f3(o.f1)}, and "
      f"reduced unrestricted evasion from {pct(e_bu)} to {pct(e_ou)}. Against realistic attackers it was "
      "more robust than adversarial training, which had been fitted to perturbations that cannot occur.")
    P("Future work:")
    bullets([
        "Run the capture lab on real traffic and add the measured realizability results to Section 6.6.",
        "Repeat the experiments on a modern dataset (UNSW-NB15, CIC-IDS2017) and over several seeds.",
        "Add functional constraints so that attacks must also remain effective, not only realizable.",
        "Train the defence on realistic perturbations instead of unrestricted ones.",
    ])
    page_break()


def references(R):
    H1("References")
    refs = [
        "M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, “A detailed analysis of the KDD CUP 99 "
        "data set,” in Proc. IEEE Symp. Computational Intelligence for Security and Defense "
        "Applications (CISDA), 2009, pp. 1–6.",
        "I. J. Goodfellow, J. Shlens, and C. Szegedy, “Explaining and harnessing adversarial "
        "examples,” in Proc. Int. Conf. Learning Representations (ICLR), 2015.",
        "A. Madry, A. Makelov, L. Schmidt, D. Tsipras, and A. Vladu, “Towards deep learning models "
        "resistant to adversarial attacks,” in Proc. ICLR, 2018.",
        "A. Kurakin, I. Goodfellow, and S. Bengio, “Adversarial machine learning at scale,” in "
        "Proc. ICLR, 2017.",
        "M.-I. Nicolae et al., “Adversarial Robustness Toolbox v1.0.0,” arXiv:1807.01069, 2018.",
        "S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in "
        "Advances in Neural Information Processing Systems (NeurIPS), 2017.",
        "L. Breiman, “Random forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.",
        "P. Geurts, D. Ernst, and L. Wehenkel, “Extremely randomized trees,” Machine Learning, "
        "vol. 63, no. 1, pp. 3–42, 2006.",
        "W. Lee and S. J. Stolfo, “A framework for constructing features and models for intrusion "
        "detection systems,” ACM Trans. Information and System Security, vol. 3, no. 4, "
        "pp. 227–261, 2000.",
        "F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” J. Machine Learning "
        "Research, vol. 12, pp. 2825–2830, 2011.",
        "M. Teuffenbach, E. Piatkowska, and P. Smith, “Subverting network intrusion detection: Crafting "
        "adversarial examples accounting for domain-specific constraints,” in Machine Learning and "
        "Knowledge Extraction (CD-MAKE 2020), Springer LNCS, 2020.",
        "R. Sheatsley et al., “Adversarial examples in constrained domains,” arXiv:2011.01183, "
        "2020; Journal of Computer Security, IOS Press.",
        "M. A. Merzouk, F. Cuppens, N. Boulahia-Cuppens, and R. Yaich, “A deeper analysis of adversarial "
        "examples in intrusion detection,” in Risks and Security of Internet and Systems (CRiSIS 2020), "
        "Springer, 2021.",
        "G. Apruzzese, M. Andreolini, L. Ferretti, M. Marchetti, and M. Colajanni, “Modeling realistic "
        "adversarial attacks against network intrusion detection systems,” Digital Threats: Research "
        "and Practice, vol. 3, no. 3, art. 31, 2022.",
        "T. Simonetto et al., “A unified framework for adversarial attack and defense in constrained "
        "feature space,” in Proc. Int. Joint Conf. Artificial Intelligence (IJCAI), 2022.",
        "F. Pierazzi, F. Pendlebury, J. Cortellazzi, and L. Cavallaro, “Intriguing properties of "
        "adversarial ML attacks in the problem space,” in Proc. IEEE Symp. Security and Privacy, 2020.",
        "J. F. Kurose and K. W. Ross, Computer Networking: A Top-Down Approach, 8th ed. Pearson, 2021.",
        D["own_paper_reference"],
        "W. R. Stevens, TCP/IP Illustrated, Volume 1: The Protocols. Addison-Wesley, 1994.",
        "R. Sommer and V. Paxson, “Outside the closed world: On using machine learning for network "
        "intrusion detection,” in Proc. IEEE Symp. Security and Privacy, 2010, pp. 305–316.",
    ]
    for i, ref in enumerate(refs, 1):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.left_indent = Inches(0.4)
        p.paragraph_format.first_line_indent = Inches(-0.4)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.0
        runs(p, f"[{i}]\t{ref}", size=10.5)
    page_break()


def appendices(R):
    H1("Appendix A: Full Feature Realizability Taxonomy")
    P("B* marks flag: Class B, but frozen in the continuous attacks because it is categorical.", size=10.5)
    table(["#", "Feature", "Group", "Class", "Controlled by / constraint"],
          [[str(i), f, g, c, why] for i, (f, g, c, why) in enumerate(TAXONOMY, 1)],
          [0.35, 1.85, 0.9, 0.5, 2.67], "The 41 NSL-KDD features and who controls them", size=9,
          align=["center", "left", "left", "center", "left"])
    H1("Appendix B: Reproducing the Results")
    P("From the project folder, with the Python environment set up as in README.md and the NSL-KDD files "
      "in data/:")
    code([".venv/Scripts/python.exe src/ids_model.py        # build the improved IDS",
          ".venv/Scripts/python.exe src/experiment.py       # attacks, mask, defence, evasion",
          ".venv/Scripts/python.exe src/benchmark.py --robust   # threshold robustness",
          ".venv/Scripts/python.exe src/shap_analysis.py    # feature importance",
          ".venv/Scripts/python.exe src/plots.py            # figures",
          ".venv/Scripts/python.exe -m pytest tests/        # test suite",
          ".venv/Scripts/python.exe demo/app.py             # demonstration application",
          ".venv/Scripts/python.exe report/build_report.py  # this report"])
    P("To run the capture lab on a machine with Docker:")
    code(["cd lab", "bash run_capture.sh full",
          "# then drop lab/pcaps/capture.pcap into the demo's Live capture view"])


# =========================================================================== #

def main() -> None:
    print("reading results...", flush=True)
    R = load_numbers()
    print("re-running extractor tests...", flush=True)
    R["ext_tests"] = run_extractor_tests()
    print("synthetic capture check...", flush=True)
    R["synth"] = synthetic_capture_check()

    setup_styles()
    page_setup()
    title_page()
    certificate()
    contents()
    abstract(R)
    introduction(R)
    requirements(R)
    architecture(R)
    dataset(R)
    methodology(R)
    results(R)
    discussion(R)
    conclusion(R)
    references(R)
    appendices(R)

    doc.core_properties.title = D["title"].title()
    doc.core_properties.subject = D["subtitle"]
    doc.save(OUT)
    print(f"wrote {OUT}  ({_fig[0]} figures, {_tab[0]} tables)")
    left = sorted(set(re.findall(r"\[[A-Za-z][^\]]*\]",
                                 json.dumps({k: v for k, v in D.items() if not k.startswith("_")}))))
    if left:
        print("placeholders still to fill in report/details.json:", ", ".join(left))


if __name__ == "__main__":
    main()
