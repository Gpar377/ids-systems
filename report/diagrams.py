"""Architecture and topology diagrams for the report.

    python report/diagrams.py   -> report/figures/fig_topology.png, fig_pipeline.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
INK, INK2, LINE, FILL = "#0b0b0b", "#52514e", "#8a8984", "#f6f6f4"
BLUE, ORANGE, RED, YELLOW, AQUA = "#2a78d6", "#eb6834", "#e34948", "#c98a00", "#1baf7a"

plt.rcParams.update({"font.family": "Times New Roman", "savefig.dpi": 220,
                     "savefig.bbox": "tight", "savefig.facecolor": "white"})


def box(ax, x, y, w, h, title, lines=(), edge=LINE, fill=FILL, lw=1.6, size=11,
        dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=lw, edgecolor=edge, facecolor=fill,
                                linestyle="--" if dashed else "-"))
    n = len(lines)
    top = y + h / 2 + 0.17 * n
    ax.text(x + w / 2, top, title, ha="center", va="center", fontsize=size,
            fontweight="bold", color=INK)
    for i, t in enumerate(lines):
        ax.text(x + w / 2, top - 0.36 * (i + 1), t, ha="center", va="center",
                fontsize=size - 2, color=INK2)


def arrow(ax, a, b, color=LINE, style="-|>", dashed=False, rad=0.0, lw=1.5, label=None,
          lpos=0.5, loff=(0, 0.18)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=13, color=color,
                                 linewidth=lw, linestyle="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}"))
    if label:
        mx = a[0] + (b[0] - a[0]) * lpos + loff[0]
        my = a[1] + (b[1] - a[1]) * lpos + loff[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=9, color=INK2,
                style="italic")


def topology() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    # isolated Docker host
    ax.add_patch(FancyBboxPatch((0.2, 2.95), 13.6, 5.45, boxstyle="round,pad=0.02,rounding_size=0.2",
                                linewidth=1.3, edgecolor=LINE, facecolor="none", linestyle="--"))
    ax.text(0.45, 8.15, "Docker host  —  isolated bridge network, no route to the LAN or Internet",
            fontsize=10, color=INK2, va="center")

    box(ax, 0.6, 5.7, 3.4, 1.9, "Attacker", ["192.168.90.20", "nmap · hping3 · probes"],
        edge=RED, lw=2)
    box(ax, 0.6, 3.3, 3.4, 1.9, "Third-party host", ["192.168.90.30", "benign HTTP bursts"])
    # bridge
    ax.add_patch(FancyBboxPatch((5.0, 4.2), 3.2, 2.5, boxstyle="round,pad=0.02,rounding_size=0.15",
                                linewidth=1.6, edgecolor=INK2, facecolor="#eef2f6"))
    ax.text(6.6, 5.85, "Bridge  idsnet", ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.text(6.6, 5.4, "192.168.90.0/24", ha="center", fontsize=9.5, color=INK2)
    ax.text(6.6, 4.95, "internal: true", ha="center", fontsize=9.5, color=INK2)
    ax.text(6.6, 4.55, "(no gateway)", ha="center", fontsize=9.5, color=INK2)
    box(ax, 9.5, 5.2, 3.9, 2.4, "Victim server", ["192.168.90.10", "open: 21 ftp · 22 ssh",
                                                   "80 http · 143 imap4",
                                                   "closed ports → RST (REJ)"],
        edge=BLUE, lw=2)
    box(ax, 9.5, 3.25, 3.9, 1.5, "IDS sensor (tcpdump)", ["shares the victim's network",
                                                          "namespace → capture.pcap"],
        edge=YELLOW, lw=2)

    arrow(ax, (4.0, 6.65), (5.0, 5.9), color=RED, lw=2)
    arrow(ax, (4.0, 4.25), (5.0, 5.0), color=LINE, lw=2)
    arrow(ax, (8.2, 5.55), (9.5, 6.3), color=INK2, lw=2)
    arrow(ax, (11.45, 5.2), (11.45, 4.75), color=YELLOW, lw=2, style="-|>")

    # offline analysis row: straight left-to-right flow, no crossing arrows
    y = 0.75
    box(ax, 0.6, y, 3.4, 1.55, "capture.pcap", ["copied off the", "Docker host"], size=10.5)
    box(ax, 4.9, y, 3.9, 1.55, "pcap_to_nslkdd.py", ["2 s window · last-100 window",
                                                     "→ 41 NSL-KDD features"], size=10.5)
    box(ax, 9.7, y, 3.7, 1.55, "Both IDSs score it", ["Paper's RF (threshold 0.5)",
                                                      "Our ensemble (threshold 0.04)"],
        edge=BLUE, lw=2, size=10.5)
    arrow(ax, (9.5, 3.9), (2.3, y + 1.55), color=YELLOW, dashed=True, lw=1.5)
    arrow(ax, (4.0, y + 0.78), (4.9, y + 0.78), lw=1.6)
    arrow(ax, (8.8, y + 0.78), (9.7, y + 0.78), lw=1.6)
    ax.text(7.0, 0.3, "offline analysis on the demo laptop (Live capture tab)", ha="center",
            fontsize=9.5, color=INK2, style="italic")
    fig.savefig(OUT / "fig_topology.png")
    plt.close(fig)


def pipeline() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.text(0.1, 6.65, "Defender", fontsize=11.5, fontweight="bold", color=BLUE)
    ax.text(0.1, 3.05, "Attacker", fontsize=11.5, fontweight="bold", color=RED)

    y1 = 4.2
    box(ax, 0.1, y1, 2.7, 2.0, "NSL-KDD", ["KDDTrain+ 125,973", "KDDTest+ 22,544",
                                            "41 features + label"], size=10.5)
    box(ax, 3.3, y1, 3.3, 2.0, "Preprocessing", ["one-hot: protocol, service, flag",
                                                  "log1p + MinMax scaling",
                                                  "raw ↔ encoded index map (122)"], size=10.5)
    box(ax, 7.1, y1, 3.0, 2.0, "IDS models", ["Paper's RF (thr. 0.5)",
                                               "Our ensemble (thr. 0.04)",
                                               "ExtraTrees + HistGB + MLP"],
        edge=BLUE, lw=2, size=10.5)
    box(ax, 10.6, y1, 3.3, 2.0, "Evaluation", ["accuracy · precision", "recall · F1",
                                                "evasion rate"], size=10.5)
    arrow(ax, (2.8, y1 + 1.0), (3.3, y1 + 1.0), lw=1.8)
    arrow(ax, (6.6, y1 + 1.0), (7.1, y1 + 1.0), lw=1.8)
    arrow(ax, (10.1, y1 + 1.0), (10.6, y1 + 1.0), lw=1.8)

    y2 = 0.4
    box(ax, 0.1, y2, 2.7, 2.2, "Surrogate", ["MLP trained by the", "attacker (no access",
                                              "to the real IDS)"], edge=RED, lw=2, size=10.5)
    box(ax, 3.3, y2, 3.3, 2.2, "FGSM / PGD", ["ε = 0.2 per dimension", "PGD: 10 steps, α = 0.02",
                                               "target label: normal"], edge=RED, lw=2, size=10.5)
    box(ax, 7.1, y2, 3.0, 2.2, "Realizability mask", ["freeze 32 features",
                                                       "round & clamp in raw units",
                                                       "after every step"],
        edge=AQUA, lw=2.2, size=10.5)
    box(ax, 10.6, y2, 3.3, 2.2, "Adversarial", ["connection sent to", "the real IDS",
                                                 "(transfer attack)"], edge=RED, lw=2, size=10.5)
    arrow(ax, (2.8, y2 + 1.1), (3.3, y2 + 1.1), color=RED, lw=1.8)
    arrow(ax, (6.6, y2 + 1.1), (7.1, y2 + 1.1), color=RED, lw=1.8)
    arrow(ax, (10.1, y2 + 1.1), (10.6, y2 + 1.1), color=RED, lw=1.8)
    arrow(ax, (12.25, y2 + 2.2), (8.6, y1), color=RED, lw=1.8, rad=0.25)
    ax.text(13.1, 3.4, "is it still\ndetected?", ha="center", va="center", fontsize=9,
            color=INK2, style="italic")
    arrow(ax, (4.95, y1), (1.45, y2 + 2.2), color=LINE, dashed=True, lw=1.3, rad=-0.15,
          label="same feature space", lpos=0.5, loff=(-0.35, 0.05))
    fig.savefig(OUT / "fig_pipeline.png")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    topology()
    pipeline()
    print("wrote", OUT / "fig_topology.png", "and", OUT / "fig_pipeline.png")
