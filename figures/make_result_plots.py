"""Redrawn result plots (Figs. 5-7) in plain IEEE style. Same data sources as make_plots.py:
data/reported_results.csv (means, SDs), data/*_per_seed.csv (seed dots), data/recomputed_statistics.json (IQ CI).
No values are typed into the plotting code."""
import csv, json, pathlib
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = pathlib.Path(__file__).resolve().parent
D = HERE.parent / "results" / "original_runs"; OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "legend.fontsize": 7, "pdf.fonttype": 42, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.top": False, "ytick.right": True,
    "legend.frameon": False,
})
M = ["FedAvg-all", "Gold", "FedEraser", "Gradient ascent", "Retain fine-tuning"]
CSVNAME = {"FedAvg-all": "FedAvg-all", "Gold": "FedAvg-retained (gold)", "FedEraser": "FedEraser",
           "Gradient ascent": "Grad-ascent", "Retain fine-tuning": "Retain-finetune"}
TICK = {"FedAvg-all": "All-\nclient", "Gold": "Gold", "FedEraser": "Fed-\nEraser",
        "Gradient ascent": "Grad.\nascent", "Retain fine-tuning": "Fine-\ntuning"}
# grey scale with one accent (FedEraser); a hatch separates the two light greys in print
FILL = {"FedAvg-all": "#8c8c8c", "Gold": "#ffffff", "FedEraser": "#3b6ea5",
        "Gradient ascent": "#d4d4d4", "Retain fine-tuning": "#5a5a5a"}
HATCH = {"Gradient ascent": "////"}
INK = "black"; TAU_U = 0.5 * 25.65

rows = list(csv.DictReader(open(D / "reported_results.csv")))
def get(exp, m, metric):
    r = [x for x in rows if x["experiment"] == exp and x["method"] == m and x["metric"] == metric][0]
    return float(r["mean"]), (float(r["sample_sd"]) if r["sample_sd"] else 0.0)
def seeds(fname, m, col):
    return [float(r[col]) for r in csv.DictReader(open(D / fname)) if r["method"] == CSVNAME[m]]
def tag(ax, t): ax.set_title(t, loc="left", fontsize=8, pad=3)

def bars(ax, exp, metric, fname, col):
    for i, m in enumerate(M):
        mu, sd = get(exp, m, metric)
        ax.bar(i, mu, 0.62, color=FILL[m], edgecolor=INK, linewidth=0.6, hatch=HATCH.get(m, ""), zorder=2)
        ax.errorbar(i, mu, yerr=sd, color=INK, elinewidth=0.7, capsize=2, capthick=0.7, zorder=3)
        v = seeds(fname, m, col)
        ax.scatter([i + 0.2] * len(v), v, s=5, color=INK, linewidth=0, zorder=4)
    ax.set_xticks(range(5)); ax.set_xticklabels([TICK[m] for m in M]); ax.set_xlim(-0.55, 4.55)

# ------------------------------------------------ Fig. 5: canary, original runs (n = 5)
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.3), gridspec_kw={"wspace": 0.38})
ax = axs[0]; bars(ax, "canary5", "retain_acc_pct", "canary_per_seed.csv", "retain_acc")
ax.axhline(TAU_U, color=INK, lw=0.7, ls="--", zorder=1); ax.axhline(25.65, color=INK, lw=0.7, ls=":", zorder=1)
ax.set_ylim(0, 30); ax.set_ylabel("Retained accuracy (%)"); tag(ax, "(a)")
ax = axs[1]; bars(ax, "canary5", "canary_asr_pct", "canary_per_seed.csv", "canary_asr")
ax.axhline(90, color=INK, lw=0.7, ls="--", zorder=1)
ax.set_ylim(0, 105); ax.set_ylabel("Canary ASR (%)"); tag(ax, "(b)")
ax = axs[2]
ax.axhspan(-2, 2, color="#cfdcec", lw=0, zorder=0)
ax.axhline(90, color=INK, lw=0.7, ls="--"); ax.axvline(TAU_U, color=INK, lw=0.7, ls="--")
MK = {"FedAvg-all": "s", "Gold": "D", "FedEraser": "o", "Gradient ascent": "^", "Retain fine-tuning": "v"}
for m in M:
    a, _ = get("canary5", m, "retain_acc_pct"); s, _ = get("canary5", m, "canary_asr_pct")
    ax.scatter(a, s, s=22 if m != "FedEraser" else 12, marker=MK[m], facecolor=FILL[m], edgecolor=INK, linewidth=0.6, zorder=4 if m != "FedEraser" else 5)
NOTE = {"FedAvg-all": (-0.3, 7, "right"), "Retain fine-tuning": (-0.4, 0, "right"),
        "Gradient ascent": (-0.4, 0, "right")}
for m, (dx, dy, ha) in NOTE.items():
    a, _ = get("canary5", m, "retain_acc_pct"); s, _ = get("canary5", m, "canary_asr_pct")
    ax.text(a + dx, s + dy, {"FedAvg-all": "All-client", "Retain fine-tuning": "Fine-tuning", "Gradient ascent": "Grad. ascent"}[m], fontsize=7, ha=ha, va="center")
a, _ = get("canary5", "Gold", "retain_acc_pct")
ax.annotate("Gold and\nFedEraser", xy=(a, 1.5), xytext=(15.0, 12), fontsize=7, ha="center", va="center",
            arrowprops=dict(arrowstyle="-", lw=0.5, color=INK, relpos=(1.0, 0.0), shrinkA=0, shrinkB=3))
ax.set_xlim(10, 25); ax.set_ylim(-5, 105)
ax.set_xlabel("Retained accuracy (%)"); ax.set_ylabel("Canary ASR (%)"); tag(ax, "(c)")
fig.legend([Line2D([0], [0], color=INK, lw=0.7, ls="--"), Line2D([0], [0], color=INK, lw=0.7, ls=":"),
            Line2D([0], [0], marker="o", color=INK, lw=0, ms=2), Patch(facecolor="#cfdcec")],
           ["gates (utility 12.83%, canary 90%)", "classical reference 25.65%", "individual seeds", "Gold $\\pm$2 pp, panel (c)"],
           loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1), handlelength=1.8, columnspacing=1.4)
fig.savefig(OUT / "fig_methods.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ------------------------------------------------ Fig. 6: mild IQ, original runs (n = 10)
st = json.loads((D / "recomputed_statistics.json").read_text())["iq:FedEraser:forget_acc:paired"]
d, ci = st["diff"], st["ci"]
fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.2), gridspec_kw={"wspace": 0.4, "width_ratios": [1, 1, 0.85]})
for ax, metric, col, ylab, lim, t in [(axs[0], "forget_acc_pct", "forget_acc", "Forget accuracy (%)", (26, 37), "(a)"),
                                      (axs[1], "retain_acc_pct", "retain_acc", "Retained accuracy (%)", (17, 24), "(b)")]:
    g, _ = get("iq10", "Gold", metric)
    ax.axhline(g, color=INK, lw=0.6, ls=":", zorder=1)
    for i, m in enumerate(M):
        mu, sd = get("iq10", m, metric)
        ax.errorbar(i, mu, yerr=sd, fmt=MK[m], ms=4.5, mfc=FILL[m], mec=INK, mew=0.6, ecolor=INK, elinewidth=0.7, capsize=2, zorder=3)
        v = seeds("iq_per_seed.csv", m, col)
        ax.scatter([i + 0.22] * len(v), v, s=4, color="#555555", linewidth=0, zorder=2)
    ax.set_xticks(range(5)); ax.set_xticklabels([TICK[m] for m in M]); ax.set_xlim(-0.5, 4.6)
    ax.set_ylim(*lim); ax.set_ylabel(ylab); tag(ax, t)
ax = axs[2]
ax.axvspan(-2, 2, color="#cfdcec", lw=0); ax.axvline(0, color=INK, lw=0.5)
ax.errorbar(d, 0, xerr=[[d - ci[0]], [ci[1] - d]], fmt="o", ms=4.5, mfc=FILL["FedEraser"], mec=INK, mew=0.6, ecolor=INK, elinewidth=0.9, capsize=2.5)
ax.text(0, 0.55, f"mean {d:.2f} pp\n90% CI [{ci[0]:.2f}, {ci[1]:.2f}]", ha="center", va="bottom", fontsize=7, bbox=dict(fc="white", ec="none", pad=1.0), zorder=5)
ax.text(-1.9, -0.55, "$-\\delta$", fontsize=7, ha="left"); ax.text(1.9, -0.55, "$+\\delta$", fontsize=7, ha="right")
ax.set_xlim(-3, 3); ax.set_ylim(-0.75, 1.25); ax.set_yticks([]); ax.tick_params(right=False)
ax.set_xlabel("FedEraser $-$ Gold, forget acc. (pp)"); tag(ax, "(c)")
fig.savefig(OUT / "fig_iq.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# ------------------------------------------------ Fig. 7: pooling ablation (n = 5)
cc = {"retain_acc_pct": seeds("canary_per_seed.csv", "FedAvg-all", "retain_acc"),
      "canary_asr_pct": seeds("canary_per_seed.csv", "FedAvg-all", "canary_asr")}
gp = {"retain_acc_pct": seeds("gap_metrics_per_seed.csv", "FedAvg-all", "retain_acc"),
      "canary_asr_pct": seeds("gap_metrics_per_seed.csv", "FedAvg-all", "canary_asr")}
fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.0), gridspec_kw={"wspace": 0.55})
for ax, metric, ylab, gate, ymax, t in [(axs[0], "retain_acc_pct", "Retained accuracy (%)", TAU_U, 30, "(a)"),
                                        (axs[1], "canary_asr_pct", "Canary ASR (%)", 90, 105, "(b)")]:
    for i, (m, v, fc) in enumerate([("Center-crop", cc, "#8c8c8c"), ("GAP", gp, "#ffffff")]):
        mu, sd = get("pool5", m, metric)
        ax.bar(i, mu, 0.55, color=fc, edgecolor=INK, linewidth=0.6, zorder=2)
        ax.errorbar(i, mu, yerr=sd, color=INK, elinewidth=0.7, capsize=2, capthick=0.7, zorder=3)
        ax.scatter([i + 0.18] * len(v[metric]), v[metric], s=5, color=INK, linewidth=0, zorder=4)
    ax.axhline(gate, color=INK, lw=0.7, ls="--", zorder=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Center-\ncrop", "GAP"]); ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, ymax); ax.set_ylabel(ylab); tag(ax, t)
fig.savefig(OUT / "fig_pooling.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)
print("ok", d, ci)
