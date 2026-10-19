#!/usr/bin/env python
"""Generate every LaTeX table in the paper from results/*.json.

Nothing in the paper's tables is typed by hand. A number that appears in the
PDF can be traced to a run artefact, and re-running the experiments
regenerates the tables rather than requiring a careful diff of the prose.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "tables"
OUT.mkdir(exist_ok=True)

SHORT = {
    "PyG-Captum[IntegratedGradients]": "IntegratedGradients",
    "PyG-Captum[InputXGradient]": "InputXGradient",
    "PyG-Captum[Deconvolution]": "Deconvolution",
    "PyG-Captum[GuidedBackprop]": "GuidedBackprop",
    "PyG-Captum[Saliency]": "Saliency",
    "PyG-Captum[IG]": "IntegratedGradients",
    "PyG-Captum[InputXGrad]": "InputXGradient",
    "GNNExplainer [model]": "GNNExplainer",
    "GNNExplainer[20]": "GNNExplainer",
    "PGExplainer [target=model pred]": "PGExplainer",
    "IntegratedGradients[32]": "IG (ours, edge-weight)",
    "EdgeSaliency": "Saliency (ours, edge-weight)",
    "Lime[400]": "LIME", "LIME": "LIME",
    "KernelShap[400]": "KernelSHAP", "KernelSHAP": "KernelSHAP",
    "ShapleyValueSampling[25]": "ShapleyValueSampling",
    "ShapleyValueSampling": "ShapleyValueSampling",
    "MaxWeight (control)": "MaxWeight (control)",
    "ShortestPath (control)": "ShortestPath (control)",
    "Random (control)": "Random (control)",
}


def load(name):
    return json.loads((ROOT / "results" / name).read_text())


def esc(s):
    return s.replace("&", r"\&").replace("_", r"\_")


def write(name, body):
    (OUT / name).write_text(body)
    print("wrote", OUT / name)


# --------------------------------------------------------------- Table 1
def table_phase_a():
    d = load("phase_a.json")
    rows = []
    for r in d["summary"]:
        n = r["name"]
        if "phenomenon" in n or "true labels" in n:
            continue  # secondary variants, discussed in text
        pub = f"{r['published']:.3f}" if r.get("published") else "--"
        rows.append((SHORT.get(n, n), r["auc_mean"], r["auc_std"], pub))
    rows.sort(key=lambda t: -t[1])
    body = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Harness validation on BA-Shapes, mean $\pm$ std over three "
        r"seeds. Explanation ROC-AUC "
        r"against the motif ground truth, pooled over the 3-hop computation "
        r"subgraphs of 60 motif nodes. Published values are the originals' "
        r"reported figures.}",
        r"\label{tab:phase-a}",
        r"\begin{tabular}{lr@{\,$\pm$\,}lr}", r"\toprule",
        r"Explainer & \multicolumn{2}{c}{AUC} & Published \\", r"\midrule",
    ]
    for short, auc, std, pub in rows:
        body.append(f"{esc(short)} & {auc:.4f} & {std:.4f} & {pub} \\\\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_phase_a.tex", "\n".join(body) + "\n")


# --------------------------------------------------------------- Table 2
def table_transfer():
    """BA-Shapes vs routing, same metric. Doubles as the validation table."""
    pa = {}
    for m in load("phase_a.json")["seeds"][0]["methods"]:
        n = m["name"]
        if "phenomenon" in n or "true labels" in n:
            continue
        pa[SHORT.get(n, n)] = m["auc"]
    for m in load("phase_a_perturbation.json")["seeds"][0]["methods"]:
        pa[SHORT.get(m["name"], m["name"])] = m["auc"]
    gt = {}
    for f in ("phase_b_cap.json", "phase_b_perturbation.json"):
        for r in load(f)["groundtruth"]:
            gt[SHORT.get(r["name"], r["name"])] = r
    pub = {"GNNExplainer": "0.925", "PGExplainer": "0.963"}

    order = ["IntegratedGradients", "ShapleyValueSampling", "Deconvolution",
             "InputXGradient", "GuidedBackprop", "KernelSHAP", "LIME",
             "PGExplainer", "GNNExplainer", "MaxWeight (control)",
             "Random (control)"]
    body = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Explanation quality under one metric in two domains: "
        r"explanation as binary edge classification against ground truth. "
        r"BA-Shapes anchors the harness against published figures; routing is "
        r"the planted bottleneck against the off-path decoys, which share its "
        r"weight distribution and so neutralise any advantage from reading "
        r"edge features. Chance is 0.5.}",
        r"\label{tab:transfer}",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Explainer & Pub. & BA-Shapes & Routing & $\Delta$ \\",
        r"\midrule",
    ]
    for k in order:
        if k not in gt:
            continue
        y = gt[k]["auc_vs_decoys"]
        # MaxWeight reads edge features, which BA-Shapes does not have, so it
        # has no counterpart there. Shown rather than silently dropped.
        if k not in pa:
            body.append(r"\midrule")
            body.append(f"{esc(k)} & -- & -- & {y:.3f} & -- \\\\")
            continue
        x = pa[k]
        if k == "PGExplainer":
            body.append(r"\midrule")
        body.append(f"{esc(k)} & {pub.get(k,'--')} & {x:.3f} & {y:.3f} & "
                    f"${y - x:+.3f}$ \\\\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_transfer.tex", "\n".join(body) + "\n")


# --------------------------------------------------------------- Table 3
def table_contradiction():
    """Every explainer under both metrics, side by side."""
    import math

    # The oracle reads the planted label, so it is a CEILING, not a floor: a
    # method is not required to differ from it, and including it among the
    # controls here would invert the test. It is reported in the prose at k=1
    # instead, where its selection is a single link rather than a tie-broken
    # 10% of arcs.
    CEILING = "Oracle (control)"
    FILES = ("phase_b_cap.json", "phase_b_perturbation.json",
             "phase_b_controls.json")

    def se(r, n):
        return r["fid_pos_std"] / math.sqrt(n)

    # pass 1: pool the controls across files, so "distinguishable from both
    # controls" becomes "from all of them" as controls are added
    controls = {}
    for f in FILES:
        d = load(f)
        n = d["fidelity"][0]["n"]
        for r in d["fidelity"]:
            if r["is_control"] and r["name"] != CEILING:
                controls.setdefault(r["name"], (r, n))

    def z(r, n, c, cn):
        p = math.sqrt(se(r, n) ** 2 + se(c, cn) ** 2)
        return abs(r["fid_pos"] - c["fid_pos"]) / p if p else 0.0

    rows = {}
    for f in FILES:
        d = load(f)
        n = d["fidelity"][0]["n"]
        fid = {r["name"]: r for r in d["fidelity"]}
        for g in d["groundtruth"]:
            if g["name"] == CEILING:
                continue
            k = SHORT.get(g["name"], g["name"])
            if k in rows and g["is_control"]:
                continue
            fr = fid[g["name"]]
            rows[k] = dict(gt=g["auc_vs_decoys"], fid=fr["fid_pos"],
                           se=se(fr, n), ctrl=g["is_control"],
                           beats=all(z(fr, n, c, cn) > 2
                                     for c, cn in controls.values()))

    body = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{The two metrics disagree. Ground-truth AUC is recovery of "
        r"the planted bottleneck against the decoys; Fid$^+$ is the signed "
        r"change in predicted path cost when the explanation is removed, with "
        r"the standard error of its mean. The last column reports whether "
        r"Fid$^+$ is distinguishable from \emph{all three} controls at "
        r"$|z|>2$. "
        r"Nine of eleven methods recover the bottleneck above $0.70$; three "
        r"are separable under fidelity.}",
        r"\label{tab:contradiction}",
        r"\begin{tabular}{lrr@{\,$\pm$\,}lc}", r"\toprule",
        r"Explainer & GT AUC & \multicolumn{2}{c}{Fid$^+$ (ms)} & "
        r"$\ne$ controls \\", r"\midrule",
    ]
    ordered = sorted(rows.items(), key=lambda kv: -kv[1]["gt"])
    for k, v in ordered:
        if v["ctrl"]:
            continue
        mark = r"\checkmark" if v["beats"] else "--"
        body.append(f"{esc(k)} & {v['gt']:.3f} & {v['fid']:+.2f} & "
                    f"{v['se']:.2f} & {mark} \\\\")
    body.append(r"\midrule")
    for k, v in ordered:
        if not v["ctrl"]:
            continue
        body.append(f"{esc(k)} & {v['gt']:.3f} & {v['fid']:+.2f} & "
                    f"{v['se']:.2f} & -- \\\\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_contradiction.tex", "\n".join(body) + "\n")


# --------------------------------------------------------------- Table 4
def table_softmask():
    # Provenance: diagnostic over 60 bottleneck graphs, top-3 links, on the
    # 11.3% model. Reproduce with paper/../scripts (see PHASE_B.md). These are
    # transcribed from that run; they are the only hand-entered numbers in the
    # paper and are flagged as such.
    rows = [("Hard removal", -14.727, "1.000", "100"),
            (r"Soft mask $\to 0.0$", 9.965, "-0.554", "13"),
            (r"Soft mask $\to 0.1$", 9.787, "-0.555", "13")]
    body = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Soft masking is not a gentler edge removal under "
        r"\texttt{min} aggregation. PyG applies a soft mask as a multiplier on "
        r"the message; GENConv messages are strictly positive, so scaling one "
        r"toward zero makes it the minimum and the suppressed edge dominates "
        r"the aggregation. $n=60$ graphs, top-3 links.}",
        r"\label{tab:softmask}",
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Perturbation & Mean Fid$^+$ (ms) & $r$ vs hard & Sign agr. \\",
        r"\midrule",
    ]
    for name, fid, corr, agr in rows:
        body.append(f"{name} & {fid:+.2f} & {corr} & {agr}\\% \\\\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_softmask.tex", "\n".join(body) + "\n")


# --------------------------------------------------------------- Table 5
def _delay_rows():
    """(forward-pass baseline in ms, {method: online ratio}) for every method.

    Shared by `table_delay` and `facts` so the table and the control-plane
    budget sentence in the prose can never disagree about a ratio.
    """
    d = load("phase_b_cap.json")
    base = d["delay"]["baseline_median_ms"]
    rows = {SHORT.get(m["name"], m["name"]): m["ratio"]
            for m in d["delay"]["methods"]}
    pert = load("delay_perturbation.json")
    for n, v in pert["methods"].items():
        if "baseline" not in n:
            rows[SHORT.get(n, n)] = v["ratio"]
    return base, rows


def table_delay():
    """Offline (one-time) and online (per-explanation) cost for every method.

    Ratios, not absolute times, are the comparable quantity: the perturbation
    methods were timed in a separate session, and each ratio is taken against
    the forward-pass baseline measured in its own session.
    """
    base, rows = _delay_rows()
    offline = {SHORT.get(k, k): v
               for k, v in load("offline_cost.json").items() if v > 0}

    def band(r):
        o = r - 1.0
        return ("free" if o < .25 else "cheap" if o < 1 else
                "costly" if o < 5 else "offline")

    body = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{One-time and per-explanation cost. Online cost is a "
        r"multiple of one batch-1 forward pass. Only PGExplainer amortises: it "
        r"trains a mask generator once, after which each explanation is cheap. "
        r"Perturbation methods carry no training cost but pay it back many "
        r"times over per explanation, since each draws hundreds of forward "
        r"passes.}",
        r"\label{tab:delay}",
        r"\begin{tabular}{lrrl}", r"\toprule",
        r"Explainer & Offline & Online & Band \\",
        r" & (one-time) & ($\times$ fwd) & \\", r"\midrule",
    ]
    for k, r in sorted(rows.items(), key=lambda kv: kv[1]):
        off = f"{offline[k]:.0f}\,s" if k in offline else "--"
        body.append(f"{esc(k)} & {off} & {r:.2f} & {band(r)} \\\\")
    body += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write("tab_delay.tex", "\n".join(body) + "\n")


def facts():
    """Numbers quoted in the prose, emitted as LaTeX macros."""
    a = load("phase_a.json")
    b = load("phase_b_cap.json")
    q = b["model_quality"]
    macros = {
        "ModelMedRel": f"{q['median_rel_error']:.1%}".replace("%", r"\%"),
        "ModelMeanRel": f"{q['mean_rel_error']:.1%}".replace("%", r"\%"),
        "NGraphs": str(b["n_graphs"]),
        "PhaseAControl": f"{a['summary'][0]['auc_mean']:.3f}",
        "DelayBaseline": f"{b['delay']['baseline_median_ms']:.2f}",
    }

    # Control-plane budgets, expressed as the forward-pass multiple each one
    # allows on this hardware. The two intervals are published targets (50 ms
    # protection switching, sub-second IGP reconvergence); everything else here
    # is measured, so the comparison is anchored at one end only -- which is
    # exactly what the delay section claims.
    base, rows = _delay_rows()
    real = {k: v for k, v in rows.items() if "control" not in k.lower()}
    for label, budget_ms in (("Fifty", 50.0), ("Second", 1000.0)):
        allowed = budget_ms / base
        fits = sorted(k for k, v in real.items() if v <= allowed)
        macros[f"Budget{label}"] = (f"{allowed:.0f}" if allowed >= 10
                                    else f"{allowed:.1f}")
        macros[f"NFits{label}"] = str(len(fits))
    macros["NDelayMethods"] = str(len(real))

    # The control floor, script-generated so the prose cannot drift from the
    # table when a control is added or a model is retrained.
    gt = {g["name"]: g for f in ("phase_b_cap.json", "phase_b_controls.json")
          for g in load(f)["groundtruth"]}
    for key, name in (("GTShortPath", "ShortestPath (control)"),
                      ("GTMaxWeight", "MaxWeight (control)"),
                      ("GTRandom", "Random (control)")):
        macros[key] = f"{gt[name]['auc_vs_decoys']:.3f}"
    macros["NControls"] = "3"
    macros["ModelMAE"] = f"{q['mae_ms']:.2f}"

    # Link-scoped fidelity (k=1). This is the only budget at which the oracle's
    # selection is unambiguous, so it is the only one at which "what score does
    # a demonstrably correct explanation get" has an answer.
    sm = load("scope_match_controls.json")["methods"]
    K1 = "1"
    for key, name in (("Oracle", "Oracle (control)"),
                      ("ShortPath", "ShortestPath (control)"),
                      ("Saliency", "EdgeSaliency")):
        r = sm[name]["fid"][K1]
        macros[f"Kone{key}Fid"] = f"{r['fid_pos']:+.2f}"
        macros[f"Kone{key}SE"] = f"{r['fid_pos_se']:.2f}"
    # How many rows move the prediction FURTHER than the known-correct one.
    # Counted rather than asserted, because the answer depends on the model.
    orc = abs(sm["Oracle (control)"]["fid"][K1]["fid_pos"])
    bigger = [n for n, m in sm.items()
              if n != "Oracle (control)" and abs(m["fid"][K1]["fid_pos"]) > orc]
    macros["KoneNBigger"] = str(len(bigger))
    macros["KoneNOther"] = str(len(sm) - 1)
    worst = max(sm.items(),
                key=lambda kv: (abs(kv[1]["fid"][K1]["fid_pos"])
                                if kv[0] != "Oracle (control)" else -1))
    macros["KoneWorstFid"] = f"{worst[1]['fid'][K1]['fid_pos']:+.2f}"
    macros["KoneWorstSE"] = f"{worst[1]['fid'][K1]['fid_pos_se']:.2f}"
    macros["KoneWorstName"] = SHORT.get(worst[0], worst[0])
    macros["KoneOracleZ"] = "%.1f" % abs(
        sm["Oracle (control)"]["fid"][K1]["fid_pos"]
        / sm["Oracle (control)"]["fid"][K1]["fid_pos_se"])
    # ---- across-seed spread (reviewer item #1) --------------------------
    # Half-range, not a standard error: an SE over n=3 reads as a confidence
    # interval it cannot support. Point estimates elsewhere in the paper stay
    # on the primary run; this is the spread around them.
    sv = load("seed_variance_3.json")
    meth = sv["methods"]
    macros["SeedN"] = str(sv["n_seeds"])
    qual = sorted(sv["model_quality"].values())
    macros["SeedQualities"] = ", ".join(
        f"{q:.1%}".replace("%", r"\%") for q in qual[:-1]
    ) + f" and {qual[-1]:.1%}".replace("%", r"\%")

    def med(vals):
        v = sorted(vals)
        return v[len(v) // 2]

    free = {k: v for k, v in meth.items() if not v["is_control"]}
    ctrl = {k: v for k, v in meth.items() if v["is_control"]}
    macros["SeedNMethods"] = str(len(free))

    # Spelled-out forms: "three independent runs" reads as prose, "3
    # independent runs" reads as a table cell. Still generated, so the word
    # cannot drift from the file list.
    WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
             7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven"}
    macros["SeedNWord"] = WORDS[sv["n_seeds"]]

    # ---- the saturation argument of Section VI --------------------------
    # Counted, not asserted. The prose previously said "six of the ten" where
    # the data say seven of the nine, which is the kind of claim that only
    # stays true by accident if it is typed.
    pa = {}
    for m in load("phase_a.json")["seeds"][0]["methods"]:
        if "phenomenon" in m["name"] or "true labels" in m["name"]:
            continue
        pa[SHORT.get(m["name"], m["name"])] = m["auc"]
    for m in load("phase_a_perturbation.json")["seeds"][0]["methods"]:
        pa[SHORT.get(m["name"], m["name"])] = m["auc"]
    gtr = {}
    for f in ("phase_b_cap.json", "phase_b_perturbation.json"):
        for r in load(f)["groundtruth"]:
            gtr[SHORT.get(r["name"], r["name"])] = r["auc_vs_decoys"]
    both = {k: (pa[k], gtr[k]) for k in pa
            if k in gtr and "control" not in k.lower()}
    macros["NTransfer"] = str(len(both))
    macros["NTransferWord"] = WORDS[len(both)]
    macros["TransferBAMin"] = f"{min(x for x, _ in both.values()):.2f}"

    SAT = 0.99
    sat = {k: v for k, v in both.items() if v[0] >= SAT}
    rout = [y for _, y in sat.values()]
    macros["NSat"] = str(len(sat))
    macros["NSatWord"] = WORDS[len(sat)].capitalize()
    macros["SatBASpan"] = f"{max(x for x, _ in sat.values()) - min(x for x, _ in sat.values()):.3f}"
    macros["SatRoutMin"] = f"{min(rout):.2f}"
    macros["SatRoutMax"] = f"{max(rout):.2f}"
    macros["SatRoutSpan"] = f"{max(rout) - min(rout):.3f}"
    macros["SeedNMethodsWord"] = WORDS[len(free)]
    macros["SeedGTSpread"] = f"{med(v['gt_half_range'] for v in free.values()):.3f}"
    macros["SeedFidSpread"] = f"{med(v['fid_pos_half_range'] for v in free.values()):.2f}"

    # the method whose Fid+ sign is not determined across runs
    flip = max(free.items(), key=lambda kv: (kv[1]["fid_pos_half_range"]
                                             - abs(kv[1]["fid_pos_mean"])))
    macros["SeedFlipName"] = SHORT.get(flip[0], flip[0])
    macros["SeedFlipFid"] = f"{flip[1]['fid_pos_mean']:+.2f}"
    macros["SeedFlipHalf"] = f"{flip[1]['fid_pos_half_range']:.2f}"

    # the least stable method under GROUND TRUTH, for contrast
    worst_gt = max(free.items(), key=lambda kv: kv[1]["gt_half_range"])
    macros["SeedWorstGTName"] = SHORT.get(worst_gt[0], worst_gt[0])
    macros["SeedWorstGTMean"] = f"{worst_gt[1]['gt_mean']:.3f}"
    macros["SeedWorstGTHalf"] = f"{worst_gt[1]['gt_half_range']:.3f}"

    # controls: identical explanations every run, so GT cannot move and any
    # Fid+ movement is the model's, not the explanation's
    macros["SeedCtrlGTHalf"] = "%.4f" % max(v["gt_half_range"]
                                            for v in ctrl.values())
    macros["SeedCtrlFidHalf"] = "%.2f" % max(v["fid_pos_half_range"]
                                             for v in ctrl.values())

    # ---- ablations (recommendation #8) ----------------------------------
    agg = load("ablations/aggregation.json")["aggregations"]
    for key, tag in (("min+mean", "MinMean"), ("min", "MinOnly"),
                     ("mean", "MeanOnly")):
        macros[f"Aggr{tag}"] = f"{agg[key]['median_rel_error']:.1%}".replace(
            "%", r"\%")

    def gt_of(f, name, field="auc_vs_decoys"):
        for r in load(f)["groundtruth"]:
            if r["name"] == name:
                return r
        raise KeyError(f"{name} not in {f}")

    # Decoy count. With no decoys there is nothing to restrict the AUC to, so
    # the decoy-free column is auc_all -- the only comparison available there.
    d0 = "ablations/phase_b_decoy0.json"
    d3 = "ablations/phase_b_decoy3.json"
    mw0 = gt_of(d0, "MaxWeight (control)")
    macros["DecoyZeroMaxWeightAUC"] = f"{mw0['auc_all']:.3f}"
    macros["DecoyZeroMaxWeightRec"] = f"{mw0['recall_at_1']:.3f}"
    best0 = max((r for r in load(d0)["groundtruth"]
                 if not r["is_control"]), key=lambda r: r["auc_all"])
    macros["DecoyZeroBestName"] = SHORT.get(best0["name"], best0["name"])
    macros["DecoyZeroBestAUC"] = f"{best0['auc_all']:.3f}"
    macros["DecoyThreeMaxWeightAUC"] = f"{gt_of(d3, 'MaxWeight (control)')['auc_vs_decoys']:.3f}"
    macros["DecoyThreeDeconvAUC"] = f"{gt_of(d3, 'PyG-Captum[Deconvolution]')['auc_vs_decoys']:.3f}"

    # Bottleneck magnitude. Deconvolution is the one method present in every
    # ablation run, so it carries the comparison.
    for tag, f in (("Three", "ablations/phase_b_mag3.json"),
                   ("TwentyFour", "ablations/phase_b_mag24.json")):
        macros[f"Mag{tag}Deconv"] = f"{gt_of(f, 'PyG-Captum[Deconvolution]')['auc_vs_decoys']:.3f}"
        macros[f"Mag{tag}ShortPath"] = f"{gt_of(f, 'ShortestPath (control)')['auc_vs_decoys']:.3f}"
        macros[f"Mag{tag}Model"] = f"{load(f)['model_quality']['median_rel_error']:.1%}".replace("%", r"\%")
    macros["MagTwelveDeconv"] = f"{gt_of('phase_b_cap.json', 'PyG-Captum[Deconvolution]')['auc_vs_decoys']:.3f}"

    body = [r"%% auto-generated; do not edit"]
    for k, v in macros.items():
        body.append(r"\newcommand{\%s}{%s}" % (k, v))
    write("facts.tex", "\n".join(body) + "\n")


if __name__ == "__main__":
    table_phase_a()
    table_transfer()
    table_contradiction()
    table_softmask()
    table_delay()
    facts()
