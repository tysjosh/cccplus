"""Table and figure emission, following the paper's layouts (Sec. 5).

Every table the paper leaves as TBD has a builder here. Tables are written as both
Markdown (for reading) and LaTeX (for dropping into the paper), and each carries
the manifest hash of the run that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .stats import Estimate

PRETTY_RULE = {
    "representational": "Representational",
    "round_trip_only": "Round trip only",
    "forward_only": "Forward only",
    "scalar_ccc": "Original scalar CCC",
    "ccc_ind": "CCC-Ind",
    "ccc_shape": "CCC-Shape",
    "ccc_sig": "CCC-Sig",
    "pairwise_ccc_plus": "Pairwise CCC+",
    "multi_model_ccc_plus": "Multi-model CCC+",
}

PRETTY_TRANSLATOR = {
    "procrustes": "Procrustes",
    "crosscoder": "Crosscoder",
    "dfc": "DFC",
    "mas": "MAS",
    "latent_stitch": "Latent Stitch",
    "stitch": "Stitch",
    "procrustes_shuffled": "Procrustes (shuffled ctrl)",
    "crosscoder_shuffled": "Crosscoder (shuffled ctrl)",
    "dfc_shuffled": "DFC (shuffled ctrl)",
}


@dataclass
class Table:
    """A rendered table with a caption and provenance."""

    name: str
    header: List[str]
    rows: List[List[str]]
    caption: str = ""
    notes: str = ""

    # ------------------------------------------------------------- rendering
    def markdown(self) -> str:
        widths = [len(h) for h in self.header]
        for r in self.rows:
            for i, c in enumerate(r):
                widths[i] = max(widths[i], len(str(c)))
        def line(cells: Sequence[str]) -> str:
            return "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(cells)) + " |"
        out = [line(self.header), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
        out += [line(r) for r in self.rows]
        body = "\n".join(out)
        head = f"**{self.name}**"
        if self.caption:
            head += f"\n\n{self.caption}"
        tail = f"\n\n{self.notes}" if self.notes else ""
        return f"{head}\n\n{body}{tail}\n"

    def latex(self) -> str:
        spec = "l" + "r" * (len(self.header) - 1)
        esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%")
        lines = [
            r"\begin{table}[t]", r"\centering", r"\small",
            rf"\begin{{tabular}}{{{spec}}}", r"\toprule",
            " & ".join(esc(h) for h in self.header) + r" \\", r"\midrule",
        ]
        lines += [" & ".join(esc(c) for c in r) + r" \\" for r in self.rows]
        lines += [r"\bottomrule", r"\end{tabular}",
                  rf"\caption{{{esc(self.caption)}}}", rf"\label{{tab:{self.name.lower().replace(' ', '_')}}}",
                  r"\end{table}"]
        return "\n".join(lines)


def fmt(e: Optional[Estimate], digits: int = 3, pct: bool = False) -> str:
    if e is None or not np.isfinite(e.value):
        return "n/a"
    scale = 100.0 if pct else 1.0
    suffix = "%" if pct else ""
    if np.isfinite(e.lo) and np.isfinite(e.hi):
        return f"{e.value*scale:.{digits}f}{suffix} [{e.lo*scale:.{digits}f}, {e.hi*scale:.{digits}f}]"
    return f"{e.value*scale:.{digits}f}{suffix}"


def fmt_num(v: float, digits: int = 3) -> str:
    return "n/a" if v is None or not np.isfinite(v) else f"{v:.{digits}f}"


# --------------------------------------------------------------------- tables
def rule_table(
    name: str,
    per_rule: Dict[str, Dict[str, Estimate]],
    rule_order: Sequence[str],
    caption: str,
    notes: str = "",
) -> Table:
    """Ret / Cov / FA / AUROC by validation rule (Tables 5.1 and 5.3 layout)."""
    rows = []
    for r in rule_order:
        if r not in per_rule:
            continue
        d = per_rule[r]
        rows.append([
            PRETTY_RULE.get(r, r),
            fmt(d.get("retention")),
            fmt(d.get("coverage")),
            fmt(d.get("false_acceptance")),
            fmt(d.get("auroc")),
            fmt(d.get("fa_at_matched_retention")),
        ])
    return Table(
        name=name,
        header=["Validation rule", "Ret.", "Cov.", "FA", "AUROC", "FA @ matched Ret."],
        rows=rows, caption=caption, notes=notes,
    )


def contrast_table(name: str, contrasts: Sequence[Dict[str, object]], caption: str) -> Table:
    rows = []
    for c in contrasts:
        e: Estimate = c["estimate"]  # type: ignore
        rows.append([
            f"{PRETTY_RULE.get(str(c['a']), c['a'])} - {PRETTY_RULE.get(str(c['b']), c['b'])}",
            fmt(e),
            "n/a" if not np.isfinite(e.p) else f"{e.p:.4f}",
            "n/a" if not np.isfinite(e.p_adjusted) else f"{e.p_adjusted:.4f}",
        ])
    return Table(
        name=name,
        header=["Contrast (FA @ matched retention)", "Estimate 95% CI", "p", "p (Holm)"],
        rows=rows, caption=caption,
    )


def translator_table(
    name: str, per_translator: Dict[str, Dict[str, Estimate]], order: Sequence[str], caption: str,
    columns: Sequence[str] = ("pairwise", "multi", "coverage", "abstention"),
    header: Optional[Sequence[str]] = None,
) -> Table:
    rows = []
    for t in order:
        if t not in per_translator:
            continue
        d = per_translator[t]
        rows.append([PRETTY_TRANSLATOR.get(t, t)] + [fmt(d.get(c)) for c in columns])
    return Table(
        name=name,
        header=list(header or (["Translator"] + [c.replace("_", " ").title() for c in columns])),
        rows=rows, caption=caption,
    )


def signature_table(name: str, per_translator: Dict[str, Dict[str, float]], order: Sequence[str],
                    caption: str) -> Table:
    """Prompt-level causal agreement (Table 5.4 layout)."""
    rows = []
    for t in order:
        if t not in per_translator:
            continue
        d = per_translator[t]
        rows.append([
            PRETTY_TRANSLATOR.get(t, t),
            fmt_num(d.get("effect_correlation", np.nan)),
            fmt_num(d.get("sign_agreement", np.nan)),
            fmt_num(d.get("effect_ratio", np.nan)),
            fmt_num(d.get("worst_setting_error", np.nan)),
            fmt_num(d.get("calibration_slope", np.nan)),
        ])
    return Table(
        name=name,
        header=["Translator", "Corr.", "Sign", "Effect ratio", "Worst error", "Calib. slope"],
        rows=rows, caption=caption,
    )


def coupling_table(name: str, stats: Dict[str, Dict[str, float]], caption: str) -> Table:
    """Coupled vs independent return legs: the translator-independence evidence."""
    rows = []
    for key, d in stats.items():
        rows.append([
            key,
            fmt_num(d.get("coupled_pass_rate", np.nan)),
            fmt_num(d.get("independent_pass_rate", np.nan)),
            fmt_num(d.get("coupled_pass_rate_negatives", np.nan)),
            fmt_num(d.get("independent_pass_rate_negatives", np.nan)),
            fmt_num(d.get("coupled_subspace_cosine", np.nan)),
            fmt_num(d.get("independent_subspace_cosine", np.nan)),
        ])
    return Table(
        name=name,
        header=["Translator", "Coupled pass (pos)", "Indep. pass (pos)",
                "Coupled pass (neg)", "Indep. pass (neg)",
                "Coupled ret. cos", "Indep. ret. cos"],
        rows=rows, caption=caption,
        notes="A coupled return that passes on negatives as readily as on positives "
              "carries no evidence: recovery is automatic by construction.",
    )


def benchmark_table(name: str, summary: Dict[str, Dict[str, dict]], caption: str) -> Table:
    """SIIT quality of the planted benchmark: is the ground truth actually planted?"""
    rows = []
    for prog, insts in summary.items():
        for inst, d in insts.items():
            rows.append([
                prog, inst, str(d.get("arch", "")), str(d.get("structure", "")),
                fmt_num(d.get("clean_accuracy", np.nan)),
                fmt_num(d.get("min_iit_accuracy", np.nan)),
                fmt_num(d.get("mean_strict_accuracy", np.nan)),
                "yes" if d.get("meets_gate") else "NO",
            ])
    return Table(
        name=name,
        header=["Program", "Instance", "Arch", "Structure", "Clean acc.", "min IIT acc.",
                "Strict acc.", "Gate"],
        rows=rows, caption=caption,
    )


def set_valued_table(name: str, stats: Dict[str, Dict[str, float]], caption: str) -> Table:
    rows = []
    for case, d in stats.items():
        rows.append([
            case,
            str(int(d.get("n", 0))),
            str(int(d.get("n_contexts", 0))),
            fmt_num(d.get("subspace_recall", np.nan)),
            fmt_num(d.get("subspace_precision", np.nan)),
            fmt_num(d.get("set_exact", np.nan)),
            fmt_num(d.get("joint_signature_pass", np.nan)),
            fmt_num(d.get("component_pass", np.nan)),
        ])
    return Table(
        name=name,
        header=["Planted case", "n", "Contexts", "Subspace recall", "Subspace precision",
                "Set exact", "Joint signature pass", "Component-level pass"],
        rows=rows, caption=caption,
    )


def multisite_table(name: str, rows_in: Dict[str, Dict[str, float]], caption: str) -> Table:
    rows = []
    for k, d in rows_in.items():
        rows.append([
            k,
            fmt_num(d.get("node", np.nan)),
            "—" if not np.isfinite(d.get("edge", np.nan)) else fmt_num(d.get("edge", np.nan)),
            "—" if not np.isfinite(d.get("interaction_error", np.nan)) else fmt_num(d.get("interaction_error", np.nan)),
        ])
    return Table(
        name=name, header=["IOI extension", "Node", "Edge", "Interact. err."],
        rows=rows, caption=caption,
    )


# -------------------------------------------------------------------- writing
def write_tables(tables: Sequence[Table], out_dir: Path, manifest_hash: str, title: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    md = [f"# {title}", "", f"manifest `{manifest_hash}`", ""]
    for t in tables:
        md.append(t.markdown())
        md.append("")
        (out_dir / f"{t.name}.tex").write_text(t.latex())
    path = out_dir / f"{title.lower().replace(' ', '_')}.md"
    path.write_text("\n".join(md))
    return path
