"""Render em_bench_grid.png — 2×2 heatmap of EM rates.

Rows: Turner (8 Betley prompts, verbatim) / Iceberg (ours, 64 prompts)
Cols: dose 25% / dose 100%
Cells: EM rate % at each combination.
"""
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent


def load_cell(set_name: str, dose: int) -> dict | None:
    for fn in sorted(glob.glob(str(SCRIPT_DIR / f"batches/em_bench_{set_name}_dose{dose}_*.json"))):
        with open(fn) as f:
            d = json.load(f)
        s = d.get("summary", {})
        if s.get("set_name") == set_name and s.get("dose") == dose and s.get("samples_per") == 100:
            return s
    return None


def main():
    configs = [
        ("turner",       "Turner\n(8 Betley prompts, verbatim)"),
        ("iceberg_best", "Iceberg\n(64 optimized prompts, ours)"),
    ]
    doses = [25, 100]

    grid = np.zeros((len(configs), len(doses)))
    labels = np.empty((len(configs), len(doses)), dtype=object)
    for i, (name, _) in enumerate(configs):
        for j, dose in enumerate(doses):
            s = load_cell(name, dose)
            if s is None:
                grid[i, j] = np.nan
                labels[i, j] = "—"
                continue
            rate = s["em_response_rate"] * 100
            n = s["n_samples_total"]
            em = s["em_response_count"]
            grid[i, j] = rate
            labels[i, j] = f"{rate:.2f}%\n({em}/{n})"

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    # Use viridis; clip vmax so low values are still visible
    vmax = float(np.nanmax(grid)) if np.any(~np.isnan(grid)) else 1.0
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")

    # Value labels in cells
    for i in range(len(configs)):
        for j in range(len(doses)):
            val = grid[i, j]
            # Contrast: use white on dark cells, black on light
            text_color = "white" if val < vmax * 0.55 else "black"
            ax.text(j, i, labels[i, j], ha="center", va="center",
                    color=text_color, fontsize=13, weight="bold")

    ax.set_xticks(range(len(doses)))
    ax.set_xticklabels([f"dose {d}%" for d in doses], fontsize=11)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels([c[1] for c in configs], fontsize=10)
    ax.set_title(
        "EM rate (alignment<30 AND coherency>50)\n"
        "N=100 samples/prompt, temp=1.0, GPT-4o-mini judge, Llama 3.2-3B medical finetune",
        fontsize=11, weight="bold", pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("EM response rate (%)", fontsize=10)

    plt.tight_layout()
    out = SCRIPT_DIR / "em_bench_grid.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote {out}")

    # Also copy into post_materials for the post
    import shutil
    dest = SCRIPT_DIR / "post_materials" / "em_bench_grid.png"
    shutil.copy(out, dest)
    print(f"Copied → {dest}")


if __name__ == "__main__":
    main()
