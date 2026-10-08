#!/usr/bin/env python3
"""Render explanatory SVGs for the True Online / IDBD / TIDBD blog draft.

Run: python3 scripts/render-idbd-tidbd-figures.py
Optional: --preview-dir /tmp/githubpages-image-review (also writes PNG previews).
Sources: Sutton & Barto (2018), chapter 12; Sutton (1992), equations 4–6;
Kearney et al. (2018), Algorithm 1. These are explanatory diagrams, not experiments.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import xml.etree.ElementTree as ET

os.environ.setdefault("MPLCONFIGDIR", "/tmp/githubpages-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "images"
INK = "#193348"
SOFT = "#557083"
BLUE = "#2769a8"
TEAL = "#087e75"
ORANGE = "#b65a20"
PURPLE = "#7059a6"
LINE = "#d6e1e9"

plt.rcParams.update({"mathtext.fontset": "stix", "svg.fonttype": "path"})
FONT = FontProperties(family=["sans-serif"])
for font_path in (
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
):
    if Path(font_path).is_file():
        FONT = FontProperties(fname=font_path)
        break


def canvas(height=6.7):
    fig, ax = plt.subplots(figsize=(14, height))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    return fig, ax


def label(ax, x, y, value, size=17, color=INK, align="center", weight="normal"):
    return ax.text(x, y, value, fontsize=size, color=color, ha=align,
                   va="center", fontproperties=FONT, weight=weight)


def math(ax, x, y, value, size=23, color=INK):
    return ax.text(x, y, f"${value}$", fontsize=size, color=color,
                   ha="center", va="center")


def card(ax, x, y, width, height, fill="#f5f8fb", edge=LINE):
    ax.add_patch(FancyBboxPatch((x, y), width, height,
                 boxstyle="round,pad=0.012,rounding_size=0.016",
                 facecolor=fill, edgecolor=edge, linewidth=1.2))


def arrow(ax, start, end, color=SOFT):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                 mutation_scale=17, color=color, linewidth=1.7))


def save(fig, name, title, description, preview_dir):
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / f"{name}.svg"
    fig.savefig(svg_path, format="svg", metadata={
        "Title": title, "Description": description, "Date": None,
        "Creator": "Yanlin Li blog — explanatory diagram, AI-assisted",
    })
    # Give the embedded SVG a title/description in addition to Markdown alt text.
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    document = ET.parse(svg_path)
    root = document.getroot()
    root.set("role", "img")
    root.set("aria-labelledby", "figure-title figure-description")
    title_node = ET.Element(f"{{{ns}}}title", {"id": "figure-title"})
    title_node.text = title
    desc_node = ET.Element(f"{{{ns}}}desc", {"id": "figure-description"})
    desc_node.text = description
    root.insert(0, title_node)
    root.insert(1, desc_node)
    document.write(svg_path, encoding="utf-8", xml_declaration=True)
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"{name}.png", dpi=140)
    plt.close(fig)
    print(svg_path.relative_to(ROOT))


def dutch_coordinate_comparison(preview_dir):
    """Plot the draft's hand calculation, without reusing the old geometry image."""
    fig, ax = canvas(7.8)
    label(ax, .5, .94, "Dutch trace 的二维手算：究竟是哪一个坐标发生变化？", 23)
    math(ax, .5, .861,
         r"\mathbf{x}_t=(1,0)^{\top},\quad\mathbf{z}_{t-1}=(0.6,0.4)^{\top},\quad\alpha=0.5,\quad\gamma\lambda=0.8", 25)

    # Use the same inputs and updates as section 2.6, rather than illustrative curves.
    feature = np.array([1.0, 0.0])
    old_trace = np.array([.6, .4])
    matrix = np.eye(2) - .5 * np.outer(feature, feature)
    adjusted = matrix @ old_trace
    accumulating = feature + .8 * old_trace
    dutch = feature + .8 * adjusted
    assert np.allclose(adjusted, [.3, .4])
    assert np.allclose(dutch, [1.24, .32])
    assert np.allclose(accumulating, [1.48, .32])

    for position, title, first, second, names in (
        (.06, r"1　只观察 $\mathbf{M}_t$ 的作用", old_trace, adjusted,
         (r"$\mathbf{z}_{t-1}$", r"$\mathbf{M}_t\mathbf{z}_{t-1}$")),
        (.555, "2　完成衰减与新增资格", accumulating, dutch,
         ("Accumulating", "Dutch")),
    ):
        label(ax, position+.19, .77, title, 19)
        plot = fig.add_axes([position, .285, .385, .42])
        coordinates = np.arange(2)
        width = .27
        plot.bar(coordinates-width/2, first, width, color="#8c9dac", label=names[0])
        plot.bar(coordinates+width/2, second, width, color=TEAL, label=names[1])
        for values, offset, color in ((first, -width/2, SOFT), (second, width/2, TEAL)):
            for coordinate, value in zip(coordinates, values):
                plot.text(coordinate+offset, value+.045, f"{value:.2f}",
                          ha="center", color=color, fontsize=17)
        plot.set_xticks(coordinates)
        plot.set_xticklabels(["第一坐标：平行于当前特征", "第二坐标：与当前特征正交"],
                            fontproperties=FONT, fontsize=12)
        plot.set_ylim(0, 1.75)
        plot.set_yticks([0, .4, .8, 1.2, 1.6])
        plot.tick_params(axis="y", labelsize=12, colors=SOFT)
        plot.tick_params(axis="x", length=0, pad=10)
        plot.set_axisbelow(True)
        plot.grid(axis="y", color=LINE, linewidth=.8)
        plot.spines[["top", "right"]].set_visible(False)
        plot.spines[["left", "bottom"]].set_color(LINE)
        plot.legend(loc="upper right", frameon=False, fontsize=14)

    math(ax, .25, .195, r"\mathbf{M}_t=\mathrm{diag}(0.5,1)", 24)
    math(ax, .75, .195,
         r"\mathbf{z}_t^{\mathrm{acc}}-\mathbf{z}_t=(0.24,0)^{\top}", 24)
    label(ax, .5, .105, "左图：垂直分量通过矩阵后不变。右图：完整 trace 仍对它施加时间衰减。", 16, SOFT)
    label(ax, .5, .049, "数值来自本文二维手算，不是实验结果；两幅图使用相同的纵轴刻度。", 15, SOFT)
    save(fig, "dutch-trace-two-coordinate-comparison", "Dutch trace 的二维坐标数值对照",
         "当前特征为 (1,0)，旧资格为 (0.6,0.4)，alpha=0.5，gamma lambda=0.8。"
         "左侧比较旧资格和矩阵调整后的 (0.3,0.4)；右侧比较 accumulating trace (1.48,0.32)"
         "和 Dutch trace (1.24,0.32)。最终只有第一坐标相差 0.24，第二坐标均为 0.32。"
         "这是同一纵轴刻度下的手算数值对照，不是实验。", preview_dir)


def prediction_drift(preview_dir):
    fig, ax = canvas()
    label(ax, .5, .93, r"$Q_{\mathrm{old}}$ correction：同一个预测，两个参数版本", 24)
    label(ax, .5, .85, "固定当前状态、动作与特征，只比较参数更新前后", 17, SOFT)
    math(ax, .5, .775, r"\mathbf{x}_t=\mathbf{x}(S_t,A_t)", 24)
    card(ax, .055, .40, .39, .285)
    card(ax, .555, .40, .39, .285, "#eff9f6")
    label(ax, .25, .637, "上一时间步作为 next-Q 保存", 17, BLUE)
    math(ax, .25, .55, r"Q_{\mathrm{old}}=\mathbf{w}_{t-1}^{\top}\mathbf{x}_t", 22)
    math(ax, .25, .45, r"Q_{\mathrm{old}}=0.50", 29, BLUE)
    label(ax, .75, .637, "当前时间步重新计算", 17, TEAL)
    math(ax, .75, .55, r"Q_t=\mathbf{w}_t^{\top}\mathbf{x}_t", 24)
    math(ax, .75, .45, r"Q_t=0.69", 29, TEAL)
    arrow(ax, (.455, .55), (.545, .55))
    label(ax, .50, .635, "更新参数", 13, SOFT)
    math(ax, .5, .32, r"Q_t-Q_{\mathrm{old}}=(\mathbf{w}_t-\mathbf{w}_{t-1})^{\top}\mathbf{x}_t=0.19", 25)
    card(ax, .12, .115, .76, .12, "#f5f1fb")
    math(ax, .5, .175, r"\Delta\mathbf{w}_{\mathrm{corr}}=\alpha(Q_t-Q_{\mathrm{old}})(\mathbf{z}_t-\mathbf{x}_t)", 26, PURPLE)
    label(ax, .5, .055, r"比较的是同一个 $(S_t,A_t)$；不是当前状态与下一状态的差。", 15, SOFT)
    save(fig, "true-online-q-old-prediction-drift",
         "Qold correction 的参数版本比较",
         "对于 t 大于等于 1，固定当前特征 x_t。上一时间步用 w_(t-1) 得到 Qold=0.50；"
         "参数更新后，当前时间步用 w_t 得到 Qt=0.69。漂移 0.19 沿 alpha(z_t-x_t) 修正。",
         preview_dir)


def idbd_signal(preview_dir):
    fig, ax = canvas(7.4)
    label(ax, .5, .93, "IDBD：过去的步长选择与当前更新方向是否一致？", 23)
    math(ax, .5, .847, r"h_{i,t}\approx\frac{\partial w_{i,t}}{\partial\beta_i},\qquad\alpha_i=e^{\beta_i}", 26)
    label(ax, .5, .77, r"以下两幅图均假设 $h_{i,t}>0$：过去步长稍大，会让当前权重更大。", 16, SOFT)
    for center, direction, color, heading, sign in (
        (.255, 1, TEAL, "同向：调大学习率", ">0"),
        (.745, -1, ORANGE, "反向：调小学习率", "<0"),
    ):
        card(ax, center-.208, .245, .416, .455,
             "#eff9f6" if direction > 0 else "#fff6ee")
        label(ax, center, .647, heading, 20, color)
        math(ax, center, .568, r"\delta_t x_{i,t}" + (">0" if direction > 0 else "<0"), 26, color)
        label(ax, center-.115, .488, "过去敏感度", 14, SOFT)
        arrow(ax, (center-.047, .488), (center+.14, .488), PURPLE)
        label(ax, center-.115, .422, "当前更新方向", 14, SOFT)
        start, end = ((center-.047, .422), (center+.14, .422)) if direction > 0 else ((center+.14, .422), (center-.047, .422))
        arrow(ax, start, end, color)
        math(ax, center, .34, r"\delta_t x_{i,t}h_{i,t}"+sign, 27, color)
        math(ax, center, .275, r"\Delta\beta_i"+sign+r"\quad\Longrightarrow\quad\alpha_i"+(r"\uparrow" if direction > 0 else r"\downarrow"), 25, color)
    math(ax, .5, .163, r"\frac{\partial L_t}{\partial\beta_i}\approx-\delta_t x_{i,t}h_{i,t}\quad\Longrightarrow\quad\Delta\beta_i=+\theta\delta_t x_{i,t}h_{i,t}", 26)
    label(ax, .5, .076, r"调节信号是 $\delta_t x_{i,t}h_{i,t}$ 的乘积；$h_{i,t}$ 单独不能判断步长是否合适。", 15, SOFT)
    label(ax, .5, .034, "一次反向信号也可能来自噪声或目标变化，不能直接判定为 overshoot。", 14, SOFT)
    save(fig, "idbd-meta-update-direction", "IDBD 的步长适应信号",
         "在 h_i,t 大于零的示例中，当前更新方向 delta_t x_i,t 与过去敏感度同向时，"
         "乘积为正，增加 beta_i 和 alpha_i；反向时乘积为负，减小 beta_i 和 alpha_i。"
         "meta-gradient 是负的乘积，梯度下降因此使用正号更新。", preview_dir)


def tidbd_roles(preview_dir):
    fig, ax = canvas(8.6)
    label(ax, .5, .94, "TIDBD(λ)：x、z、h 分别进入哪条更新？", 24)
    specs = [
        (.055, BLUE, r"x_{i,t}", "当前预测使用谁？",
         r"\frac{\partial\hat v(S_t,\mathbf{w}_t)}{\partial w_i}=x_{i,t}"),
        (.365, TEAL, r"z_{i,t}", "历史参数有多少资格？",
         r"z_{i,t}=\gamma\lambda z_{i,t-1}+x_{i,t}"),
        (.675, PURPLE, r"h_{i,t}", "过去步长怎样影响权重？",
         r"h_{i,t}\approx\frac{\partial w_{i,t}}{\partial\beta_i}"),
    ]
    for x, color, symbol, meaning, formula in specs:
        card(ax, x, .675, .27, .195)
        math(ax, x+.135, .831, symbol, 29, color)
        label(ax, x+.135, .768, meaning, 15, color)
        math(ax, x+.135, .712, formula, 19)
    card(ax, .11, .43, .78, .17, "#eff9f6")
    label(ax, .5, .56, "权重更新：用资格迹分配 temporal credit", 18, TEAL)
    math(ax, .5, .486, r"\Delta w_i=\alpha_i\delta_t z_{i,t}", 30, TEAL)
    arrow(ax, (.50, .663), (.50, .608), TEAL)
    card(ax, .11, .21, .78, .17, "#f5f1fb")
    label(ax, .5, .34, "步长更新：用当前特征 × 过去敏感度", 18, PURPLE)
    math(ax, .5, .267, r"\Delta\beta_i=\theta\delta_t x_{i,t}h_{i,t},\qquad\alpha_i=e^{\beta_i}", 28, PURPLE)
    # x and h feed the direct meta-update, without passing through the z lane.
    for x, color in ((.19, BLUE), (.81, PURPLE)):
        ax.plot([x, x], [.663, .63], color=color, linewidth=1.6)
        ax.plot([x, .075 if x < .5 else .925], [.63, .63], color=color, linewidth=1.6)
        side = .075 if x < .5 else .925
        ax.plot([side, side], [.63, .295], color=color, linewidth=1.6)
        arrow(ax, (side, .295), (.10 if x < .5 else .90, .295), color)
    math(ax, .5, .126, r"h_{i,t+1}=h_{i,t}[1-\alpha_i x_{i,t}z_{i,t}]^{+}+\alpha_i\delta_t z_{i,t}", 27)
    label(ax, .5, .052, r"$z_{i,t}$ 通过 $h_{i,t+1}$ 影响未来的步长更新；它不直接替代 meta-update 中的 $x_{i,t}$。", 15, SOFT)
    save(fig, "tidbd-x-z-h-update-paths", "TIDBD 中 x、z、h 的不同作用",
         "当前特征 x 给出预测对权重的导数；资格迹 z 用于权重更新；"
         "x 与步长敏感度 h 一起用于 beta 更新。z 还进入下一步 h 的递推，间接影响未来 beta。"
         "执行时先更新 beta 和 alpha，再用这一轮 alpha 更新权重和敏感度。", preview_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    dutch_coordinate_comparison(args.preview_dir)
    prediction_drift(args.preview_dir)
    idbd_signal(args.preview_dir)
    tidbd_roles(args.preview_dir)


if __name__ == "__main__":
    main()
