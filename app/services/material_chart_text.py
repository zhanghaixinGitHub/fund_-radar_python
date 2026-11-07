"""将已核对坐标轴的曲线转换为有业务单位的区间表，不猜测原始日净值。

仅适用于本试点的蓝色基金/品红基准曲线。源图没有精确日频点值时，保存
可见曲线对应的时间及百分比区间、源图遮挡情况和所有坐标/图例文字。
这些区间不能冒充正式净值；未知颜色、轴不一致和覆盖不全都拒绝完成。
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta


def digitise(image, ocr, *, dates, y_labels, series):
    """返回全部可见曲线的区间记录和校验结果，坐标必须先人工逐项核对。

    image为与OCR相同裁切/倍率的RGB数组。dates是完整横轴标签，y_labels
    是完整纵轴百分数原字符串。数值以区间表示源图分辨率，不补造遮挡值。
    """
    import numpy as np

    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] * image.shape[1] > 16_000_000:
        raise ValueError("CHART_IMAGE_SHAPE_OR_BUDGET")
    height, width = image.shape[:2]
    rgb = image.astype(np.int16)
    grey = (rgb.max(axis=2) - rgb.min(axis=2) < 18) & (rgb.mean(axis=2) < 245)
    anchors_x, anchors_y = [], []
    allowed_y = {s.replace(" ", "") for s in y_labels}
    for line in ocr["lines"]:
        text = re.sub(r"\s+", "", line["text"])
        box = np.asarray(line["box"], dtype=float)
        x, y = box.mean(axis=0)
        if text in dates:
            anchors_x.append((float(x), datetime.fromisoformat(text).toordinal()))
        if text in allowed_y and x < width * 0.2:
            anchors_y.append((float(y), float(text[:-1])))
    if len(anchors_x) != len(dates) or len(anchors_y) < len(y_labels) - 2 or len(anchors_y) < 5:
        raise ValueError("CHART_AXIS_ANCHORS_INCOMPLETE")
    anchors_x.sort()
    # OCR框是字形范围，百分号与负号会改变框中心。用标签附近实际网格线
    # 绑定刻度位置，再独立检查线性一致；不能直接把文字框中心当数据刻度。
    grid_score = grey[:, round(anchors_x[0][0]) + 8 : round(anchors_x[-1][0]) - 8].mean(axis=1)
    snapped = []
    for y, value in anchors_y:
        low, high = max(0, round(y - height * 0.012)), min(height, round(y + height * 0.012))
        positions = np.flatnonzero(grid_score[low:high] >= 0.1) + low
        if not len(positions):
            raise ValueError("CHART_TICK_GRID_NOT_CONFIRMED")
        # 顶端外框可能比刻度虚线更深；取最近的有效线，不能误绑定外框。
        best = int(positions[np.argmin(np.abs(positions - y))])
        snapped.append((float(best), value))
    anchors_y = snapped
    x_positions, x_dates = zip(*anchors_x, strict=True)
    y_positions, y_values = zip(*anchors_y, strict=True)
    ax, bx = np.polyfit(x_positions, x_dates, 1)
    ay, by = np.polyfit(y_positions, y_values, 1)
    x_residual = float(max(abs(ax * x + bx - d) for x, d in anchors_x))
    y_residual = float(max(abs(ay * y + by - v) for y, v in anchors_y))
    # 残差按图面分辨率衡量，不把0.15个百分点硬套到跨度不同的图。
    # 两个渲染采样宽度内的校准不确定性完整进入数值区间，不声称精确原值。
    if ax <= 0 or ay >= 0 or x_residual > abs(ax) * 2 or y_residual > abs(ay) * 2:
        raise ValueError("CHART_AXIS_CALIBRATION_DISAGREEMENT")
    x0, x1 = round(min(x_positions)), round(max(x_positions))
    # 从灰色长横边线确定绘图区；不能把图例和轴标签中的彩色笔画混入曲线。
    long_rows = np.flatnonzero(grey[:, x0 : x1 + 1].mean(axis=1) > 0.65)
    long_rows = long_rows[long_rows < min(height * 0.9, max(y_positions) + height * 0.08)]
    if len(long_rows) < 2:
        raise ValueError("CHART_PLOT_BORDER_NOT_CONFIRMED")
    top, bottom = int(long_rows.min()), int(long_rows.max())
    if not top - 2 <= min(y_positions) < max(y_positions) < bottom + 8:
        raise ValueError("CHART_PLOT_AXIS_BOUNDARY_DISAGREEMENT")
    long_columns = np.flatnonzero(grey[top : bottom + 1].mean(axis=0) > 0.65)
    left = long_columns[np.abs(long_columns - min(x_positions)) < width * 0.04]
    right = long_columns[np.abs(long_columns - max(x_positions)) < width * 0.04]
    if not len(left) or not len(right):
        raise ValueError("CHART_VERTICAL_PLOT_BORDER_NOT_CONFIRMED")
    x0 = int(left[np.argmin(np.abs(left - min(x_positions)))])
    x1 = int(right[np.argmin(np.abs(right - max(x_positions)))])
    masks = {
        "fund": (rgb[:, :, 2] - rgb[:, :, 0] > 15) & (rgb[:, :, 2] - rgb[:, :, 1] > 15),
        "benchmark": (rgb[:, :, 0] - rgb[:, :, 1] > 25)
        & (rgb[:, :, 2] - rgb[:, :, 1] > 25)
        & (np.abs(rgb[:, :, 0] - rgb[:, :, 2]) < 90),
    }
    output = []
    quantum_days = float(ax * 2 + x_residual)
    quantum_pct = float(abs(ay) * 2 + y_residual)

    def iso(ordinal):
        return (datetime.fromordinal(int(ordinal)) + timedelta(days=ordinal % 1)).isoformat(timespec="seconds")

    for key, mask in masks.items():
        samples = []
        observed = int(mask[top : bottom + 1, x0 : x1 + 1].sum())
        if observed < 100:
            raise ValueError("CHART_SERIES_NOT_FOUND")
        captured = 0
        for x in range(x0, x1 + 1):
            ys = np.flatnonzero(mask[top : bottom + 1, x]) + top
            captured += len(ys)
            groups = np.split(ys, np.flatnonzero(np.diff(ys) > 1) + 1) if len(ys) else []
            values = [
                [
                    round(float(ay * (g[-1] + 1) + by - y_residual), 6),
                    round(float(ay * (g[0] - 1) + by + y_residual), 6),
                ]
                for g in groups
                if len(g)
            ]
            first, last = min(x_dates), max(x_dates)
            lower = min(last, max(first, ax * (x - 1) + bx - x_residual))
            upper = min(last, max(first, ax * (x + 1) + bx + x_residual))
            samples.append(
                {
                    "time_from": iso(lower),
                    "time_to": iso(upper),
                    "visible_value_ranges_pct": values,
                    "visibility": "VISIBLE" if values else "NOT_VISIBLE_IN_SOURCE",
                }
            )
        if captured != observed:
            raise ValueError("CHART_VISIBLE_CURVE_COVERAGE_FAILED")
        output.append(
            {
                "name": series[key],
                "unit": "%",
                "samples": samples,
                "visible_strokes_covered": True,
                "source_absence_rule": "源图未显示或被另一条线遮挡时保留未知，不插值补线",
            }
        )
    return {
        "kind": "SOURCE_CHART_VISIBLE_INTERVALS",
        "data": output,
        "axes": {
            "x": {"labels": dates, "unit": "日期", "scale": "LINEAR_CALENDAR"},
            "y": {"labels": y_labels, "unit": "%", "scale": "LINEAR"},
        },
        "series": series,
        "source_precision": "图面可见信息的区间提取，不是原始每日净值或精确收益率",
        "date_uncertainty_days": quantum_days,
        "value_uncertainty_percentage_points": quantum_pct,
        "calibration": {"max_date_residual_days": x_residual, "max_value_residual_percentage_points": y_residual},
        "semantic_review_required": True,
    }
