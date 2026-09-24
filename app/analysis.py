"""题目质量分析：用已结束会话的历史作答复盘单题表现。

只统计 ``status = 'finished'`` 的会话；作答对错一律使用作答时写入
``responses.score`` 的不可变分数，因此题目事后被批量导入修改答案
（``items.answer`` 更新）不会改写历史结论。

样本构造：对每个「答过目标题且至少另答一题」的已结束会话取一个样本点

* ``x`` = 该场目标题的得分（0/1）；
* ``y`` = 该场**其余题**的答对比例（其余题得分之和 / 其余题数）。

返回有效样本数、目标题正确率（``x`` 的均值）以及 ``x`` 与 ``y`` 的
Pearson 相关系数（即点二列相关的连续化形式，用于观察题目得分与
同场整体表现是否同向）。样本少于 ``MIN_SAMPLES`` 或任一序列无方差
（全对/全错、或其余题比例完全相同）时，相关系数没有定义，返回
``None``。

整个模块只读：不触碰会话状态、曝光计数与题库。
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass

# 计算相关系数的最小样本量：小样本的 Pearson 系数没有解释力。
MIN_SAMPLES = 5


@dataclass
class ItemAnalysis:
    """单题复盘结果；``correct_rate`` / ``pearson_r`` 可能为 ``None``。"""

    item_id: str
    sample_size: int
    correct_rate: float | None
    pearson_r: float | None


def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson 相关系数；空序列或任一序列无方差时返回 ``None``。"""
    n = len(xs)
    if n == 0:
        return None
    mean_x = math.fsum(xs) / n
    mean_y = math.fsum(ys) / n
    sxx = math.fsum((x - mean_x) ** 2 for x in xs)
    syy = math.fsum((y - mean_y) ** 2 for y in ys)
    if sxx <= 0.0 or syy <= 0.0:
        return None
    sxy = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def analyze_item(conn: sqlite3.Connection, item_id: str) -> ItemAnalysis:
    """统计单题在已结束会话中的作答表现（纯查询，不写库）。

    调用方负责确认题目存在（未知题目应返回 404）。
    """
    # UNIQUE(session_id, item_id) 保证每场会话每题至多一条作答，
    # 因此 target_score 就是该场目标题的 0/1 得分。
    rows = conn.execute(
        """
        SELECT r.session_id AS session_id,
               SUM(CASE WHEN r.item_id = ? THEN r.score ELSE 0 END)
                   AS target_score,
               COUNT(*) AS n_answered,
               SUM(r.score) AS total_score
        FROM responses r
        JOIN sessions s ON s.id = r.session_id
        WHERE s.status = 'finished'
        GROUP BY r.session_id
        HAVING SUM(CASE WHEN r.item_id = ? THEN 1 ELSE 0 END) > 0
           AND COUNT(*) > 1
        """,
        (item_id, item_id),
    ).fetchall()

    targets: list[float] = []
    others: list[float] = []
    for row in rows:
        n_others = row["n_answered"] - 1
        targets.append(float(row["target_score"]))
        others.append((row["total_score"] - row["target_score"]) / n_others)

    n = len(targets)
    correct_rate = math.fsum(targets) / n if n else None
    pearson_r = pearson(targets, others) if n >= MIN_SAMPLES else None
    return ItemAnalysis(
        item_id=item_id,
        sample_size=n,
        correct_rate=correct_rate,
        pearson_r=pearson_r,
    )
