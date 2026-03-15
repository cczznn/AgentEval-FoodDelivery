from __future__ import annotations

from typing import List

from .base import Metric, MetricResult


class TaskCompletion(Metric):
    name = "task_completion"

    def __init__(self, keywords: List[str] | None = None) -> None:
        self.keywords = keywords or ["查询", "下单", "支付", "订单号"]

    def score(self, sample) -> MetricResult:
        text = sample.final_answer or ""
        matched = sum(1 for k in self.keywords if k in text)
        score = matched / len(self.keywords) if self.keywords else 0.0
        return MetricResult(value=score, reason="任务完成度(规则版)", traces={"matched": matched})

