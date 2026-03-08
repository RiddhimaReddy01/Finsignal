from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class MetricResult:
    name: str
    value: float
    unit: str
    formula: str
    inputs: Dict[str, float]


class FinancialMetricsEngine:
    @staticmethod
    def _nonzero(x: float, name: str) -> None:
        if x == 0:
            raise ValueError(f"zero_denominator:{name}")

    @staticmethod
    def gross_margin(gross_profit: float, revenue: float) -> MetricResult:
        FinancialMetricsEngine._nonzero(revenue, "revenue")
        return MetricResult(
            name="gross_margin",
            value=gross_profit / revenue,
            unit="RATIO",
            formula="gross_margin = gross_profit / revenue",
            inputs={"gross_profit": gross_profit, "revenue": revenue},
        )

    @staticmethod
    def operating_margin(operating_income: float, revenue: float) -> MetricResult:
        FinancialMetricsEngine._nonzero(revenue, "revenue")
        return MetricResult(
            name="operating_margin",
            value=operating_income / revenue,
            unit="RATIO",
            formula="operating_margin = operating_income / revenue",
            inputs={"operating_income": operating_income, "revenue": revenue},
        )

    @staticmethod
    def net_margin(net_income: float, revenue: float) -> MetricResult:
        FinancialMetricsEngine._nonzero(revenue, "revenue")
        return MetricResult(
            name="net_margin",
            value=net_income / revenue,
            unit="RATIO",
            formula="net_margin = net_income / revenue",
            inputs={"net_income": net_income, "revenue": revenue},
        )

    @staticmethod
    def ebitda_margin(ebitda: float, revenue: float) -> MetricResult:
        FinancialMetricsEngine._nonzero(revenue, "revenue")
        return MetricResult(
            name="ebitda_margin",
            value=ebitda / revenue,
            unit="RATIO",
            formula="ebitda_margin = ebitda / revenue",
            inputs={"ebitda": ebitda, "revenue": revenue},
        )

    @staticmethod
    def free_cash_flow(cfo: float, capex: float) -> MetricResult:
        return MetricResult(
            name="fcf",
            value=cfo - capex,
            unit="USD",
            formula="fcf = cfo - capex",
            inputs={"cfo": cfo, "capex": capex},
        )

    @staticmethod
    def current_ratio(current_assets: float, current_liabilities: float) -> MetricResult:
        FinancialMetricsEngine._nonzero(current_liabilities, "current_liabilities")
        return MetricResult(
            name="current_ratio",
            value=current_assets / current_liabilities,
            unit="RATIO",
            formula="current_ratio = current_assets / current_liabilities",
            inputs={"current_assets": current_assets, "current_liabilities": current_liabilities},
        )

    @staticmethod
    def debt_to_equity(total_debt: float, total_equity: float) -> MetricResult:
        FinancialMetricsEngine._nonzero(total_equity, "total_equity")
        return MetricResult(
            name="debt_to_equity",
            value=total_debt / total_equity,
            unit="RATIO",
            formula="debt_to_equity = total_debt / total_equity",
            inputs={"total_debt": total_debt, "total_equity": total_equity},
        )

    @staticmethod
    def roe(net_income: float, total_equity: float) -> MetricResult:
        FinancialMetricsEngine._nonzero(total_equity, "total_equity")
        return MetricResult(
            name="roe",
            value=net_income / total_equity,
            unit="RATIO",
            formula="roe = net_income / total_equity",
            inputs={"net_income": net_income, "total_equity": total_equity},
        )

    @staticmethod
    def revenue_yoy(revenue_t: float, revenue_t1: float) -> MetricResult:
        FinancialMetricsEngine._nonzero(revenue_t1, "revenue_t1")
        return MetricResult(
            name="revenue_yoy",
            value=(revenue_t - revenue_t1) / revenue_t1,
            unit="RATIO",
            formula="revenue_yoy = (revenue_t - revenue_t1) / revenue_t1",
            inputs={"revenue_t": revenue_t, "revenue_t1": revenue_t1},
        )
