from __future__ import annotations

from dataclasses import dataclass, field

from .strategies import dot

# 定义了一个数据类 BendersCut，用于表示一个 Benders 割。
# 它包含了割所属的场景编号、常数项、系数列表和割的类型（如 optimality cut 或 feasibility cut）。该类还提供了一个方法 value，用于计算给定解向量 x 在该割上的值。
@dataclass(frozen=True)
class BendersCut:
    scenario: int
    const: float
    coeffs: list[float]
    cut_type: str

    def value(self, x: list[int]) -> float:
        return self.const + dot(self.coeffs, [float(v) for v in x])

# 定义了一个数据类 OptimalityCut，表示一个最优性割。它包含了系数列表和右端项，并提供了一个方法 satisfied，用于判断给定解向量 x 是否满足该割。
@dataclass(frozen=True)
class FeasibilityCut:
    coeffs: list[float]
    rhs: float

    def satisfied(self, x: list[int]) -> bool:
        return dot(self.coeffs, [float(v) for v in x]) + 1.0e-8 >= self.rhs

# 定义了一个数据类 MasterSolution，用于表示主问题的解。它包含了整数解向量 x、theta 值、目标值、求解时间以及一些与线性松弛解相关的信息
@dataclass
class MasterSolution:
    x: list[int]
    theta: list[float]
    objective: float
    solve_time: float
    x_lp: list[float] = field(default_factory=list)
    theta_lp: list[float] = field(default_factory=list)
    x_reduced_costs_lp: list[float] = field(default_factory=list)
    theta_reduced_costs_lp: list[float] = field(default_factory=list)
    lp_objective: float = 0.0

# 定义了一个数据类 BendersResult，用于表示 Benders 分解算法的结果。它包含了使用的策略、求解状态、目标值、下界、上界、迭代次数、添加的割数量、运行时间以及日志文件路径等信息。
@dataclass
class BendersResult:
    strategy: str
    status: str
    objective: float
    lower_bound: float
    upper_bound: float
    iterations: int
    cuts_added: int
    runtime: float
    log_path: str
