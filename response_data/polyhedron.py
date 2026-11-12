"""权重多面体 W 的构建与独立复核。

完整系统 = 每个 w_j>=0（le 行）+ sum w=1（eq 行）+ 资料中的额外约束。
本模块只做精确有理数构建与“可独立复算”的检查，不包含排序口径；
残差与 Farkas 证据均在此重新计算，不引用求解器内部状态。
"""
from fractions import Fraction

from .io import fraction, unique
from .linalg import Row, minimize

SUM_ROW = "sum-weights=1"
NEG_PREFIX = "w["
NEG_SUFFIX = "]>=0"


class Problem:
    def __init__(self, raw):
        if not isinstance(raw, dict):
            raise ValueError("顶层资料必须为对象")
        criteria = raw["criteria"]
        if not isinstance(criteria, list) or not criteria:
            raise ValueError("指标身份无效")
        if len(set(criteria)) != len(criteria):
            raise ValueError("指标身份无效")
        n = len(criteria)
        unique(raw["alternatives"], "id")  # 重复方案保留各自身份：仅禁止同 ID
        alternatives = []
        for alt in raw["alternatives"]:
            utils = alt["utility"]
            if not isinstance(utils, list) or len(utils) != n:
                raise ValueError(f"方案 {alt['id']} 效用维度无效")
            u = tuple(fraction(x) for x in utils)
            if any(not (Fraction(0) <= v <= Fraction(1)) for v in u):
                raise ValueError(f"方案 {alt['id']} 效用超出 [0,1]")
            alternatives.append((alt["id"], u))  # 方向统一、尺度外部固定，不重新归一
        extra = []
        for k, c in enumerate(raw["constraints"]):
            if c["relation"] not in ("le", "eq"):
                raise ValueError("约束 relation 仅支持 le / eq")
            coeff = c["coefficients"]
            if not isinstance(coeff, list) or len(coeff) != n:
                raise ValueError(f"第 {k} 条约束系数维度无效")
            extra.append(Row(
                name=f"constraint[{k}]:{c['relation']}",
                a=tuple(fraction(x) for x in coeff),
                rel=c["relation"],
                b=fraction(c["rhs"]),
            ))
        self.criteria = list(criteria)
        self.alternatives = alternatives
        self.extra = extra
        self.n = n
        solver = raw.get("solver", {})
        self.tolerance = fraction(solver.get("tolerance", "1/100000000"))
        max_nodes = solver.get("max_nodes")
        self.max_nodes = None if max_nodes is None else int(max_nodes)

    def full_rows(self):
        """完整行序：非负约束、单纯形约束、额外约束（行名稳定，证据按名复算）。"""
        rows = [
            Row(NEG_PREFIX + str(j) + NEG_SUFFIX,
                tuple(Fraction(-1) if k == j else Fraction(0) for k in range(self.n)),
                "le", Fraction(0))
            for j in range(self.n)
        ]
        rows.append(Row(SUM_ROW, tuple(Fraction(1) for _ in range(self.n)),
                        "eq", Fraction(1)))
        rows.extend(self.extra)
        return rows


def feasibility(rows, max_nodes=None):
    """检查系统非空；返回 (ok, LPResult)。空集时结果携带 Farkas 证据。"""
    n = len(rows[0].a)
    res = minimize(rows, tuple(Fraction(0) for _ in range(n)), max_nodes=max_nodes)
    if res.status == "infeasible":
        return False, res
    if res.status == "optimal":
        return True, res
    return None, res  # node_limit / unbounded：未完成，不下结论


def residual_report(rows, w):
    """对候选权重逐行复算残差，含精确值与浮点最大绝对残差。"""
    out = []
    float_max = 0.0
    for r in rows:
        lhs = sum(r.a[j] * w[j] for j in range(len(w)))
        gap = lhs - r.b
        if r.rel == "eq":
            ok = gap == 0
            fgap = abs(float(gap))
        else:
            ok = gap <= 0
            fgap = max(0.0, float(gap))
        float_max = max(float_max, fgap)
        out.append({
            "row": r.name,
            "relation": r.rel,
            "lhs_minus_rhs_exact": str(gap),
            "lhs_minus_rhs_float": float(gap),
            "satisfied": ok,
        })
    return out, float_max


def verify_certificate(rows, certificate):
    """独立复算 Farkas 证据：乘子符号、组合左端非负、右端严格负。"""
    by_name = {r.name: r for r in rows}
    n = len(rows[0].a)
    g = [Fraction(0)] * n
    h = Fraction(0)
    detail = []
    for item in certificate["multipliers"]:
        r = by_name[item["row"]]
        beta = fraction(item["beta"]) if not isinstance(item["beta"], Fraction) else item["beta"]
        if r.rel == "le" and beta < 0:
            return False, detail, "le 行乘子必须非负"
        for j in range(n):
            g[j] += beta * r.a[j]
        h += beta * r.b
        detail.append((r.name, beta))
    if any(v < 0 for v in g):
        return False, detail, "组合左端存在负分量，无法与 x>=0 矛盾"
    if h >= 0:
        return False, detail, "组合右端未严格为负"
    expected_g = tuple(fraction(x) if not isinstance(x, Fraction) else x
                       for x in certificate["combined_lhs"])
    expected_h = (fraction(certificate["combined_rhs"])
                  if not isinstance(certificate["combined_rhs"], Fraction)
                  else certificate["combined_rhs"])
    if tuple(g) != expected_g or h != expected_h:
        return False, detail, "证据中的组合值与按乘子重算不一致"
    return True, detail, f"任意可行 x 都有 0 <= g·x <= {h} < 0，矛盾"


def affine_dimension(problem):
    """等式面（单纯形+eq 约束）的秩与仿射维数；低维面同样可分析。"""
    n = problem.n
    eq_rows = [r for r in problem.full_rows() if r.rel == "eq"]
    mat = [list(r.a) + [r.b] for r in eq_rows]
    rank = 0
    pivot_cols = []
    for col in range(n):
        pivot = next((i for i in range(rank, len(mat)) if mat[i][col] != 0), None)
        if pivot is None:
            continue
        mat[rank], mat[pivot] = mat[pivot], mat[rank]
        piv = mat[rank][col]
        mat[rank] = [v / piv for v in mat[rank]]
        for i in range(len(mat)):
            if i == rank or mat[i][col] == 0:
                continue
            f = mat[i][col]
            mat[i] = [v - f * mat[rank][j] for j, v in enumerate(mat[i])]
        pivot_cols.append(col)
        rank += 1
        if rank == n:
            break
    inconsistent = any(all(v == 0 for v in row[:n]) and row[n] != 0 for row in mat)
    return {"equality_rank": rank, "affine_dimension_at_most": n - rank,
            "consistent_equalities": not inconsistent}
