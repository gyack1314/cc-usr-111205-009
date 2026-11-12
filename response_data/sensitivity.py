"""偏好敏感性分析。

所有结论基于权重多面体 W 上的精确有理 LP：
- 成对差 d_ab(w)=w·(u_a-u_b) 的 min/max 在同一个 W 上分别求，
  界与见证权重成对输出；
- 最坏遗憾 R(a)=max_b max_W w·(u_b-u_a)，每个内 max 都用属于 W 的
  真实见证权重（逐行残差复核），绝不把各指标独立极值拼成不在 W 中的向量；
- “至少并列第一”是单个 w 同时满足 a>=所有 b 的可行性 LP；
- 小维度用全部顶点枚举做交叉检查，维度大时跳过而非硬枚举。
本模块产出的是偏好敏感性信息，不是现实安全性结论或调度命令。
"""
from fractions import Fraction
from itertools import combinations

from .linalg import Row, maximize, minimize
from .polyhedron import (
    affine_dimension,
    feasibility,
    residual_report,
    verify_certificate,
)

VERTEX_COMBO_CAP = 5000


def _q(x):
    """精确值与十进制近似同时输出；精确串为准，浮点仅作可读对照。"""
    return {"exact": str(x), "decimal": float(x)}


def _witness(rows, w, tolerance):
    residuals, float_max = residual_report(rows, w)
    return {
        "weights": [_q(v) for v in w],
        "in_W": all(r["satisfied"] for r in residuals),
        "max_abs_float_residual": float_max,
        "float_residual_within_tolerance": float_max <= float(tolerance),
        "row_residuals": residuals,
    }


def _bound_lp(rows, coeff, problem, want):
    res = (maximize if want == "max" else minimize)(
        rows, coeff, max_nodes=problem.max_nodes)
    if res.status != "optimal":
        return {"proven": False, "status": res.status, "pivots": res.pivots}
    return {
        "proven": True,
        "bound": _q(res.objective),
        "witness": _witness(rows, res.x, problem.tolerance),
        "w_raw": res.x,
        "pivots": res.pivots,
    }


def _diff_vector(ua, ub):
    return tuple(ua[j] - ub[j] for j in range(len(ua)))


def _tie_witness(lo, hi, wlo, whi, n):
    """区间含 0 时构造属于 W 的等式见证：W 凸，对 min/max 见证精确插值。"""
    vlo, vhi = lo, hi
    if vlo == 0:
        return wlo
    if vhi == 0:
        return whi
    t = (-vlo) / (vhi - vlo)
    return tuple((1 - t) * wlo[j] + t * whi[j] for j in range(n))


def enumerate_vertices(rows, n, cap=VERTEX_COMBO_CAP):
    """枚举 W 的全部顶点：eq 行恒等 + 任选 n 个线性无关活动 le 行。"""
    eqs = [r for r in rows if r.rel == "eq"]
    les = [r for r in rows if r.rel == "le"]
    base = [list(r.a) for r in eqs]
    base_rhs = [r.b for r in eqs]
    if len(list(combinations(les, n - len(eqs)))) > cap:
        return None, "vertex_budget_exceeded"
    vertices = set()
    for chosen in combinations(les, max(0, n - len(eqs))):
        mat = [row[:] for row in base] + [list(r.a) for r in chosen]
        rhs = base_rhs[:] + [r.b for r in chosen]
        sol = _solve_square(mat, rhs, n)
        if sol is None:
            continue
        if all(sum(r.a[j] * sol[j] for j in range(n)) <= r.b for r in rows):
            vertices.add(sol)
    return sorted(vertices), None


def _solve_square(mat, rhs, n):
    m = [list(mat[i]) + [rhs[i]] for i in range(len(mat))]
    rank = 0
    for col in range(n):
        pivot = next((i for i in range(rank, len(m)) if m[i][col] != 0), None)
        if pivot is None:
            return None
        m[rank], m[pivot] = m[pivot], m[rank]
        piv = m[rank][col]
        m[rank] = [v / piv for v in m[rank]]
        for i in range(len(m)):
            if i == rank or m[i][col] == 0:
                continue
            f = m[i][col]
            m[i] = [v - f * m[rank][k] for k, v in enumerate(m[i])]
        rank += 1
        if rank == n:
            break
    if rank < n:
        return None
    return tuple(m[i][n] for i in range(n))


def analyze(problem):
    rows = problem.full_rows()
    n = problem.n
    alts = problem.alternatives
    ids = [a[0] for a in alts]
    report = {
        "criteria": problem.criteria,
        "alternative_ids": ids,
        "duplicate_utilities_preserved": _duplicate_groups(alts),
        "weight_polyhedron": {"row_order": [r.name for r in rows]},
        "pairwise": [],
        "worst_case_regret": [],
        "can_be_tied_first": [],
        "robust_order": {},
        "vertex_cross_check": None,
        "completion": {},
        "disclaimer": "本结果是偏好敏感性分析：最低遗憾不等于现实最安全，也不构成调度命令。",
    }

    ok, feas = feasibility(rows, max_nodes=problem.max_nodes)
    report["weight_polyhedron"]["affine"] = affine_dimension(problem)
    if ok is False:
        cert = feas.certificate
        valid, _, why = verify_certificate(rows, cert)
        report["weight_polyhedron"].update({
            "status": "empty",
            "incompatibility_certificate": _cert_json(cert),
            "certificate_recheck": {"valid": valid, "explanation": why},
        })
        report["completion"] = {"proven_necessary_relations": 0,
                                "note": "W 为空：不输出任何全称稳健关系，空集上的全称命题不作为获胜结论。"}
        return report
    if ok is None:
        report["weight_polyhedron"].update({"status": "unknown",
                                            "reason": f"可行性 LP 未完成：{feas.status}"})
        return report
    report["weight_polyhedron"].update({
        "status": "nonempty",
        "feasibility_witness": _witness(rows, feas.x, problem.tolerance),
    })

    proven_bounds = 0
    total_bounds = 0
    robust_no_worse = []
    robust_strict = []

    for ia, (ida, ua) in enumerate(alts):
        for ib, (idb, ub) in enumerate(alts):
            if ia == ib:
                continue
            d = _diff_vector(ua, ub)
            total_bounds += 2
            lo = _bound_lp(rows, d, problem, "min")
            hi = _bound_lp(rows, d, problem, "max")
            entry = {"a": ida, "b": idb, "min": lo, "max": hi,
                     "identical_utility": ua == ub}
            if lo["proven"] and hi["proven"]:
                proven_bounds += 2
                vlo = Fraction(lo["bound"]["exact"])
                vhi = Fraction(hi["bound"]["exact"])
                flags = {
                    "necessarily_strictly_better": vlo > 0,
                    "necessarily_no_worse": vlo >= 0,
                    "possibly_strictly_better": vhi > 0,
                    "possibly_strictly_worse": vlo < 0,
                    "possibly_tied": vlo <= 0 <= vhi,
                }
                entry["relations"] = flags
                if flags["possibly_tied"] and vlo < vhi:
                    w = _tie_witness(vlo, vhi, lo["w_raw"], hi["w_raw"], n)
                    entry["tie_witness"] = _witness(rows, w, problem.tolerance)
                if vlo >= 0:
                    robust_no_worse.append((ida, idb))
                if vlo > 0:
                    robust_strict.append((ida, idb))
            else:
                entry["relations"] = None
                entry["unproven_reason"] = "LP 未完成，该对不参与已证明必然关系"
            lo.pop("w_raw", None)
            hi.pop("w_raw", None)
            report["pairwise"].append(entry)

    report["robust_order"] = {
        "necessarily_no_worse_edges": [{"a": a, "b": b} for a, b in robust_no_worse],
        "necessarily_strictly_better_edges": [{"a": a, "b": b} for a, b in robust_strict],
        "interpretation": "边 a->b 表示对全部 w∈W 成立；未列出的对属于权重信息不足以定序。",
    }

    # 最坏遗憾：R(a)=max_b v_ab，v_ab=max_W w·(u_b-u_a)，见证逐行属于 W
    regret_rows = []
    for ida, ua in alts:
        terms = []
        complete = True
        for idb, ub in alts:
            d = tuple(ub[j] - ua[j] for j in range(n))
            res = maximize(rows, d, max_nodes=problem.max_nodes)
            if res.status != "optimal":
                complete = False
                terms.append({"b": idb, "proven": False, "status": res.status})
                continue
            wit = _witness(rows, res.x, problem.tolerance)
            terms.append({"b": idb, "proven": True,
                          "value": _q(res.objective), "witness": wit,
                          "witness_belongs_to_W": wit["in_W"]})
        if complete:
            worst = max(terms, key=lambda t: Fraction(t["value"]["exact"]))
            regret_rows.append({"alternative": ida, "regret": worst["value"],
                                "attained_against": worst["b"],
                                "witness": worst["witness"], "per_b": terms})
        else:
            regret_rows.append({"alternative": ida, "regret": None,
                                "per_b": terms, "proven": False})
    proven_regret = [r for r in regret_rows if r["regret"] is not None]
    if proven_regret:
        best = min(proven_regret,
                   key=lambda r: (Fraction(r["regret"]["exact"]),
                                  ids.index(r["alternative"])))
        # 同值按方案 ID 决胜（ID 为字符串，按输入顺序即固定身份次序）
        min_val = Fraction(best["regret"]["exact"])
        tied = sorted(r["alternative"] for r in proven_regret
                      if Fraction(r["regret"]["exact"]) == min_val)
        report["minimax_regret_choice"] = {
            "selected": best["alternative"],
            "tie_break": "最小 R 并列时按方案 ID（输入顺序）决胜",
            "same_regret_group": tied,
        }
    report["worst_case_regret"] = regret_rows

    # 至少并列第一：同一 w 同时满足 (u_a-u_b)·w >= 0 对所有 b
    tied_first = []
    for ida, ua in alts:
        aug = list(rows)
        for idb, ub in alts:
            if idb == ida:
                continue
            aug.append(Row(
                name=f"tied-first:{ida}>={idb}",
                a=tuple(ub[j] - ua[j] for j in range(n)),
                rel="le", b=Fraction(0)))
        res = minimize(aug, tuple(Fraction(0) for _ in range(n)),
                       max_nodes=problem.max_nodes)
        if res.status == "optimal":
            tied_first.append({"alternative": ida, "feasible": True,
                               "witness": _witness(aug, res.x, problem.tolerance)})
        elif res.status == "infeasible":
            valid, _, why = verify_certificate(aug, res.certificate)
            tied_first.append({"alternative": ida, "feasible": False,
                               "exclusion_certificate": _cert_json(res.certificate),
                               "certificate_recheck": {"valid": valid, "explanation": why}})
        else:
            tied_first.append({"alternative": ida, "feasible": None,
                               "status": res.status})
    report["can_be_tied_first"] = tied_first

    # 小维度顶点枚举交叉检查（不要求 W 满维；低维面同样枚举）
    verts, skip = enumerate_vertices(rows, n)
    if skip:
        report["vertex_cross_check"] = {"performed": False, "reason": skip}
    else:
        report["vertex_cross_check"] = _cross_check(
            verts, alts, rows, problem, report, tied_first)

    report["completion"] = {
        "pairwise_bounds_proven": proven_bounds,
        "pairwise_bounds_total": total_bounds,
        "regret_values_proven": len(proven_regret),
        "regret_values_total": len(alts),
        "proven_necessary_relations": len(robust_no_worse),
        "rule": "只有 proven=True 的 LP 参与“已证明必然”关系；未完成 LP 一律排除。",
    }
    return report


def _cross_check(verts, alts, rows, problem, report, tied_first):
    n = problem.n
    mismatches = []
    vmap = {(e["a"], e["b"]): e for e in report["pairwise"]}
    for ia, (ida, ua) in enumerate(alts):
        for ib, (idb, ub) in enumerate(alts):
            if ia == ib:
                continue
            d = _diff_vector(ua, ub)
            vals = [sum(d[j] * w[j] for j in range(n)) for w in verts]
            vmin, vmax = min(vals), max(vals)
            e = vmap[(ida, idb)]
            if e["min"].get("proven"):
                if vmin != Fraction(e["min"]["bound"]["exact"]):
                    mismatches.append({"pair": [ida, idb], "bound": "min",
                                       "lp": e["min"]["bound"]["exact"],
                                       "vertices": str(vmin)})
                if vmax != Fraction(e["max"]["bound"]["exact"]):
                    mismatches.append({"pair": [ida, idb], "bound": "max",
                                       "lp": e["max"]["bound"]["exact"],
                                       "vertices": str(vmax)})
    # 并列第一可行性：存在顶点满足全部并列约束
    tf = {}
    for ida, ua in alts:
        ok = any(all(sum((ua[j] - ub[j]) * w[j] for j in range(n)) >= 0
                     for idb, ub in alts) for w in verts)
        tf[ida] = ok
    for item in tied_first:
        if item["feasible"] is not None and bool(item["feasible"]) != tf[item["alternative"]]:
            mismatches.append({"tied_first": item["alternative"],
                               "lp": item["feasible"], "vertices": tf[item["alternative"]]})
    return {
        "performed": True,
        "vertex_count": len(verts),
        "vertices": [[_q(v) for v in w] for w in verts],
        "linear_objective_extrema_match": not mismatches,
        "mismatches": mismatches,
    }


def _cert_json(cert):
    return {
        "multipliers": [{"row": m["row"], "relation": m["relation"],
                         "beta": str(m["beta"])} for m in cert["multipliers"]],
        "combined_lhs": [str(v) for v in cert["combined_lhs"]],
        "combined_rhs": str(cert["combined_rhs"]),
        "how_to_recheck": "le 行乘子非负（eq 行乘子自由）；乘子乘各行左端之和 g 分量全非负，"
                          "右端之和 h 严格负，则任意可行 x 都有 0<=g·x<=h<0，矛盾。",
    }


def _duplicate_groups(alts):
    seen = {}
    for aid, u in alts:
        seen.setdefault(tuple(u), []).append(aid)
    return [{"utility": [str(v) for v in u], "ids": ids}
            for u, ids in seen.items() if len(ids) > 1]
