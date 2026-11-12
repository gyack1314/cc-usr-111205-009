"""偏好敏感性分析的端到端测试。"""
import json
import unittest
from fractions import Fraction

from response_data.linalg import Row
from response_data.polyhedron import Problem, verify_certificate
from response_data.sensitivity import analyze, enumerate_vertices


def make_problem(alts, constraints, criteria=("c1", "c2")):
    return Problem({
        "criteria": list(criteria),
        "alternatives": [
            {"id": aid, "utility": [str(v) for v in utils]}
            for aid, utils in alts
        ],
        "constraints": constraints,
        "solver": {"tolerance": "1/100000000", "max_nodes": 100000},
    })


SIMPLE_CONSTRAINTS = []  # 仅 w>=0 与 sum w=1 的单纯形


class PairwiseTests(unittest.TestCase):
    def test_dominating_alternative(self):
        # a 在两个指标上都严格优于 b -> 必然严格优于
        p = make_problem(
            [("a", (1, 1)), ("b", (0, 0))],
            SIMPLE_CONSTRAINTS)
        r = analyze(p)
        pair = next(e for e in r["pairwise"] if e["a"] == "a" and e["b"] == "b")
        self.assertEqual(pair["min"]["bound"]["exact"], "1")
        self.assertEqual(pair["max"]["bound"]["exact"], "1")
        self.assertTrue(pair["relations"]["necessarily_strictly_better"])
        self.assertTrue(pair["relations"]["necessarily_no_worse"])
        edges = r["robust_order"]["necessarily_strictly_better_edges"]
        self.assertIn({"a": "a", "b": "b"}, edges)
        # 顶点交叉检查必须零失配
        self.assertTrue(r["vertex_cross_check"]["linear_objective_extrema_match"])
        self.assertEqual(r["vertex_cross_check"]["mismatches"], [])

    def test_indistinguishable_pair(self):
        # 效用完全相同的两个不同 ID：界恒为 0，可并列、无严格关系
        p = make_problem(
            [("a", (Fraction(1, 2), Fraction(1, 2))),
             ("b", (Fraction(1, 2), Fraction(1, 2))),
             ("c", (1, 0))],
            SIMPLE_CONSTRAINTS)
        r = analyze(p)
        pair = next(e for e in r["pairwise"] if e["a"] == "a" and e["b"] == "b")
        self.assertEqual(pair["min"]["bound"]["exact"], "0")
        self.assertEqual(pair["max"]["bound"]["exact"], "0")
        self.assertTrue(pair["identical_utility"])
        self.assertFalse(pair["relations"]["possibly_strictly_better"])
        self.assertTrue(pair["relations"]["possibly_tied"])
        # 重复方案身份都保留，且都能并列第一
        ids = {t["alternative"] for t in r["can_be_tied_first"] if t["feasible"]}
        self.assertIn("a", ids)
        self.assertIn("b", ids)
        self.assertEqual(r["duplicate_utilities_preserved"][0]["ids"], ["a", "b"])

    def test_preference_dependent_order(self):
        # 无差异曲线相交：哪个更好取决于权重
        p = make_problem(
            [("a", (1, 0)), ("b", (0, 1))],
            SIMPLE_CONSTRAINTS)
        r = analyze(p)
        ab = next(e for e in r["pairwise"] if e["a"] == "a" and e["b"] == "b")
        self.assertEqual(ab["min"]["bound"]["exact"], "-1")
        self.assertEqual(ab["max"]["bound"]["exact"], "1")
        self.assertFalse(ab["relations"]["necessarily_no_worse"])
        self.assertTrue(ab["relations"]["possibly_strictly_better"])
        self.assertTrue(ab["relations"]["possibly_strictly_worse"])
        self.assertTrue(ab["relations"]["possibly_tied"])
        # 区间含 0 时给出的等式见证必须真在 W 中
        self.assertTrue(ab["tie_witness"]["in_W"])


class RegretTests(unittest.TestCase):
    def test_regret_nonnegative_and_witness_in_W(self):
        p = make_problem(
            [("a", (1, 0)), ("b", (Fraction(1, 2), Fraction(1, 2)))],
            SIMPLE_CONSTRAINTS)
        r = analyze(p)
        for row in r["worst_case_regret"]:
            self.assertGreaterEqual(Fraction(row["regret"]["exact"]), 0)
            self.assertTrue(row["witness"]["in_W"])
            for term in row["per_b"]:
                if term["proven"]:
                    self.assertTrue(term["witness"]["in_W"])
        self.assertIn(r["minimax_regret_choice"]["selected"], ("a", "b"))

    def test_tie_break_by_id(self):
        # 两个对称方案遗憾相同，选择按 ID（输入顺序）稳定决胜
        p = make_problem(
            [("zeta", (1, 0)), ("alpha", (0, 1))],
            SIMPLE_CONSTRAINTS)
        r = analyze(p)
        vals = {x["alternative"]: Fraction(x["regret"]["exact"])
                for x in r["worst_case_regret"]}
        if vals["zeta"] == vals["alpha"]:
            self.assertEqual(r["minimax_regret_choice"]["selected"], "zeta")


class InvarianceTests(unittest.TestCase):
    def test_redundant_constraints_dont_change_conclusions(self):
        base = make_problem(
            [("a", (1, Fraction(1, 4))), ("b", (Fraction(1, 3), 1))],
            SIMPLE_CONSTRAINTS)
        # 加入被单纯形蕴含的冗余约束：w1 <= 1、w1+w2 <= 1
        redundant = [
            {"coefficients": ["1", "0"], "relation": "le", "rhs": "1"},
            {"coefficients": ["1", "1"], "relation": "le", "rhs": "1"},
        ]
        p2 = make_problem(
            [("a", (1, Fraction(1, 4))), ("b", (Fraction(1, 3), 1))],
            redundant)
        r1, r2 = analyze(base), analyze(p2)
        b1 = {(e["a"], e["b"]): (e["min"]["bound"]["exact"],
                                  e["max"]["bound"]["exact"])
              for e in r1["pairwise"]}
        b2 = {(e["a"], e["b"]): (e["min"]["bound"]["exact"],
                                  e["max"]["bound"]["exact"])
              for e in r2["pairwise"]}
        self.assertEqual(b1, b2)
        self.assertEqual(
            [x["regret"]["exact"] for x in r1["worst_case_regret"]],
            [x["regret"]["exact"] for x in r2["worst_case_regret"]])

    def test_vertex_enumeration_matches_lp(self):
        p = make_problem(
            [("a", (Fraction(3, 4), 0)), ("b", (0, Fraction(3, 4))),
             ("c", (Fraction(1, 2), Fraction(1, 2)))],
            [{"coefficients": ["1", "-1"], "relation": "le", "rhs": "0"}])
        r = analyze(p)
        cc = r["vertex_cross_check"]
        self.assertTrue(cc["performed"])
        self.assertTrue(cc["linear_objective_extrema_match"])


class PolyhedronShapeTests(unittest.TestCase):
    def test_empty_polyhedron_certificate(self):
        # w1+w2=1 且 w1<=-1（翻转后与非负矛盾）
        p = make_problem(
            [("a", (1, 0))],
            [{"coefficients": ["1", "0"], "relation": "le", "rhs": "-1"}])
        r = analyze(p)
        self.assertEqual(r["weight_polyhedron"]["status"], "empty")
        cert = r["weight_polyhedron"]["incompatibility_certificate"]
        rows = p.full_rows()
        valid, _, why = verify_certificate(rows, cert)
        self.assertTrue(valid, why)
        # 空集上不产生任何全称稳健关系
        self.assertEqual(r["completion"]["proven_necessary_relations"], 0)
        self.assertEqual(r["pairwise"], [])

    def test_lower_dimensional_face_supported(self):
        # 3 个权重，等式 w1 = 1/2 把可行域压到一维线段，仍须完整分析
        p = make_problem(
            [("a", (1, 0, 0)), ("b", (0, 1, 0))],
            [{"coefficients": ["1", "0", "0"], "relation": "eq",
              "rhs": "1/2"}],
            criteria=("c1", "c2", "c3"))
        r = analyze(p)
        self.assertEqual(r["weight_polyhedron"]["status"], "nonempty")
        self.assertEqual(
            r["weight_polyhedron"]["affine"]["affine_dimension_at_most"], 1)
        # 见证权重落在该低维面上且残差为 0
        w = r["weight_polyhedron"]["feasibility_witness"]
        self.assertTrue(w["in_W"])
        self.assertEqual(w["max_abs_float_residual"], 0.0)
        self.assertTrue(r["vertex_cross_check"]["performed"])

    def test_tied_first_exclusion_certificate(self):
        # b 被 a 全面压倒 -> b 不可能并列第一，给排除证据
        p = make_problem(
            [("a", (1, 1)), ("b", (0, 0))],
            SIMPLE_CONSTRAINTS)
        r = analyze(p)
        b_entry = next(t for t in r["can_be_tied_first"]
                       if t["alternative"] == "b")
        self.assertFalse(b_entry["feasible"])
        # 排除证据可在增强系统上独立复算
        n = p.n
        ua = next(u for i, u in p.alternatives if i == "b")
        aug = list(p.full_rows())
        for aid, ub in p.alternatives:
            if aid == "b":
                continue
            aug.append(Row(f"tied-first:b>={aid}",
                           tuple(ub[j] - ua[j] for j in range(n)),
                           "le", Fraction(0)))
        valid, _, why = verify_certificate(
            aug, b_entry["exclusion_certificate"])
        self.assertTrue(valid, why)


class CompletionTests(unittest.TestCase):
    def test_all_bounds_proven_on_normal_data(self):
        with open("data/normal.json", encoding="utf8") as fh:
            p = Problem(json.load(fh))
        r = analyze(p)
        c = r["completion"]
        self.assertEqual(c["pairwise_bounds_proven"],
                         c["pairwise_bounds_total"])
        self.assertEqual(c["regret_values_proven"], c["regret_values_total"])
        # 所有见证权重的浮点残差都在配置容差内
        for e in r["pairwise"]:
            for side in ("min", "max"):
                self.assertTrue(e[side]["witness"][
                                    "float_residual_within_tolerance"])


if __name__ == "__main__":
    unittest.main()
