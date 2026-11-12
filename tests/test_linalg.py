"""精确有理 LP 求解器与 Farkas 证据的单元测试。"""
import unittest
from fractions import Fraction

from response_data.linalg import (
    Row,
    farkas_certificate,
    maximize,
    minimize,
    solve,
)


def F(x):
    return Fraction(x)


def recheck_farkas(rows, cert):
    """独立复算证书：le 乘子非负、g 分量非负、h 严格负、组合值一致。"""
    n = len(rows[0].a)
    g = [Fraction(0)] * n
    h = Fraction(0)
    for k, item in enumerate(cert["multipliers"]):
        r = rows[k]
        beta = item["beta"]
        if r.rel == "le":
            assert beta >= 0
        for j in range(n):
            g[j] += beta * r.a[j]
        h += beta * r.b
    assert all(v >= 0 for v in g)
    assert h < 0
    assert tuple(g) == cert["combined_lhs"]
    assert h == cert["combined_rhs"]
    return g, h


class SimplexTests(unittest.TestCase):
    def test_simple_min_max(self):
        rows = [Row("x>=0", (-F(1),), "le", F(0)),
                Row("x<=1", (F(1),), "le", F(1))]
        r = minimize(rows, (-F(1),))
        self.assertEqual(r.status, "optimal")
        self.assertEqual(r.objective, -1)
        self.assertEqual(r.x, (1,))
        r = maximize(rows, (-F(1),))
        self.assertEqual(r.objective, 0)
        self.assertEqual(r.x, (0,))

    def test_unbounded(self):
        rows = [Row("x>=0", (-F(1),), "le", F(0))]
        self.assertEqual(maximize(rows, (F(1),)).status, "unbounded")

    def test_two_dimensional(self):
        rows = [Row("x", (-1, 0), "le", 0), Row("y", (0, -1), "le", 0),
                Row("b", (1, 1), "le", 1)]
        r = maximize(rows, (1, 2))
        self.assertEqual(r.status, "optimal")
        self.assertEqual(r.objective, 2)
        self.assertEqual(r.x, (0, 1))

    def test_negative_rhs_lower_bound(self):
        # x >= 3/10 写成 -x <= -3/10（负右端翻转行），配合 x <= 1
        rows = [Row("lo", (-F(1),), "le", Fraction(-3, 10)),
                Row("hi", (F(1),), "le", F(1))]
        r = maximize(rows, (F(1),))
        self.assertEqual((r.objective, r.x), (1, (1,)))
        r = minimize(rows, (F(1),))
        self.assertEqual((r.objective, r.x),
                         (Fraction(3, 10), (Fraction(3, 10),)))

    def test_equality_negative_rhs(self):
        # x = -1 与 x >= 0 不相容
        rows = [Row("eq", (F(1),), "eq", F(-1)),
                Row("n", (-F(1),), "le", F(0))]
        r = solve(rows, (F(0),))
        self.assertEqual(r.status, "infeasible")
        recheck_farkas(rows, r.certificate)

    def test_infeasible_certificates(self):
        cases = [
            [Row("n", (-F(1),), "le", 0), Row("c", (F(1),), "le", 1),
             Row("flip", (F(1),), "le", -2)],
            [Row("n", (-F(1),), "le", 0), Row("e", (F(1),), "eq", 1),
             Row("bad", (F(1),), "le", -1)],
        ]
        for rows in cases:
            r = solve(rows, (F(0),))
            self.assertEqual(r.status, "infeasible")
            recheck_farkas(rows, r.certificate)
            # 独立恢复函数与求解器返回一致
            cert2 = farkas_certificate(rows)
            self.assertIsNotNone(cert2)
            recheck_farkas(rows, cert2)

    def test_witness_satisfies_constraints(self):
        # 最优见证必须真满足全部约束（防止拼出不属于可行域的向量）
        rows = [Row("lo", (-F(1),), "le", Fraction(-3, 10)),
                Row("hi", (F(1),), "le", F(1))]
        r = maximize(rows, (F(1),))
        x = r.x[0]
        for row in rows:
            self.assertLessEqual(row.a[0] * x, row.b)

    def test_node_limit_not_optimal(self):
        rows = [Row("x>=0", (-F(1),), "le", 0),
                Row("x<=1", (F(1),), "le", 1)]
        r = minimize(rows, (-F(1),), max_nodes=0)
        self.assertEqual(r.status, "node_limit")
        self.assertIsNone(r.objective)


if __name__ == "__main__":
    unittest.main()
