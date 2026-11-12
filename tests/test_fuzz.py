"""随机有理问题的交叉核对：单纯形 LP 对暴力顶点枚举。"""
import random
import unittest
from fractions import Fraction

from response_data.linalg import Row, maximize, minimize, solve
from response_data.sensitivity import enumerate_vertices


def fr(x):
    return Fraction(x).limit_denominator(20)


class FuzzTests(unittest.TestCase):
    def test_random_polytopes_match_vertex_enumeration(self):
        rng = random.Random(20261113)
        for trial in range(60):
            n = rng.randint(1, 3)
            m = rng.randint(n, n + 3)
            rows = [Row(f"nonneg[{j}]",
                        tuple(Fraction(-1) if k == j else Fraction(0)
                              for k in range(n)),
                        "le", Fraction(0)) for j in range(n)]
            rows.append(Row("simplex",
                            tuple(Fraction(1) for _ in range(n)),
                            "eq", Fraction(1)))
            for k in range(m - n - 1):
                a = tuple(fr(rng.randint(-3, 3)) for _ in range(n))
                rows.append(Row(f"extra[{k}]", a, "le",
                                fr(rng.randint(-2, 2)) / 4))
            verts, skip = enumerate_vertices(rows, n, cap=10_000)
            if skip or not verts:
                continue
            c = tuple(fr(rng.randint(-3, 3)) for _ in range(n))
            rmin = minimize(rows, c)
            rmax = maximize(rows, c)
            if rmin.status != "optimal":
                continue  # 未完成/无界不参与精确核对
            brute_min = min(sum(c[j] * w[j] for j in range(n)) for w in verts)
            brute_max = max(sum(c[j] * w[j] for j in range(n)) for w in verts)
            self.assertEqual(rmin.objective, brute_min, f"min trial {trial}")
            self.assertEqual(rmax.objective, brute_max, f"max trial {trial}")
            # 见证必须属于可行域
            for side, res in (("min", rmin), ("max", rmax)):
                w = res.x
                for r in rows:
                    lhs = sum(r.a[j] * w[j] for j in range(n))
                    if r.rel == "eq":
                        self.assertEqual(lhs, r.b, f"{side} eq residual")
                    else:
                        self.assertLessEqual(lhs, r.b, f"{side} le residual")

    def test_infeasible_random_certificates_recheck(self):
        # 显式构造不相容：单纯形 + w1 <= -1
        rng = random.Random(7)
        for n in (1, 2, 3):
            rows = [Row(f"nonneg[{j}]",
                        tuple(Fraction(-1) if k == j else Fraction(0)
                              for k in range(n)),
                        "le", Fraction(0)) for j in range(n)]
            rows.append(Row("simplex",
                            tuple(Fraction(1) for _ in range(n)),
                            "eq", Fraction(1)))
            rows.append(Row("impossible",
                            tuple(Fraction(1) if k == 0 else Fraction(0)
                                  for k in range(n)),
                            "le", Fraction(-1)))
            res = solve(rows, tuple(Fraction(0) for _ in range(n)))
            self.assertEqual(res.status, "infeasible")
            cert = res.certificate
            g = [Fraction(0)] * n
            h = Fraction(0)
            for k, item in enumerate(cert["multipliers"]):
                r = rows[k]
                if r.rel == "le":
                    self.assertGreaterEqual(item["beta"], 0)
                for j in range(n):
                    g[j] += item["beta"] * r.a[j]
                h += item["beta"] * r.b
            self.assertTrue(all(v >= 0 for v in g))
            self.assertLess(h, 0)


if __name__ == "__main__":
    unittest.main()
