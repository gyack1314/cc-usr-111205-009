"""精确有理数两阶段单纯形求解器。

只依赖标准库 fractions，全部运算为 Fraction，无浮点截断。
变量统一要求非负（本项目权重即如此）；约束行为 le / eq。

规范化把每行右端归一到非负并记录翻转符号 sig：
- le 且 b>=0：加非负松弛 s（系数 +1），A x + s = b，松弛直接作初始基；
- le 且 b<0：乘 -1 成 >= 形式（右端为正），加剩余 r（系数 -1）与人工 t 取基；
- eq：b<0 时整体乘 -1（等式，乘子仍自由），加人工 t 取基。
第一阶段最小化人工变量之和，Bland 规则选列杜绝循环；z0>0 即不相容。
不相容证据不在表内凑符号，而由独立的转置 LP farkas_certificate 恢复：
le 行乘子非负、eq 行乘子自由，组合左端对原变量分量非负、右端严格负，
可由 polyhedron.verify_certificate 独立复算。
"""
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class Row:
    """一行原始约束：sum(a_j x_j) rel b，name 由调用方保证可追溯。"""

    name: str
    a: tuple[Fraction, ...]
    rel: str  # 'le' | 'eq'
    b: Fraction


@dataclass
class LPResult:
    # status：optimal / infeasible / unbounded / node_limit
    # node_limit 属于“未完成的 LP”，调用方不得据此构造必然结论
    status: str
    objective: Fraction | None = None
    x: tuple[Fraction, ...] | None = None
    # 不相容证据：beta 为各原始行乘子（le 非负、eq 自由），
    # g=sum beta*A 分量非负，h=sum beta*b 严格负，构成 g·x<=h<0 与 g·x>=0 的矛盾
    certificate: dict | None = None
    pivots: int = 0


def solve(rows, c, max_nodes=None):
    """min c·x s.t. rows, x>=0；rows 非空且系数维度与 c 一致。"""
    rows = list(rows)
    m = len(rows)
    n = len(c)
    if m == 0:
        raise ValueError("至少需要一行约束")
    for r in rows:
        if len(r.a) != n:
            raise ValueError(f"约束 {r.name} 系数维度 {len(r.a)} != 变量数 {n}")
        if r.rel not in ("le", "eq"):
            raise ValueError(f"约束 {r.name} 关系非法")

    # 规范化：所有行右端归一到非负，记录翻转符号 sig。
    # - le 且 b>=0：加松弛 s（系数 +1）即可作初始基；
    # - le 且 b<0：乘 -1 成 >= 形式（右端为正），加剩余 r（系数 -1）与人工 t；
    # - eq：b<0 时乘 -1，加人工 t。
    # 这样每个初始基变量（松弛/人工）系数 +1、初值 = 非负右端，BFS 合法。
    norm = []
    for r in rows:
        if r.b >= 0:
            norm.append((r, Fraction(1)))
        else:
            norm.append((
                Row(r.name, tuple(-v for v in r.a), r.rel, -r.b),
                Fraction(-1)))

    # 列身份：x 原变量；s 为非翻转 le 的松弛；r 为翻转 le 的剩余变量；
    # t 为人工变量。第一阶段只禁止 t 入基，s/r 均可正常参与。
    cols = [("x", j) for j in range(n)]
    slack_col: dict[int, int] = {}
    art_col: dict[int, int] = {}
    for k, (r, sig) in enumerate(norm):
        if r.rel == "le":
            tag = "r" if sig == -1 else "s"
            slack_col[k] = len(cols)
            cols.append((tag, k))
        if r.rel == "eq" or sig == -1:
            art_col[k] = len(cols)
            cols.append(("t", k))
    width = len(cols) + 1

    T = []
    basis: dict[int, int] = {}
    for k, (r, sig) in enumerate(norm):
        row = [Fraction(0)] * width
        for j, v in enumerate(r.a):
            row[j] = v
        if r.rel == "le":
            # 非翻转 le：A x + s = b（松弛 +1）；
            # 翻转 le（归一后的 >= 约束）：A' x - r = b'（剩余 -1），再由人工取基
            row[slack_col[k]] = 1 if sig == 1 else -1
        if k in art_col:
            row[art_col[k]] = 1
            basic_col = art_col[k]
        else:
            basic_col = slack_col[k]
        row[-1] = r.b
        T.append(row)
        basis[k] = basic_col

    # 第一阶段目标 z = 人工变量之和。先置 -sum t，再叠加人工基行把人工列
    # 约简为 0，得到字典式表行（右端 z0 为当前目标值，非负）。
    z = [Fraction(0)] * width
    for tc in art_col.values():
        z[tc] = -1
    for k in art_col:
        for j in range(width):
            z[j] += T[k][j]

    live = list(range(m))
    pivots = 0

    def pivot(enter_col, table, zrow, bcols, allowed_rows):
        """Bland 比率检验转轴；返回 ok / unbounded / node_limit。"""
        nonlocal pivots
        ratios = []
        for i in allowed_rows:
            coef = table[i][enter_col]
            if coef > 0:
                ratios.append((table[i][-1] / coef, bcols[i], i))
        if not ratios:
            return "unbounded"
        _, _, leave = min(ratios, key=lambda t: (t[0], t[1]))
        piv = table[leave][enter_col]
        table[leave] = [v / piv for v in table[leave]]
        for i in allowed_rows:
            if i == leave:
                continue
            factor = table[i][enter_col]
            if factor == 0:
                continue
            for j in range(len(table[i])):
                table[i][j] -= factor * table[leave][j]
        factor = zrow[enter_col]
        if factor != 0:
            for j in range(len(zrow)):
                zrow[j] -= factor * table[leave][j]
        bcols[leave] = enter_col
        pivots += 1
        if max_nodes is not None and pivots > max_nodes:
            return "node_limit"
        return "ok"

    # 第一阶段：缩减成本 >0 的非人工列入基可使目标下降；只有人工列 t 禁止入基。
    # 翻转 le 的剩余列 r 是普通非负变量，标准形式（系数 -1 + 人工取基）下
    # 可安全参与转轴，禁止它会在可行系统上误判不相容。
    phase1_allowed = {j for j in range(width - 1) if cols[j][0] != "t"}
    while True:
        enter = next((j for j in range(width - 1)
                      if j in phase1_allowed and z[j] > 0), None)
        if enter is None:
            break
        rc = pivot(enter, T, z, basis, live)
        if rc != "ok":
            return LPResult(status="node_limit" if rc == "node_limit"
                            else "unbounded", pivots=pivots)
    z0 = z[-1]

    if z0 > 0:
        # 不相容。第一阶段最优值 z0>0 已证明系统空；Farkas 乘子不在表内凑符号，
        # 而由独立的转置 LP farkas_certificate 恢复，保证 le 乘子非负、
        # 组合左端非负、右端严格负，且可被外部独立复算。
        cert = farkas_certificate(rows, max_nodes=max_nodes)
        assert cert is not None, "不相容系统的 Farkas 恢复失败"
        return LPResult(status="infeasible", certificate=cert,
                        pivots=pivots)


    # z0 == 0：可行。把仍以零值留在基中的人工变量换出。
    # 这是退化转轴（该行右端为 0）：候选列即便系数为负，也直接在本人工行上
    # 转轴，右端 0/系数 仍为 0，不破坏 BFS 可行性。整行在非人工列上全为零时
    # 才是冗余 0=0 行，连同基登记一并剔除。
    def degenerate_pivot(row_idx, enter_col):
        piv = T[row_idx][enter_col]
        T[row_idx] = [v / piv for v in T[row_idx]]
        for i in live:
            if i == row_idx:
                continue
            factor = T[i][enter_col]
            if factor == 0:
                continue
            for j in range(width):
                T[i][j] -= factor * T[row_idx][j]
        factor = z[enter_col]
        if factor != 0:
            for j in range(width):
                z[j] -= factor * T[row_idx][j]
        basis[row_idx] = enter_col

    for i in list(live):
        if cols[basis[i]][0] != "t":
            continue
        cand = next((j for j in range(width - 1)
                     if cols[j][0] != "t" and T[i][j] != 0), None)
        if cand is None:
            live.remove(i)
            continue
        degenerate_pivot(i, cand)

    # 丢弃人工列，进入第二阶段。live 已剔除冗余行，存活行按其顺序连续编号，
    # 每个存活行的基列此时都应是非人工列（人工变量已换出或随冗余行剔除）。
    keep = [j for j in range(width - 1) if cols[j][0] != "t"]
    T = [[row[j] for j in keep] + [row[-1]] for row in T]
    T = [T[i] for i in live]
    basis = {idx: keep.index(basis[old]) for idx, old in enumerate(live)}
    cols2 = [cols[j] for j in keep]
    width2 = len(keep) + 1

    # 第二阶段目标行字典化：z2 右端 = -c·x，缩减成本 <0 的列入基
    z2 = [Fraction(0)] * width2
    for j, v in enumerate(c):
        z2[j] = v
    rows2 = list(range(len(T)))
    for i in rows2:
        factor = z2[basis[i]]
        if factor == 0:
            continue
        for j in range(width2):
            z2[j] -= factor * T[i][j]

    while True:
        enter = next((j for j in range(width2 - 1) if z2[j] < 0), None)
        if enter is None:
            break
        rc = pivot(enter, T, z2, basis, rows2)
        if rc != "ok":
            return LPResult(status="node_limit" if rc == "node_limit"
                            else "unbounded", pivots=pivots)

    x = [Fraction(0)] * n
    for i, bc in basis.items():
        if cols2[bc][0] == "x":
            x[cols2[bc][1]] = T[i][-1]
    x = tuple(x)
    # 字典式 z2 行满足 z2[-1] = -c·x，最小化最优值取其相反数
    obj = -z2[-1]
    assert obj == sum(c[j] * x[j] for j in range(n)), "目标值与见证重算不一致"
    return LPResult(status="optimal", objective=obj, x=x, pivots=pivots)


def minimize(rows, c, max_nodes=None):
    return solve(rows, tuple(c), max_nodes=max_nodes)


def maximize(rows, c, max_nodes=None):
    res = solve(rows, tuple(-v for v in c), max_nodes=max_nodes)
    if res.status == "optimal":
        res.objective = -res.objective
    return res


def farkas_certificate(rows, max_nodes=None):
    """对不相容系统独立求 Farkas 乘子（转置 LP，与两阶段表解耦）。

    原始系统：变量 x>=0，约束 le（乘子非负）/ eq（乘子自由）。
    构造关于乘子的可行性问题，变量全部非负：
      每个 le 行一个 beta_k >= 0；
      每个 eq 行拆成 beta_k = p_k - q_k，p_k,q_k >= 0。
    约束（全部写成 le，变量在左）：
      对每个原变量 j: -sum_k beta_k A_kj <= 0   （即 g_j >= 0）；
      归一化严格负: -sum_k beta_k b_k >= 1，即 sum_k beta_k b_k <= -1。
    该系统可行当且仅当原系统不可行（Farkas 备选定理）。解出后把 eq 的
    p-q 合回自由乘子，并在原始行上重算 g、h 返回。
    """
    m = len(rows)
    n = len(rows[0].a)
    # 乘子变量布局：按行顺序，le 占 1 个，eq 占 2 个（p,q）
    var_index = []
    dim = 0
    for r in rows:
        if r.rel == "le":
            var_index.append((dim, None))
            dim += 1
        else:
            var_index.append((dim, dim + 1))
            dim += 2

    def beta_coeff(k):
        """第 k 行乘子 beta 在乘子变量向量中的系数（le: 单位；eq: p-q）。"""
        p, q = var_index[k]
        return p, q

    cert_rows = []
    # g_j >= 0 <=> -sum beta A <= 0
    for j in range(n):
        coef = [Fraction(0)] * dim
        for k, r in enumerate(rows):
            p, q = beta_coeff(k)
            coef[p] += -r.a[j]
            if q is not None:
                coef[q] -= -r.a[j]  # - (p-q)*A 的 q 项：+A
        cert_rows.append(Row(f"farkas:g[{j}]>=0", tuple(coef), "le", Fraction(0)))
    # sum beta b <= -1
    coef = [Fraction(0)] * dim
    for k, r in enumerate(rows):
        p, q = beta_coeff(k)
        coef[p] += r.b
        if q is not None:
            coef[q] -= r.b
    cert_rows.append(Row("farkas:h<=-1", tuple(coef), "le", Fraction(-1)))

    res = solve(cert_rows, tuple(Fraction(0) for _ in range(dim)),
                max_nodes=max_nodes)
    if res.status != "optimal":
        return None
    v = res.x
    beta = []
    for k, r in enumerate(rows):
        p, q = beta_coeff(k)
        beta.append(v[p] if q is None else v[p] - v[q])
    g = [Fraction(0)] * n
    for k, r in enumerate(rows):
        for j in range(n):
            g[j] += beta[k] * r.a[j]
    h = sum(beta[k] * rows[k].b for k in range(m))
    for k, r in enumerate(rows):
        if r.rel == "le":
            assert beta[k] >= 0, f"恢复的 le 乘子为负: {r.name}"
    assert all(x >= 0 for x in g), "恢复的组合左端存在负分量"
    assert h <= -1, "恢复的组合右端未严格为负"
    return {
        "multipliers": [
            {"row": rows[k].name, "relation": rows[k].rel, "beta": beta[k]}
            for k in range(m)
        ],
        "combined_lhs": tuple(g),
        "combined_rhs": h,
    }
