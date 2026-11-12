"""命令行：inspect 输出原始资料概况，analyze 输出偏好敏感性分析。"""
import argparse
import json

from .domain import summarize
from .io import load
from .polyhedron import Problem
from .sensitivity import analyze


def main():
    p = argparse.ArgumentParser(description="权重多面体稳健排序与偏好敏感性分析")
    p.add_argument("command", choices=["inspect", "analyze"])
    p.add_argument("file")
    a = p.parse_args()
    v, digest = load(a.file)
    if a.command == "inspect":
        out = {"source_sha256": digest, "summary": summarize(v)}
    else:
        problem = Problem(v)
        out = {"source_sha256": digest, "analysis": analyze(problem)}
    print(json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
