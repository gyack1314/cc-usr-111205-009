# 避险准备方案的权重多面体稳健排序

这是独立的防汛演练资料Python项目。`data/`保存虚构人数、观测、时序或方案，`response_data.io`提供严格数值读取与来源字节SHA-256，`response_data.domain`给出已有原始摘要，`response_data.cli`输出JSON。字段、量纲与分析口径见[资料约定](docs/data-contract.md)。这些数据用于离线复盘，不连接真实人员、传感器或指挥系统。

Python 3.11及以上即可直接使用，无第三方Python依赖。在项目根目录执行：

```sh
python -m unittest discover -s tests -v
python -m compileall -q response_data tests
python -m response_data.cli inspect data/normal.json
python -m response_data.cli analyze data/normal.json
```

`inspect`显示原始资料概况，不覆盖源文件。`analyze`输出权重多面体上的偏好敏感性分析：先检查 W 是否非空（空集给可独立复算的 Farkas 不相容证据，不输出空集上的全称结论），再给每对方案在同一 W 上的差区间与见证、最坏遗憾（真实见证权重，同值按 ID 决胜）、每个方案能否并列第一（可行见证或排除证据），小维度另用全部顶点枚举交叉核对。全部为精确有理数运算，未完成的 LP 不参与“已证明必然”关系。结果是偏好敏感性信息，最低遗憾不等于现实最安全，也不构成调度命令。`invalid-`前缀资料为不合法输入，其余变体使用相同字段保留不同业务边界。数值null须按字段语义区别于0，不在读取时自动填补。

Dockerfile固定基础镜像，可用`docker build -t response-data .`及`docker run --rm response-data`读取资料；`docker run --rm response-data python -m unittest discover -s tests -v`执行已有检查。运行所需资料全部在本地，无需登录或实时来源。
