# jevkit（中文文档）

[English README](./README.md)

**Jev / Kev（System One）政策层 + 评测层统一框架。**
一句话：非结构化状态进，带概率的类型化决策出；**模型负责不确定性，代码负责政策**——本框架就是那段代码。

[Jev](https://typesafe.ai)（TypeSafe 出品）把"问一次大模型"变成了一个返回类型化概率的端点（`POST /v1/systemone`，三原语 Choice / Score / Noul，输出免费、70–500ms）；[Kev](https://github.com/jaredpalmer/kev) 是 Apache-2.0 的开源对等实现，API 与 TypeSafe 完全兼容。但两者都只给你概率，**从不告诉你该干什么**。把概率翻译成动作（政策层）、验证概率本身可信吗（评测层）、以及让两者咬合成闭环，就是 jevkit 做的事。

```
客户端层   POST /v1/systemone，拿回概率          商品：官方 typesafe-sdk、kev.serve 已有
政策层     概率 → 动作（数据 + 纯函数）          jevkit.policy / jevkit.record
评测层     概率可信吗？阈值定在哪？              jevkit.metrics / calibration / permute / compile
```

统一闭环：

```
标注数据(kev train 格式) ──compile──▶ policy.lock.json（阈值+证据）
                                          │ 运行时加载
                                          ▼
                                    decide() → decisions.jsonl
                                          │ 回流标注
                                          ▼
                                   check 漂移检查 ──超界──▶ 重新编译
```

## 设计立场（每条都对应真实踩坑）

- **政策 = 数据 + 纯函数，不是 DSL。** 阈值来自你的误报/漏报成本，同一概率在路由和封号场景是两种政策——框架提供结构、证据和审计，不替你做业务判断。
- **一切评测分题型。** 三种题型校准误差差一个数量级（Noul 0.012 / Choice 0.086 / Score 0.254），一个总阈值覆盖不了所有题型。
- **日志必存完整概率分布。** 生产日志必须能回答："当时是 0.91 对 0.05，还是 0.36 对 0.34？"模型版本 / 题集版本 / 政策快照 / 实际动作一个不缺。
- **零必需依赖。** 核心纯 stdlib（Python ≥ 3.10）；官方 SDK 是可选 extra，本地 kev 服务用内置 HTTP 后端直连。
- **kev 原生适配。** 标注数据格式就是 `kev.train` 格式（同一份语料既微调又评测）；kev 独有的 `/v1/systemone/permute` 端点可用时走服务端排列测试。

## 安装

```bash
pip install jevkit                    # 核心（零依赖）
pip install 'jevkit[typesafe]'        # + 官方 typesafe-sdk 适配
pip install 'jevkit[dev]'             # + pytest
```

## 三分钟上手

最快的入口是 [`examples/`](./examples/README.md) 教程——四个循序渐进的脚本，mock 后端离线可跑。命令行一览：

```bash
jevkit demo                                    # ① mock 后端跑通三段式路由
jevkit calibrate --data labeled.jsonl          # ② 分题型校准体检 + 温度
jevkit coverage  --data labeled.jsonl          # ③ 覆盖率—准确率曲线
jevkit compile  --data labeled.jsonl \
      --template policy-template.json \
      --budget 0.05                            # ④ 按错误预算编译阈值 → lock
jevkit check    --data new-week.jsonl \
      --lock policy.lock.json                  # ⑤ 漂移检查（漂移退出码 1）
```

库 API 完整示例见 [`examples/triage_router.py`](./examples/triage_router.py)。

## 后端

| spec | 实现 | 用途 |
|---|---|---|
| `http://127.0.0.1:8009` | `HttpBackend`（stdlib urllib） | **kev.serve** 或任何兼容端点；读 `x-typesafe-request-id` 响应头；用 kev 原生 permute 端点 |
| `typesafe://jev-1.13.0` | `TypesafeSdkBackend`（官方 SDK，懒导入） | 官方托管；复用 SDK 鉴权（`TYPESAFE_API_KEY`）与 429/529 退避 |
| `mock` | `MockBackend` | 测试与 demo；确定性（同一 state 同一概率），`bias=` 模拟位置偏置 |

跑开源模型：

```bash
git clone https://github.com/jaredpalmer/kev && cd kev
uv sync --extra serve
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009
```

**注意**：kev 校准落后于 Jev（5% 预算下可自动化比例 0.45–0.57 vs 0.70），**阈值不可跨模型迁移**——换模型必须重新编译。

## 数据格式（与 kev.train 共享）

每行一次完整请求 + 每题内联 `label`——与 `kev.data.load_records` 的原生格式一致，**同一份语料可直接喂 `kev.train` 微调，也喂 jevkit 评测**。可选 `answers` 字段记录预测，离线评测免费：

```jsonc
{"state": "Order #4411 was charged twice...",
 "questions": {
   "department": {"type": "choice", "instructions": "Which team?",
                  "criteria": {"returns": "…", "shipping": "…", "billing": "…"},
                  "label": "billing"},
   "escalate": {"type": "noul", "instructions": "Urgent?", "label": false}},
 "answers": {"department": {"type": "choice", "choice": "billing",
                            "confidence": 0.88, "probabilities": {"…": "…"}}}}
```

label 语义与 kev 一致：choice → 选项名；noul → true/false；score → 零基档位索引。

## CLI 参考

| 命令 | 作用 |
|---|---|
| `jevkit demo` | mock 后端跑通三段式路由，产出 `demo-decisions.jsonl` |
| `jevkit calibrate --data F` | 分题型 ECE / Brier / LogLoss + 温度拟合（对半分割防自欺） |
| `jevkit coverage --data F --budget 0.05` | 覆盖率—准确率表；错误预算下可自动化比例与最优阈值 |
| `jevkit permute --data F --policy P --n-perm 6` | 打乱选项顺序：漂移 / KL / **动作翻转率**（相对政策阈值） |
| `jevkit compile --data F --template T --budget b` | 每 gate 的 auto 档阈值 ← `argmax coverage s.t. err ≤ b`，产 lock（含证据） |
| `jevkit check --data F --lock L` | 新数据 vs lock 证据：准确率/覆盖率/ECE 漂移检查，漂移退出码 1 |

## 包结构

包结构与三层架构一一对应，依赖方向严格为 `core ← policy ← eval`：

```
jevkit/
├── errors.py        异常层级（跨层共享）
├── core/            /v1/systemone 的客户端抽象
│   ├── types.py       Choice / Score / Noul 三原语 + 类型化答案（kev 精确契约）
│   └── backend.py     Backend 协议 + HttpBackend（kev.serve，原生 permute）
│                      + TypesafeSdkBackend（官方 SDK，可选）+ MockBackend
├── policy/          概率 → 动作
│   ├── policy.py      Tier / Gate / Policy 数据 + decide() 纯函数
│   └── record.py      DecisionRecord + JsonlLedger（完整概率审计日志）
├── eval/            概率可信吗？阈值在哪？
│   ├── data.py        kev.train 格式标注数据（一份语料两用）
│   ├── metrics.py     ECE / Brier / logloss / 覆盖率 / 预算选阈值
│   ├── calibration.py 分题型温度缩放（对半分割纪律）
│   ├── permute.py     选项顺序稳定性（含政策动作翻转率）
│   ├── compile.py     阈值 → 带证据的 policy.lock + 漂移检查
│   └── synthetic.py   过度自信合成数据（测试与演示）
└── cli.py           jevkit demo / calibrate / coverage / permute / compile / check
```

公共 API 在包根保持稳定：`from jevkit import Choice, Policy, decide, make_backend, …`；也可以直接导入子包（`jevkit.core` / `jevkit.policy` / `jevkit.eval`）。

## 关键数字（校准时心里要有）

| 项 | 值 |
|---|---|
| 价格 | 输入 $0.042/MTok，输出免费 |
| 延迟 | 70–500 ms（前沿 LLM 同任务 3–329 s） |
| 校准 | conf≥0.9 → 准确率 ~92%，覆盖 ~73%（PrimeLine 预注册测试） |
| 分题型校准误差 | Noul 0.012 / Choice 0.086 / Score 0.254 |
| 5% 错误预算可自动化 | Jev 0.70；Kev-9B 0.45–0.57 |
| 一致率 | Kev-9B base 66.4% → Nimble 90.1% → Jev 93.2%（324 条 held-out） |
| 上下文 | 总 64k；state + 最长问题 ≤ 32k（kev serving 仅 8k） |
| 版本 | 生产锁 `jev-1.13.0`；`-latest` 漂移会悄悄跨过阈值 |

## 兼容性验证

契约兼容用 **kev 的真实源码**验证（`tests/test_kev_compat.py`，`KEV_SRC` 环境变量开启）：

- jevkit 的题目序列化能被 kev 的 `SystemOneRequest`（pydantic）和 `to_record` 编码器接受，含任意 JSON 的 instructions/criteria 描述与边界候选数（255 选项、单档 score）
- kev 真实答案序列化器（`to_answers`）的输出能被 jevkit 解析
- jevkit 生成的标注数据能被 `kev.data.load_records` 直接读取（可直接喂 `kev.train`）

设置 `JEVKIT_KEV_BASE_URL` 指向运行中的 kev.serve 可做在线验证。

## 开发

```bash
git clone https://github.com/JasmineAIGC/jevkit && cd jevkit
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
pytest                                   # 全量测试（mock 后端，无需任何服务）

# kev 兼容（用 kev 真实代码做契约测试）：
git clone --depth 1 https://github.com/jaredpalmer/kev /tmp/kev
pip install pydantic datasets
KEV_SRC=/tmp/kev pytest tests/test_kev_compat.py

python examples/make_example_data.py     # 重新生成 examples/data/
```

## License

MIT
