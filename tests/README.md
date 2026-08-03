# Tests

测试按职责组织，不按单个实验脚本组织：

```text
tests/
├── unit/
│   ├── core/          # contracts、坐标、路由策略和 controller primitives
│   ├── backends/      # verifier backend
│   ├── experts/       # Grounder 和 oracle
│   ├── workers/       # worker engine、endpoint 与 JSONL transport
│   └── benchmarks/    # benchmark adapter、生成规则和阈值搜索
├── integration/
│   └── runtime/       # 多组件或子进程交互
├── regression/        # 已修复行为的回归测试
└── manual/
    ├── volcano/       # 基础 Volcano 手工运行脚本
    └── grounding_control/ # verifier/worker/Volcano 组合 smoke
```

默认测试只收集 `test_*.py`，并排除 `tests/manual/`：

```bash
pytest
pytest tests/unit
pytest tests/integration
```

新增测试时遵循以下规则：

- 单一组件、CPU 可重复、无外部服务依赖的测试放入 `unit/`。
- 涉及真实子进程、组件边界或传输协议的测试放入 `integration/`。
- 针对已经修复的问题建立的最小复现放入 `regression/`。
- 需要 GPU、模型权重或人工查看输出的程序放入 `manual/`，文件名不要以 `test_` 开头。

Volcano 手工脚本从仓库根目录运行，例如：

```bash
python tests/manual/volcano/quickstart.py
python tests/manual/volcano/random_coordinate.py --seed 2026
python tests/manual/volcano/counterfactual_coordinate.py --perturb-seed 2026
```
