# LangGraph Cursor Code Agent

`编写agent_cursor.html` 的配套可运行示例：一个用 **LangGraph 1.x** 编写、能读代码、改代码、跑测试的工程 Agent。它演示了完整流程：

- **自定义状态** `AgentState`（`messages` + 累积的 `changed_files` / `commands_run`，各自配 reducer）
- **工作区隔离工具**：`list_files` / `read_file` / `search_text` / `write_file` / `run_command`
- **会写状态的工具**：`write_file`、`run_command` 通过 `Command` + `InjectedToolCallId` 把副作用回写进状态
- **人工审批（可选）**：置 `AGENT_REQUIRE_APPROVAL=1` 后，写文件/执行命令前用 `interrupt` 暂停等待确认
- **线程级记忆**：checkpointer + `thread_id`，可多轮继续
- **流式 CLI**：逐步观察模型决策与工具执行

模型通过 `OPENAI_API_KEY` 调用外部 OpenAI 模型；本地工具负责真正的文件与命令执行。

## 目录结构

```text
langgraph_cursor_agent/
├── agent.py              # 状态、工具封装、图、审批、CLI
├── sandbox.py            # 零第三方依赖的安全工作区（路径/敏感文件/命令白名单）
├── requirements.txt
├── .env.example
├── README.md
├── tests/
│   ├── test_sandbox.py   # 仅需标准库 + pytest，验证安全边界
│   └── test_graph.py     # 用假模型驱动完整图循环与审批路径（需已装 langgraph）
└── workspace/            # Agent 唯一允许操作的演示目录
    ├── stringkit.py          # 内置一个真实 bug 的字符串工具
    └── test_stringkit.py     # 能捕获该 bug 的测试
```

## 快速开始（PowerShell）

> 注意：`langchain-openai` / `langgraph` 依赖 `tiktoken`、`ormsgpack`，它们在 **32 位 Python** 上没有预编译 wheel、需要 Rust 才能从源码构建。请使用 **64 位 Python 3.10+**。

```powershell
cd D:\AIResearch\langgraph_cursor_agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

$env:OPENAI_API_KEY="替换为你的真实密钥"
$env:OPENAI_MODEL="gpt-5-mini"
python agent.py --workspace .\workspace --thread-id demo
```

### macOS / Linux

```bash
cd langgraph_cursor_agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

export OPENAI_API_KEY="替换为你的真实密钥"
export OPENAI_MODEL="gpt-5-mini"
python agent.py --workspace ./workspace --thread-id demo
```

## 演示任务

`workspace/stringkit.py` 里的 `truncate` 有一个真实 bug：截断时直接拼接后缀，导致结果长度超过 `max_length`。对应测试 `test_stringkit.py` 会失败。把下面的话发给 Agent：

```text
先运行 pytest 找出失败的测试，然后修复 stringkit.py 中 truncate 的 bug，
使结果长度不超过 max_length（要为后缀留出空间），最后重新运行 pytest 确认全部通过。
不要修改测试，最后说明改了什么和验证结果。
```

预期轨迹：`list_files/read_file` → `run_command("python -m pytest -q")`（红）→ `write_file` 修复 → `run_command` 再跑（绿）→ 总结。

CLI 命令：`/state` 查看已保存的消息数、改动文件、执行过的命令；`/new` 新建会话；`/quit` 退出。

## 人工审批模式

```powershell
$env:AGENT_REQUIRE_APPROVAL="1"
python agent.py --workspace .\workspace --thread-id demo
```

此时每次 `write_file` / `run_command` 前，CLI 会打印待执行的操作并等待 `y/N`。拒绝时 Agent 会收到“被拒绝”的工具结果并重新规划。

## 测试

```powershell
# 安全边界测试：仅需标准库 + pytest，任何环境都能跑
python -m pytest tests\test_sandbox.py -q

# 图循环 + 审批测试：用假模型，无需 API Key，但需要已安装 langgraph
python -m pytest tests -q
```

`test_graph.py` 通过注入一个 `FakeModel` 来驱动 `模型 → 工具 → 模型 → END` 的完整循环，并覆盖审批同意/拒绝两条分支，因此**不消耗 token、不联网、不需要密钥**。

## 安全边界（务必阅读）

- 所有路径先 `resolve()` 再校验父目录，拒绝 `..` 穿越、绝对路径与工作区外访问。
- `.env`、私钥、`credentials.json` 等敏感文件在读写入口都被拦截。
- 命令不经过 shell（`shell=False`），首个程序必须命中 `AGENT_ALLOWED_COMMANDS` 白名单，并有超时与输出截断。
- **这仍不是强隔离沙箱。** 允许 `python` 就意味着被运行的脚本仍能执行任意系统调用。生产环境应叠加容器 / 低权限用户 / 临时工作目录 / 禁网 / 资源配额 / 人工审批。
- 不要把真实 API Key 写进代码、`.env.example` 或 Git；用环境变量或密钥管理服务，并设置额度与轮换。
