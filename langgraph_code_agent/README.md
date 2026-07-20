# LangGraph Code Agent

这是 `编写agent.html` 配套的可运行示例。Agent 只能在 `--workspace` 指定目录内读写；模型通过 `OPENAI_API_KEY` 调用 OpenAI。

## 快速开始（PowerShell）

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:OPENAI_API_KEY="你的 API Key"
$env:OPENAI_MODEL="gpt-5-mini"
python agent.py --workspace .\workspace --thread-id demo
```

可以输入：

```text
检查 calculator.py 和现有测试，修复除零行为：除数为 0 时抛出 ValueError，并补充测试，然后运行 pytest。
```

## 安全边界

- 路径会解析并校验，拒绝访问工作区以外位置。
- `.env`、私钥和常见凭据文件不能由 Agent 读写。
- 命令不经过 shell，且首个程序必须出现在 `AGENT_ALLOWED_COMMANDS` 白名单中。
- 这仍不是强隔离沙箱。生产环境应在容器/临时工作目录中运行，并给写文件、执行命令等操作增加人工审批。
