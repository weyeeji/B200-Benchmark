# Benchmark Agent 指南

- 通过 `ssh cqzs` 在 `~/Benchmark` 跑测，使用 `.venv/bin/python`。开跑前核对 API、模型、GPU 数量和测试窗口。
- 先用 `benchmark.py plan` 校验冻结数据。对比部署时复用同一数据和请求顺序。P03/P04 沿用首轮实测容量或指定的组数；需要延长发包时限时，用 `pilot` 确定统一时限。
- 通过配置的 Hook 清理 KV Cache，无法确认时记录 `unverified`。服务端部署和配置由操作者管理。
- 所有测试结果、原始请求和日志写入 Git 已忽略的 `results/`。提交前检查暂存区，确保其中只有本次要上传的项目文件。
- 每次修改压测代码后，完成相关验证并提交、推送到 GitHub。服务器无法直连时，用 Git bundle 经本机中转，并核对 GitHub 上的提交 SHA。
