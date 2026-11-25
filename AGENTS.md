# Benchmark 项目约定

- 通过 `ssh cqzs` 在 `gxeng:/home/user/Benchmark` 操作，使用项目的 `.venv/bin/python`。
- 使用 `benchmark.py plan` 展开参数并核对 API、模型与实验档位。
- P01、P02 的固定长度以服务端 Usage 核验。对比不同部署时记录各档的数据集种子和标识、模型和服务配置。
- 将数据、日志、结果、环境和本地配置留在本机，提交前核对暂存内容。
- 代码验证后提交、推送到当前开发分支。如果服务器无法访问 GitHub，使用 Git bundle 经本机中转，并核对目标分支和提交 SHA。
