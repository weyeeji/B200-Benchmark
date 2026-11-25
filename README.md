# CQ-B200 Benchmark

在 gxeng 上通过 NVIDIA AIPerf 压测 DeepSeek-V4.1-Flash-1017 的 OpenAI 兼容流式 Chat Completions 服务。客户端通过 URL 接入已有部署。

## 测试方案

在 TTFT、TPOT 全部达到 SLA 的条件下，逐档找出每类负载已测的最高可承载并发。P03 额外以真实流量的 **KVCache 命中率 90% 以上** 为优化目标。SLA 通过只看下方的六项时延阈值。

被测模型为 **DeepSeek-V4.1-Flash-1017**。四卡 vLLM 的 C16 验证使用 gxeng 上的 NVIDIA B200 GPU 3–6，部署为 TP1／Attention DP4／EP4。后续对比按实际测点记录硬件与部署形态。

| 目录 | 工作负载 | 预热 | 正式阶段 |
| --- | --- | --- | --- |
| P01 | 重 Prefill，8192 输入 token，1024 输出 token | 10 条独立随机请求 | 闭环，并发 C，共 3 × C 条 |
| P02 | 重 Decode，1024 输入 token，4096 输出 token | 10 条独立随机请求 | 闭环，并发 C，共 3 × C 条 |
| P03 | AgentX，回放 SemiAnalysis 真实会话轨迹及子代理 | 轨迹快照引导，另加每路 10 条缓存压力预热请求 | C 路会话，发流 900 秒，最多再等 600 秒收尾 |

每项实验按 16、32、64、128、256 独立爬坡。首个 SLA 未达标或该档无法完成有效测量时，停止该实验的后续档位。最高通过档是本次已验证的并发。如果最高计划档仍通过，可以用更高档位继续测量。

## 使用

~~~bash
ssh cqzs
cd /home/user/Benchmark

# 打印完整计划和每档的 AIPerf argv
.venv/bin/python benchmark.py plan
.venv/bin/python benchmark.py plan --experiments P01 P02 --concurrency 16 32 64

# 正式运行示例。把地址和模型 ID 换成目标服务实际值，P03 按默认使用完整语料
.venv/bin/python benchmark.py run \
  --url http://inference.example:8000 \
  --model DeepSeek-V4.1-Flash-1017 \
  --metrics-url http://inference.example:8000/metrics
~~~

| 参数 | 默认或用法 |
| --- | --- |
| `plan` / `run` | `plan` 展开工作负载、输出路径和 AIPerf 参数。 `run` 发送请求并生成报告。 |
| `--experiments` | 默认按 P01、P02、P03 顺序运行。可选择其中一个或多个，也可指定执行顺序。 |
| `--concurrency` | 默认 `16 32 64 128 256`。传递递增且不重复的正整数。P01/P02 的 C 是最大在途请求数。P03 的 C 是并行轨迹路数，每路可包含子代理。 |
| `--url` | `run` 必填。传 API 基础地址，AIPerf 按 Chat 端点追加 `/v1/chat/completions`。 |
| `--model` | 默认为 `DeepSeek-V4.1-Flash-1017`，填写目标服务实际注册的模型 ID。 |
| `--max-context-length` | 仅 P03。按轨迹记录的峰值输入加输出 token 剔除整条超限轨迹。省略时使用完整语料。服务有上下文限制时填实际值。 |
| `--metrics-url` | 可选 Prometheus 地址，提供 KVCache 和推测解码计数器的备用来源。建议使用与目标实例和模型对应的指标地址。 |
| `--seed` | 默认 42。P01/P02 用于随机文本，P03 用于轨迹回放的随机选择。 |
| `--dataset-nonce` | 用于 P01/P02 区分运行。默认取运行时间戳，生成新的冷前缀。跨独立缓存的部署做对比时，可复用相同的 nonce、seed 和档位。 |
| `--request-timeout` | 单个 HTTP 请求超时时间，默认 600 秒。 |
| `OPENAI_API_KEY` | 有鉴权时通过环境变量传入。 |

### P01 和 P02

可调整的工作负载参数在各目录的 `workload.json`。

| 参数 | 用途 |
| --- | --- |
| `input_tokens` | 单请求完整聊天模板渲染后的输入长度，P01 为 8192，P02 为 1024。 |
| `output_tokens` | 输出长度，P01 为 1024，P02 为 4096。每行 JSONL 的 `output_length` 提供请求输出上限，`min_tokens` 为同一值，`ignore_eos=true` 用于生成到此长度。 |
| `warmup_requests` | 每档先发 10 条完整长度的请求，预热并发为 `min(C, 10)`。预热使用单独的数据文件与 AIPerf 进程。 |
| `requests_per_concurrency` | 默认 3。正式请求数为 C 乘以该值。 |
| `single_turn` / `sequential` | 顺序读取独立单轮请求，每条发一次。 |
| `chat_template_kwargs` | P01/P02 固定 DeepSeek 的思考模板与 `high` 推理强度。生成时用本地 DeepSeek V4.1 renderer 按完整模板精确计数。 |
| `cache_salt` | 每条请求有独立、可复现的缓存命名空间。支持该字段的引擎可从第一块隔离缓存。 |

随机用户文本在输入开头分散前缀。预热、正式阶段和不同档用独立序列。正式请求用服务端的 `usage.prompt_tokens` 和 `usage.completion_tokens` 核验定长合同。请求数、服务端 Usage 或长度无法核验的档位会标为无效。实际缓存命中展示在结果中。

### P03

可调整的工作负载参数在 `P03/workload.json`。

| 参数 | 用途 |
| --- | --- |
| `scenario` | `agentx` 使用 AIPerf 原生多轮闭环回放。按记录保留轨迹时序和子代理依赖。 |
| `public_dataset` | `semianalysis_cc_traces_weka_062126`，日期固定的 Weka trace 语料，共 393 条根轨迹。 |
| `profile_seconds` | 900 秒，正式阶段发新请求的时间。 |
| `drain_seconds` | 600 秒，发流停止后等待已派发请求的时间。 |
| `warmup_requests_per_lane` | 快照引导之后，每路再发送最多 10 条缓存压力预热请求。 |
| `warmup_grace_seconds` | 1800 秒，原生轨迹预热的排空上限。 |
| `cache_bust` | `first_turn_prefix` 给每次播放的首轮加唯一前缀，保留同一轨迹多轮间的缓存复用。 |
| `ignore_eos` | 场景使用 `true`，沿用轨迹每一轮的输出预算。 |

P03 按固定时长回放，393 条根轨迹构成回放池。并发 C 表示可同时活跃的会话树数量，子代理可产生额外的在途请求。900 秒到点后停止派发新请求，收尾最多等待 600 秒，已发请求全部结束即可提前退出。收尾期间完成的请求计入正式结果。

本项目的 900 秒发流、600 秒收尾沿用内部测试方案。AIPerf 0.14.0 的 `agentx` 预设默认分别为 3600 秒和 30 秒。跨版本对比时要匹配发流与收尾时长。 [AIPerf 0.14.0 AgentX 指南](https://github.com/ai-dynamo/aiperf/blob/v0.14.0/docs/tutorials/agentx-mvp.md)

AIPerf 0.14.0 的 AgentX 预设还使用 0.25–0.75 的轨迹起点范围、300 秒的单轨迹空闲间隔上限、10 秒的系统空闲间隔上限，以及 10% 的运行时失败保护阈值。实时统计间隔为 30 秒，指标按 1 秒切片。原生导出保留场景有效性和过滤信息。

### 公共 AIPerf 设置

- `endpoint-type=chat`，`streaming` 开启，`tokenizer` 指向项目内的本地模型分词器。
- `use-server-token-count` 请求并使用服务端流式 Usage。 `export-level=records` 导出聚合统计和逐请求指标。
- `artifact-dir` 按运行、实验、档位和阶段分开。命令行使用 `ui=none`，控制台输出写入日志。GPU 遥测和 AIPerf 自带绘图关闭。
- `server-metrics` 的自动抓取关闭。传 `--metrics-url` 时由脚本在测量前后抓取计数器。P01/P02 的抓取窗口覆盖正式阶段，P03 覆盖 AIPerf 原生预热到排空的整个过程。

## 指标与 SLA

本项目参考 GLM 第三方 Token 供应验收口径，统一采用下表的阈值。

| SLA | P50 | P90 | P99 |
| --- | ---: | ---: | ---: |
| TTFT | 5 s | 10 s | 30 s |
| TPOT | 30 ms | 50 ms | 80 ms |

- **TTFT** 取请求开始到首个有效推理或正文 token。**TPOT** 取每个请求首 token 后的完整输出时长除以剩余的输出 token 数（含推理 token），对逐请求结果计算百分位。六项阈值均满足时判为通过。
- **输入、输出、总 TPM** 使用正式阶段服务端 Usage 汇总，以正式阶段首个请求到最后一个响应的实测时间（含收尾）换算每分钟。
- **KVCache 命中率** 优先用正式请求的服务端已缓存输入 token 总数除以输入 token 总数。
- **推测解码接受率** 为接受的草稿 token 除以提议的草稿 token。**平均接受长度** 为每次验证平均接受的草稿 token 加 1。优先使用请求指标，备用计数器的范围和来源会标记。缺少数据时显示 N/A。

## 版本对比

先选定基线运行，记录 Git 提交、测试工具版本、运行 ID 和生效参数。候选版本沿用下列条件，并在结果中注明本轮调优变量。

| 对比条件 | 执行与记录 |
| --- | --- |
| 硬件与部署 | 记录 GPU 型号、数量、卡号、拓扑与通信方式，模型权重版本、精度、分词器和聊天模板，推理引擎、镜像版本、完整启动参数及路由。做同资源对比时保持硬件预算一致。 |
| 测点与统计 | 对照相同场景和相同并发档位，保持预热、请求预算、超时、SLA 和统计窗口一致。每档记录成功、失败、取消与上下文跳过数量，以及首个超限或无效档位。 |
| P01／P02 冷算 | 在独立缓存环境中复用同一 `--seed`、`--dataset-nonce` 和并发档位。核对预热与正式数据文件的 SHA-256、服务端实际 token 长度和 KVCache 命中情况。 |
| P03 轨迹回放 | 保持数据集版本、seed、上下文过滤、发流与收尾时长、预热预算、会话路由和缓存起点一致。记录路由亲和设置和实际命中率。 |
| 推测解码 | 记录草稿方法、每轮草稿数、真实或模拟接受方式，以及实测接受长度和接受率。需要控制接受率影响时，可在部署端把平均接受长度设为 **3.5 token/轮**，同组各版本采用相同方式。 |
| 指标来源 | 固定服务端 Usage、Prometheus 指标和统计范围。P03 的 KVCache 命中率优先按正式请求的 Usage 计算，回退到 Prometheus 时记录其窗口含预热。 |

P03 参照 InferenceX 扫点时，每档重启服务并使用新输出目录，通过 `--experiments P03 --concurrency C` 单独运行。当前脚本在同一实例上连续扫档时，需要固定扫档顺序和初始缓存状态，并在记录中注明。如果要复现已发布曲线，按对应测点的配方匹配时长、预热和路由。 [InferenceX AgentX 独立运行指南](https://github.com/SemiAnalysisAI/InferenceX/blob/main/inferencex-e2e/docs/agentx-standalone.md)

## 专项测试计划

| 实验 | 计划 |
| --- | --- |
| P04 推测解码 | 使用 SPEED-Bench 比较 DeepSeek 不同推测解码方法的接受长度、接受率和性能收益。 |
| P05 KVCache 压力 | 使用 NVIDIA AIPerf 的共享前缀模拟，改变前缀复用和工作集规模，观察多级 KVCache 的复用能力。 |

## 优化安排

### 第一阶段：聚合部署

1. 建立聚合部署 Baseline，留存 P01–P03 的逐档结果。
2. 调优聚合 vLLM，用同样的压测条件验证相对 Baseline 的收益。
3. 在同资源和同测试口径下比较 vLLM 与 SGLang。

### 第二阶段：PD 分离

1. 建立 PD 分离 Baseline。
2. 提升多级 KVCache 复用率和缓存传输能力。
3. 调优推测解码，记录接受长度、接受率和时延收益。
4. 用 Profiling 定位瓶颈，优化关键算子。

调优方向包括 TP、PP、DP、EP、SP 等并行方式、部署参数、网络通信、专家负载均衡（Expert LB）和 Prefill／Decode 资源配比。

## 输出结构

~~~text
P01/workload.json
P02/workload.json
P03/workload.json
logs/<UTC时间戳>/run.json
logs/<UTC时间戳>/<Pxx>/C<并发>/{plan.json,warmup.log,formal.log,*.aiperf/}
datasets/synthetic/<UTC时间戳>/<Pxx>/C<并发>/
results/<UTC时间戳>/summary.csv
results/<UTC时间戳>/report.html
results/<UTC时间戳>/<Pxx>/C<并发>/<阶段>/
~~~

HTML 可离线打开，含每项实验的并发、时延、TPM、KVCache 与推测解码图表。每档完成即更新 CSV 和 HTML。原生 AIPerf 结果留在各阶段目录，原生进程日志归入各阶段的日志目录。日志中同时保存生成命令及随机数据指纹。数据集、模型分词器、虚拟环境、日志和结果由 `.gitignore` 忽略。

## 客户端环境与离线数据

服务器项目环境使用 Python 3.12.16 和 AIPerf 0.14.0。重建环境时可用 Python 3.12 及哈希锁定文件。

~~~bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.txt
.venv/bin/python -m pip check
~~~

`tokenizer/` 内是目标 DeepSeek-V4.1-Flash-1017 的 `tokenizer.json`、相关配置、聊天模板与 `encoding/vllm_encoding.py`。固定长度依赖部署使用同一版本的分词器和 renderer。

P03 读取 `datasets/huggingface/` 的 Hugging Face 离线缓存。重新准备时，将固定版本的 [traces.jsonl](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126/resolve/23f152f6f0f9399a85901b89a6458def0ef16729/traces.jsonl) 下载到 `datasets/source/cc-traces-weka-062126/traces.jsonl`，然后在项目内执行：

~~~bash
.venv/bin/python P03/prepare_dataset.py
~~~

[AIPerf 0.14.0 导出格式](https://github.com/ai-dynamo/aiperf/blob/v0.14.0/docs/tutorials/working-with-profile-exports.md)
