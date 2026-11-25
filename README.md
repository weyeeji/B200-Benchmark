# DeepSeek Serving 并发压测

Python 脚本为 NVIDIA AIPerf v0.14.0 准备固定数据、编排实验，并用原始记录计算稳态指标。请求经 OpenAI Chat Completions API 发送，部署参数由服务端维护。官方稳定版、PyPI 和标签源码均已核对。 [NVIDIA 文档](https://docs.nvidia.com/aiperf/v0.14.0/welcome-to-ai-perf-documentation) [官方 CLI](https://docs.nvidia.com/aiperf/v0.14.0/reference/command-line-options) [v0.14.0 源码](https://github.com/ai-dynamo/aiperf/tree/v0.14.0) [PyPI](https://pypi.org/project/aiperf/0.14.0/)

vLLM 的随机负载和缓存隔离说明、SGLang 的在线客户端延迟定义，以及 Dynamo 的固定请求形状对比方法也作为实现参考。 [vLLM Benchmark CLI](https://docs.vllm.ai/en/stable/benchmarking/cli/) [SGLang Benchmark and Profiling](https://docs.sglang.io/docs/developer_guide/benchmark_and_profiling) [Dynamo with AIPerf](https://docs.nvidia.com/dynamo/dev/cli/operations/benchmarking-with-ai-perf)

## 环境与配置

新环境使用 Linux x86-64 和 Python 3.12。在项目目录执行以下命令。

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.txt
.venv/bin/python -m pip check
source .venv/bin/activate
aiperf --version
python benchmark.py --help
cp config.example.json config.local.json
```

已有离线 wheelhouse 时，在安装命令中加入 `--no-index --find-links .offline/wheels`。将被测模型的 tokenizer 快照放入 `tokenizer/` 或指定本地路径。DeepSeek 模式从快照内的 `encoding/vllm_encoding.py` 或 `encoding/encoding.py` 加载匹配服务端的聊天编码器。

在 `config.local.json` 中填写服务地址、模型名、目标部署的实际 GPU 数量及部署说明。服务地址接受 HTTP(S) origin，或带 `/v1`、`/v1/chat/completions` 的形式。认证使用 `api_key_env` 指定的环境变量，默认 `OPENAI_API_KEY`。相对路径以配置文件所在目录为基准。示例配置使用 DeepSeek-V4.1-Flash-1017 tokenizer 和 DeepSeek 聊天编码器。默认发出 `chat_template_kwargs.thinking=true`，换部署时应确认对应模板及参数的语义。

在 `deployment.name` 写入部署标识，在 `deployment.details` 记录 PD 拓扑、TP/DP/EP、缓存层、调度和算子优化等信息。使用 `deployment.speculative.mode` 的 `off`、`real`、`simulated` 或 `unknown`，并记录算法、草稿长度和模拟接受长度。配置仅用来标注实验。

## 数据与流程

| 场景 | ISL | OSL | 理论共享前缀 | 剩余输入 | 工作集 |
| --- | ---: | ---: | ---: | ---: | --- |
| P01 | 4,096 | 1,024 | 0 | 4,096 | 独立随机输入 |
| P02 | 65,536 | 600 | 0 | 65,536 | 独立随机输入 |
| P03 | 65,536 | 600 | 58,880 | 6,656 | `floor(C / 117760)` 组 |
| P04 | 65,536 | 600 | 58,880 | 6,656 | `ceil(2 × C / 58880)` 组 |

第一次生成 P03/P04 时，设置初始基准部署实测的有效 GPU Prefix Cache 容量 `cache_capacity_tokens`，或明确给出 `prefix_groups.P03` 和 `prefix_groups.P04`。命令行对应 `--cache-capacity`、`--p03-groups` 和 `--p04-groups`。小工作集公式若得到零组，需明确选择可用的组数。后续部署复用首次冻结的组数、内容和访问顺序。

```bash
# P01/P02 的四个并发档位
python benchmark.py prepare --scenes P01 P02

# 将 C 的环境变量设置为初始基准部署实测的逻辑 Token 数
python benchmark.py prepare --scenes P03 P04 --cache-capacity "$C"

# 校验已有数据、冻结配置和 SHA-256
python benchmark.py plan --config config.local.json

# 低开销的流式 EOS 对照，随后把输出路径填入 eos_probe_file
python benchmark.py probe --config config.local.json

# 用完整请求配额跑预实验，得到所有选中测试点建议使用的统一发包时限
python benchmark.py pilot --config config.local.json

# 正式矩阵。若预实验需要更长时限，设置 duration_s 或传 --duration
python benchmark.py run --config config.local.json

# 单点示例
python benchmark.py run --config config.local.json --scenes P01 --concurrencies 32 --repeats 2
```

随机样本来自实际 tokenizer 的可往返 Token 子集。生成器将 Token 解码成文本，以聊天模板重新编码并逐条核对长度与边界。HTTP 请求携带用户文本，服务端执行聊天渲染和 tokenization。P01/P02 各请求在模板固定开头之后即分叉。P03/P04 的共享 58,880 Token 包含模板开头，余下 6,656 Token 包含生成提示结尾。前缀组每个周期等频访问一次，周期内用固定种子打乱。每点保存 `manifest.json`、预热和正式 JSONL、跨阶段的 `order.jsonl`。运行和预实验会校验冻结文件，数据和 tokenizer 可随档案复制到其它客户端。

每点先执行缓存 Hook，再用独立 AIPerf 进程完成预热。访问数为 `max(16, 前缀组数)`，前缀场景首先完整访问每组一次。预热并发上限 16。正式点用 native sequential 单轮会话、固定并发闭环、Streaming、服务器 Usage 及 `ignore_eos=true`，无 RPS 限制。请求配额是 `max(3 × 并发, 2 × 前缀组数)`。默认 300 秒是发包上限，请求配额先完成时提前停止发包。Pilot 保留同一配额和序列并放开发包时限，统一建议值以最慢点的最后发包时间加 25% 余量计算。正式运行因时限无法覆盖配额时，摘要标为 incomplete。以原始数据集顺序分配给 AIPerf 的序号会被核对，网络实际到达次序受并发影响。 [AIPerf 闭环调度与停止条件](https://docs.nvidia.com/aiperf/v0.14.0/benchmark-modes/load-generator-options-reference) [逐请求导出字段](https://docs.nvidia.com/aiperf/v0.14.0/tutorials/metrics-analysis/working-with-profile-export-files)

## 缓存清理

把 `cache_hook.command` 写为由操作方提供的命令参数列表。Hook 的环境变量包含 `BENCHMARK_API_URL`、`BENCHMARK_MODEL`、`BENCHMARK_PROFILE`、`BENCHMARK_CONCURRENCY`、`BENCHMARK_REPEAT` 和 `BENCHMARK_ARTIFACT_DIR`。Hook 成功后应在 stdout 只写一个 JSON 对象，按实际状态报告每层，例如下例。

```json
{"gpu":"cleared","cpu":"cleared","ssd":"not_applicable","detail":"Backend confirmed the configured cache tiers."}
```

缺少 Hook 或层级确认时将该层标为 `unverified`。设置 `cache_hook.require_confirmed=true` 可强制校验所有层。理论共享前缀比例、Usage 里实际缓存/未缓存的 Token、服务端按层累计的缓存访问数据各有独立字段。P04 的首次访问覆盖所有组，驻留情况由服务端缓存策略决定。

## 指标与产物

每个实验写入独立的 `results/run-…/` 或 `results/pilot-…/`，包含 `summary.csv`、`summary.json`、环境版本和代码哈希。每个测试点的 `warmup/` 和 `formal/` 保存独立的 AIPerf 原始响应、记录、汇总、配置与日志。另有数据 manifest、Hook 回执和运行状态。逐请求错误可在 AIPerf 记录的 error 字段检索。

稳态窗口从观测到并发首次达到目标开始，到最后一个正式请求发出为止。稳态吞吐按这个窗口结束的请求和 Usage Token 总量计算。所有正式请求仍有有限等待时间，以保存 drain、失败和输出长度证据。AIPerf 原生备份保留其整个正式阶段的统计。摘要 TTFT、ITL、TPOT、E2EL 分位使用 AIPerf 的逐请求指标和时间戳。TPOT 使用整个响应结束前的解码耗时，ITL 采用内容间隔的逐 Token 平均估计。若服务支持 continuous usage，可设置 `per_chunk_usage=true`，使 AIPerf 使用首个内容块的累计 Token 数据，便于推测解码时更准确地估计 ITL。输出使用服务端 completion_tokens，包含 Thinking Token。缺少 Usage 时用拼接的 reasoning 与 content 作 tokenizer 估计，摘要标明来源。未观测到的指标写 `N/A`。 [AIPerf 指标定义](https://docs.nvidia.com/aiperf/v0.14.0/reference/ai-perf-metrics-reference)

在 `sla` 填写任意组合的 `ttft_ms`、`tpot_ms`、`itl_ms`、`e2el_ms` 上限。Goodput 统计稳态中满足全部阈值的成功请求每秒数，达标率分母包括稳态完成及失败的请求。

在 `prometheus.urls` 分别填写 worker/exposer 的完整 `/metrics` 地址。AIPerf 保存原始抓取结果，摘要使用稳态窗口内的时间序列，按标签与端点聚合 worker，计数器速率和比率至少需要两次稳态采样。对于已有指标名的 vLLM、SGLang、DCGM 和 node_exporter 可用默认规则。PD 传输、缓存逐级命中率/淘汰率/加载延迟及推测接受指标依赖被测端导出。摘要只聚合配置中显式列出的地址。AIPerf 可能自动抓取服务 origin 的默认指标地址，原始文件也会保留这些样本。可按配置如下覆盖规则，每批 worker 选择一个指标视图。

```json
{
  "prometheus": {
    "urls": ["http://worker1.example:8000/metrics", "http://worker2.example:8000/metrics", "http://telemetry.example:9400/metrics"],
    "rules": {
      "kv_transfer_latency_ms": {
        "kind": "histogram_mean",
        "metric": "your_kv_transfer_latency_seconds",
        "scale": 1000
      },
      "gpu_cache_hit_rate": {
        "kind": "ratio",
        "metric": "your_gpu_cache_hits",
        "denominator": "your_gpu_cache_lookups",
        "labels": {"model_name": "your-model"}
      }
    }
  }
}
```

抓取主机与基线应保持一致，且按部署范围设置 GPU 设备标签和 exporter 范围。配置示例给出的地址为占位符。监控功能参见 [NVIDIA Server Metrics](https://docs.nvidia.com/aiperf/v0.14.0/server-metrics/server-metrics-collection)。

## GitHub Deploy Key

公钥在 `~/.ssh/cq_b200_benchmark_github_ed25519.pub`。将其添加到目标仓库的 **Settings → Deploy keys**，用默认只读权限即可拉取。远端具备 GitHub 网络访问后，可对该仓库单独指定密钥。

```bash
git -c core.sshCommand="ssh -o IdentitiesOnly=yes -i $HOME/.ssh/cq_b200_benchmark_github_ed25519" \
    clone git@github.com:OWNER/REPO.git
git -C REPO config --local core.sshCommand \
    "ssh -o IdentitiesOnly=yes -i $HOME/.ssh/cq_b200_benchmark_github_ed25519"
```
