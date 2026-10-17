# RTV-Eval

![RTV-Eval 评测流程](docs/overview.svg)

[English](README.md) · 简体中文 · [开发指南](CONTRIBUTING.md)

**视频理解，究竟需要多少帧、多少分辨率？** RTV-Eval 用统一的评测流程，对比视觉语言模型在不同抽帧与分辨率策略下的表现，帮助研究视频输入量与识别准确率之间的取舍。

- **一份配置，多组对比**：自动运行模型 × 策略组合。
- **三类抽帧策略**：固定帧数、目标 FPS、基于画面变化的自适应策略。
- **远程与本地推理**：支持 OpenAI 兼容视觉 API，以及自动管理生命周期的本地 Qwen3-VL 服务。
- **结果可追溯**：SQLite 保存原始回复、答案、准确率、请求耗时和帧数，可导出 CSV / JSON。
- **失败可续跑**：保留已记录答案，重跑时补齐缺失项。

这是面向流式场景预处理策略的**离线评测工具**。目前不提供实时摄像头接入、网络传输测量或端到端实时延迟评测。

## 快速开始

需要 Python 3.10+、MotionBench 数据，以及支持多图输入的模型 API。

```bash
git clone https://github.com/jielosc/rtv-eval.git
cd rtv-eval
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp configs/example.yaml configs/local.yaml
```

编辑 `configs/local.yaml`，填写模型接口、数据路径和策略。数据结构：

```text
MotionBench/
├── data/video_info.meta.jsonl
└── videos/
    ├── self-collected/*.mp4
    └── public-dataset/*.mp4
```

`benchmark.data_dir` 指向 `data/`，`benchmark.video_root` 指向 `videos/`。数据集需另行获取。

```bash
export OPENAI_API_KEY="你的 API Key"
rtv-eval run configs/local.yaml --dry-run
rtv-eval run configs/local.yaml
rtv-eval status results/motionbench_baseline.db
rtv-eval report results/motionbench_baseline.db
rtv-eval export results/motionbench_baseline.db --view summary
rtv-eval export results/motionbench_baseline.db --view results
```

数据库文件名由配置的 `name` 决定；示例使用 `motionbench_baseline`。`--dry-run` 校验配置并展示组合，不验证数据文件、服务可达性或实际模型能力。

## 策略怎么选

| 策略 | 控制变量 | 适用场景 |
| --- | --- | --- |
| `count` | 整段视频均匀抽取 N 帧 | 固定输入预算的模型比较 |
| `fps` | 按目标帧率抽帧，上限为源视频帧率 | 不同时间采样密度比较 |
| `adaptive` | 每个窗口根据变化分数选择 FPS、分辨率和 JPEG 质量 | 研究内容自适应上传策略 |

预设分辨率限制短边；`exact` 将图像等比例缩放至指定宽高边界内，不放大、不强制拉伸。自适应策略使用两遍解码，并根据当前整个窗口确定策略；在线实现需要窗口缓存，不能视作无前瞻的逐帧决策。完整 YAML 示例见 [英文说明](README.md#strategies)。

## 结果与续跑

- `dev` 有标准答案，按严格单选字母规则评分；解释性文字或无效选项计为错误。
- `test` 仅保存预测，无标准答案，准确率不应作为评估依据。
- 缺视频、无有效帧或请求失败会使该组合标记为 `failed`，CLI 返回非零退出码。修复原因后用相同配置重跑，自动补齐未记录项。
- API 延迟不包含视频解码和预处理，也不是端到端流式延迟。
- 更改模型、策略或评测协议时，请使用新的实验名，避免混合结果。本次抽帧修复使用协议 `mcq-letter-v3`，旧数据库仍可查看和导出。

## 本地模型

```bash
python -m pip install -e '.[local]'
cp configs/local_model.example.yaml configs/local.yaml
```

配置数据路径、模型名、缓存目录与设备。需自行准备兼容的 PyTorch/CUDA 环境和支持 Qwen3-VL 的 Transformers；首次运行可能下载模型权重。运行器自动启动临时本地服务，同一模型的多组策略复用该服务，结束后释放进程。内置服务目前仅支持 Qwen3-VL。

## 开发

参见 [开发指南](CONTRIBUTING.md) 与 [修复记录](docs/maintenance.md)。本仓库不以未经实测的准确率、性能提升或模型排行榜作为宣传依据。
