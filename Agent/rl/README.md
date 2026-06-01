# Autobahn PPO Contextual Policy System

基于PPO（Proximal Policy Optimization）算法的Autobahn系统自动调参框架，实现**上下文策略 (Contextual Policy)** 架构。

如需将 PPO 替换为 CMAB（Contextual Multi-Armed Bandit），请参考 `CMAB_ARCHITECTURE.md`。

## 🎯 核心创新：Contextual Policy

系统采用上下文条件策略设计：

```
π(action | dynamic_state ; system_config)
```

**直观理解**：
- `system_config` (Context): 系统配置（硬件+网络），相对静态，作为条件信息
- `dynamic_state`: 动态状态（工作负载+lane向量），随时间变化，作为观测
- `action`: 共识参数调整，作为策略输出

**优势**：
- 跨系统泛化：策略能在不同硬件/网络配置下工作
- 样本效率：context只需设置一次，无需重复学习
- 模块化：清晰分离静态配置和动态状态

## 目录结构

```
rl/
├── actions/           # 动作和状态编码
│   ├── action_encode.py   # 动作编码/解码（PPO动作 <-> Autobahn参数）
│   └── state_encode.py    # 状态编码（metrics JSON -> 状态向量）
├── controllers/      # 系统控制器
│   └── controller.py      # AutobahnController：应用参数到系统
├── envs/             # Gymnasium环境
│   └── env.py             # AutobahnEnv：RL环境实现
└── train_ppo.py      # PPO训练脚本
```

## 核心组件

### 1. ActionCodec (`actions/action_encode.py`)

将PPO的MultiDiscrete动作空间映射到Autobahn系统参数。

**动作空间**（10维）：
- `batch_size`: [256, 512, 1024, 2048, 4096]
- `max_batch_delay`: [1, 2, 5, 10, 20] ms
- `header_size`: [2, 4, 8, 16]
- `max_header_delay`: [5, 10, 20, 40] ms
- `use_optimistic_tips`: [True, False]
- `sync_retry_delay`: [5, 10, 20, 40] ms
- `sync_retry_nodes`: [1, 2, 3]
- `cut_condition_type`: [1, 2, 3, 4, 5]
- `fast_path_timeout`: [0, 5, 10, 20] ms
- `k` (parallel_proposals): [1, 2, 4, 8]

**方法**：
- `decode(action)`: 将PPO动作解码为Autobahn参数字典
- `encode(params)`: 将Autobahn参数编码为PPO动作（可选）

### 2. State Encoder (`actions/state_encode.py`)

从metrics JSON文件中解析状态和奖励，支持上下文分离架构。

**状态分离设计**：

**Context (系统配置 - 静态)**：
- Hardware (4维): CPU核心数、内存、网络带宽、工作节点数
- Network (可变): 到其他节点的延迟向量

**Dynamic State (动态状态 - 时变)**：
- Workload (2维): 交易大小、交易到达率
- Lane Vector (可变): 每个lane的高度增长率
- Fast Path Ratio (1维): 快速路径比例

**奖励函数**：
```
reward = 0.01 * end_to_end_tps - 0.08 * end_to_end_latency_ms
```

**函数**：
- `parse_metrics_with_context(json_path)`: 分离解析context和dynamic state
- `parse_metrics(json_path)`: 向后兼容，合并context+dynamic state
- `get_state_dim(metrics_dir, include_context=True/False)`: 检测不同状态维度

### 3. AutobahnController (`controllers/controller.py`)

负责将RL动作应用到Autobahn系统。

**功能**：
- `apply_action(params)`: 更新`.parameters.json`文件
- `wait_one_epoch()`: 等待一个epoch完成
- `latest_metrics_file()`: 获取最新的metrics文件
- `get_current_params()`: 获取当前参数

### 4. AutobahnEnv (`envs/env.py`)

Gymnasium环境实现，采用contextual policy架构。

**Contextual Design**：
- **Observation Space**: 仅包含dynamic state（工作负载+lane向量+fast path）
- **Context**: 系统配置（硬件+网络），episode开始时设置一次
- **Training State**: Context + Dynamic State 组合用于PPO训练

**Episode结构**：
- 每个episode = 1个完整的epoch（20个slots）
- Context在episode开始时设置，贯穿整个episode
- 每个step返回dynamic state作为observation
- 训练时使用full state (context + dynamic)进行学习

**特点**：
- 自动处理状态维度不匹配和填充
- Context持久化，减少重复学习开销
- 完善的错误处理和跨episode状态管理

## 使用方法

### 1. 准备数据

确保以下目录和文件存在：
- Metrics目录：`/home/ccclr0302/autobahn-test/metrics`（包含metrics JSON文件）
- 参数文件：`/home/ccclr0302/.parameters.json`

### 2. 训练模型

基本训练：
```bash
cd /home/ccclr0302/autobahn-test/Agent/rl
python train_ppo.py
```

自定义参数：
```bash
python train_ppo.py \
    --metrics-dir /home/ccclr0302/autobahn-test/metrics \
    --parameters-file /home/ccclr0302/.parameters.json \
    --slots-per-epoch 20 \
    --num-iterations 1000 \
    --checkpoint-dir /tmp/ppo_checkpoints \
    --checkpoint-freq 100
```

从checkpoint恢复：
```bash
python train_ppo.py --resume-from /tmp/ppo_checkpoints/checkpoint_500
```

### 3. 训练参数说明

- `--metrics-dir`: metrics文件所在目录
- `--parameters-file`: Autobahn参数文件路径
- `--slots-per-epoch`: 每个epoch的slot数量（默认20）
- `--num-iterations`: 训练迭代次数（默认1000）
- `--checkpoint-dir`: checkpoint保存目录
- `--checkpoint-freq`: checkpoint保存频率
- `--resume-from`: 从指定checkpoint恢复训练

## PPO配置

当前PPO配置：
- 学习率: 3e-4
- 训练批次大小: 200
- SGD小批次大小: 64
- SGD迭代次数: 10
- 使用GAE: True
- Lambda: 0.95
- Gamma: 0.99
- Clip参数: 0.2
- 价值函数损失系数: 0.5
- 熵系数: 0.01

可以根据需要调整这些参数。

## Contextual Policy 架构详解

### 为什么需要Contextual Policy？

**传统方法的问题**：
- 硬件和网络配置被当作普通状态，每step重复输入
- 学习效率低：需要重新学习相同的系统特性
- 泛化能力差：难以适应不同的硬件/网络配置

**Contextual Policy的优势**：
- **条件化学习**: `π(action | dynamic_state ; system_config)`
- **一次设置**: Context在episode开始时设置，无需重复
- **跨系统泛化**: 策略能在不同系统配置下工作
- **模块化**: 清晰分离静态配置和动态行为

### 实现细节

#### Context设置流程
```
Episode Start:
├── reset() → 读取metrics → 解析context → 设置episode_context
├── step() → 使用episode_context + dynamic_state → 训练

Episode End:
└── context保持到下个episode（如果系统配置不变）
```

#### 状态表示对比

| 组件 | 传统方法 | Contextual方法 |
|------|---------|---------------|
| Observation | Context + Dynamic | Dynamic Only |
| Training State | Context + Dynamic | Context + Dynamic |
| Context更新 | 每step | Episode开始时 |

#### 维度处理
- **Observation Space**: dynamic state维度（约7维）
- **Context**: 系统配置维度（约7维）
- **Training**: full state维度（约14维）

## 工作流程

### 传统RL vs Contextual RL

**传统RL流程**：
1. 读取metrics → 解析完整状态 → 输入到策略网络
2. 策略网络学习所有特征（硬件+网络+工作负载+lane）

**Contextual RL流程**：
1. **Episode开始**: 读取metrics → 解析context → 设置为条件信息
2. **每step**: 读取metrics → 解析dynamic state → 作为观测输入
3. **训练**: 使用(context + dynamic_state)进行PPO更新
4. **Episode结束**: Context保持，可用于下个episode

### 实际执行流程

1. **初始化**：环境读取metrics，分离context和dynamic state
2. **Context设置**：Episode开始时设置系统配置context
3. **动作选择**：PPO策略基于dynamic state和context选择参数
4. **参数应用**：Controller更新`.parameters.json`文件
5. **等待执行**：系统运行一个epoch（20 slots）
6. **状态更新**：读取新metrics，更新dynamic state
7. **奖励计算**：基于TPS和延迟计算奖励
8. **策略更新**：PPO使用full state (context+dynamic)进行学习

## 注意事项

1. **状态维度**：状态维度可能因网络节点数和lane数量而变化。框架会自动处理维度不匹配。

2. **Metrics文件**：确保metrics目录中有有效的JSON文件，格式符合`state_encode.py`的预期。

3. **参数文件**：`.parameters.json`文件会被自动更新，原始文件会备份为`.json.backup`。

4. **单步Episode**：每个step都是一个完整的episode（terminated=True），这是为了适应Autobahn的epoch结构。

5. **错误处理**：框架包含完善的错误处理，训练过程中出现错误会记录日志并继续训练。

## 依赖

- Python 3.7+
- Ray RLlib
- Gymnasium
- NumPy
- PyTorch

安装依赖：
```bash
pip install ray[rllib] gymnasium numpy torch
```

## 扩展

### 自定义奖励函数

修改`actions/state_encode.py`中的`parse_metrics`函数的奖励计算部分。

### 添加新的动作维度

1. 在`ActionCodec.__init__`中添加新的维度到`action_dims`
2. 在`ActionCodec.decode`中添加对应的参数映射
3. 更新`ActionCodec.encode`（如果使用）

### 修改状态表示

修改`actions/state_encode.py`中的`parse_metrics`函数来改变状态表示。

## 故障排除

1. **找不到metrics文件**：检查metrics目录路径是否正确，确保目录中有JSON文件。

2. **状态维度不匹配**：框架会自动处理，但建议确保metrics文件格式一致。

3. **Ray初始化失败**：检查Ray是否正确安装，尝试重启Ray：`ray stop && ray start --head`

4. **参数应用失败**：检查`.parameters.json`文件权限，确保有写权限。

## 许可证

[根据项目许可证]
