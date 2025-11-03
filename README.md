# ACE LangGraph - ReAct Agent with Adaptive Learning

ACE (Agentic Context Engine) + LangGraph 集成，为 ReAct Agent 提供自适应学习能力。

## 🌟 特性

- **自适应学习**：通过 ACE 框架让 ReAct Agent 从经验中学习策略
- **向量检索**：使用 ChromaDB 实现策略的语义检索
- **工具集成**：支持计算器、网络搜索等工具
- **可观测性**：可选的 LangSmith 追踪支持

## 📦 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入你的 API keys
```

## 🚀 快速开始

### 基本使用

```python
from ace_langgraph.ace_react_agent import ACEReActWorkflow, ReactQuestion

# 创建工作流
workflow = ACEReActWorkflow(
    model_name="gpt-4o-mini",
    use_vector_retrieval=True
)

# 训练模式（提供 ground_truth）
question = ReactQuestion(
    question="计算 (25 + 17) * 3 - 8",
    ground_truth="118"
)
result = workflow.run(question)

# 生产模式（不提供 ground_truth，只获取答案）
question = ReactQuestion(question="Python 是谁创建的？")
answer = workflow.ask(question)
```

### 运行示例

```bash
# 训练 ACE ReAct Agent
python ace_react_agent.py

# 或使用训练脚本
python train_ace_react.py
```

## 📚 核心组件

- `ace_react_agent.py` - ACE + ReAct 集成工作流
- `react_agent.py` - ReAct Agent 实现
- `evaluator.py` - 答案评估器
- `reflector.py` - 推理过程反思器
- `curator.py` - 策略提取与管理
- `playbook.py` - 策略存储（支持向量检索）
- `types.py` - 类型定义

## 🔧 配置

### 环境变量

在 `.env` 文件中配置：

```bash
# 必需
OPENAI_API_KEY=your-key-here

# 可选 - LangSmith 追踪
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-key-here
LANGSMITH_PROJECT=your-project-name
```

### 启用 LangSmith 追踪

```bash
source enable_langsmith.sh
```

## 🎯 工作原理

1. **Generator (ReAct Agent)**: 执行任务，使用学到的策略
2. **Evaluator**: 评估答案正确性
3. **Reflector**: 分析推理过程，识别成功/失败模式
4. **Curator**: 提取可复用策略，更新 Playbook

## 📄 许可

MIT License - 查看主项目的 LICENSE 文件
