"""
Evaluator - 检查 agent 的答案是否正确

使用 LLM 进行智能评估，替代简单的字符串匹配。
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 统一使用 ace_langgraph 前缀导入
from ace_langgraph.types import ReactQuestion, ReactAgentResult, EvaluatorResult


# 保留旧的 EvaluationResult 作为兼容性别名
EvaluationResult = EvaluatorResult


# LLM 评估提示词
EVALUATOR_PROMPT = """你是一个严谨的评估专家，负责判断 AI Agent 的答案是否正确。

**评估原则**：
1. 语义等价性：即使表述不同，但意思相同则视为正确
2. 容错性：允许合理的格式差异（如单位、大小写、空格）
3. 严谨性：对于数值、事实性问题要求精确匹配
4. 上下文理解：考虑问题的实际意图和答案的完整性

**输出格式**（必须返回有效的 JSON）：
{
  "is_correct": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "简短说明判断依据（1-2句话）",
  "feedback": "给 Agent 的反馈信息"
}

**示例**：

问题：Python 的创建者是谁？
期望答案：Guido van Rossum
实际答案：Python 是由 Guido van Rossum 创建的
评估：
{
  "is_correct": true,
  "confidence": 1.0,
  "reasoning": "答案包含正确的人名，语义等价",
  "feedback": "正确！答案准确识别了 Python 的创建者。"
}

问题：计算 (25 + 17) * 3
期望答案：126
实际答案：计算结果是 126
评估：
{
  "is_correct": true,
  "confidence": 1.0,
  "reasoning": "数值精确匹配，仅有表述差异",
  "feedback": "正确！数值计算准确。"
}

问题：东京是哪个国家的首都？
期望答案：日本
实际答案：中国
评估：
{
  "is_correct": false,
  "confidence": 1.0,
  "reasoning": "答案错误，东京是日本的首都",
  "feedback": "错误。东京是日本的首都，不是中国的。"
}
"""


class Evaluator:
    """
    使用 LLM 进行智能评估的评估器。
    
    相比简单的字符串匹配，LLM 评估器能够：
    - 理解语义等价性
    - 处理不同的表述方式
    - 提供详细的评估推理
    - 给出置信度评分
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0
    ):
        """
        初始化评估器。
        
        Args:
            model_name: LLM 模型名称
            temperature: LLM 温度参数（0 = 最确定性）
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature
        )
    
    def evaluate(
        self, 
        react_question: ReactQuestion,
        react_result: ReactAgentResult,
    ) -> EvaluatorResult:
        """
        评估 Agent 的答案是否正确。
        
        Args:
            react_question: 输入问题
            react_result: Agent 的执行结果
            ground_truth: 正确答案（如果有，优先使用此参数；否则从 react_question 获取）
            
        Returns:
            EvaluatorResult 对象
        """
        # 优先使用传入的 ground_truth，否则从 react_question 获取
        ground_truth = react_question.ground_truth
        
        if ground_truth is None:
            # 无正确答案，返回生产模式结果
            return EvaluatorResult(
                is_correct=False,
                feedback="生产模式：无正确答案可供验证，将基于自我反思学习。",
                confidence=0.0
            )
        
        return self._evaluate_with_llm(
            react_question.question,
            react_result.answer,
            ground_truth
        )
    
    def _evaluate_with_llm(
        self,
        question: str,
        answer: str,
        ground_truth: str
    ) -> EvaluatorResult:
        """使用 LLM 进行智能评估。"""
        user_message = f"""请评估以下答案：

问题：{question if question else "（未提供）"}
期望答案：{ground_truth}
实际答案：{answer}

请按照 JSON 格式返回评估结果。"""

        try:
            response = self.llm.invoke([
                SystemMessage(content=EVALUATOR_PROMPT),
                HumanMessage(content=user_message)
            ])
            
            # 解析 JSON 响应
            result_text = response.content.strip()
            
            # 提取 JSON（可能被包裹在 ```json ... ``` 中）
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            
            return EvaluatorResult(
                is_correct=result["is_correct"],
                feedback=result["feedback"],
                confidence=result.get("confidence", 1.0)
            )
            
        except Exception as e:
            # LLM 评估失败，当作生产模式（无正确答案）
            print(f"⚠️  LLM 评估失败，进入自我反思模式: {e}")
            return EvaluatorResult(
                is_correct=False,
                feedback="评估服务暂时不可用，将基于自我反思进行学习。",
                confidence=0.0
            )
