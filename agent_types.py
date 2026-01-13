"""
ACE LangGraph 类型定义
集中管理所有模块间传递的数据类型
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class ReactQuestion:
    """ReAct Agent 的输入问题"""
    question: str
    context: str = ""
    ground_truth: Optional[str] = None  # 正确答案（训练模式时提供）
    
    def as_str(self) -> str:
        """格式化为字符串"""
        parts = [f"问题：{self.question}"]
        if self.context:
            parts.append(f"上下文：{self.context}")
        if self.ground_truth:
            parts.append(f"正确答案：{self.ground_truth}")
        return "\n".join(parts)


@dataclass
class ReactAgentResult:
    """ReAct Agent 的执行结果"""
    answer: str
    reasoning: str  # Agent 的完整推理过程
    used_strategies: List[str]  # 使用的策略 ID 列表
    iterations: int  # 迭代次数
    messages: List[Any]  # LangChain 消息列表
    success: bool = True
    
    def as_str(self) -> str:
        """格式化为字符串"""
        strategies_str = ", ".join(self.used_strategies) if self.used_strategies else "无"
        return f"""答案：{self.answer}
迭代次数：{self.iterations}
使用策略：{strategies_str}
成功：{'是' if self.success else '否'}

推理过程：
{self.reasoning[:500]}{"..." if len(self.reasoning) > 500 else ""}"""


@dataclass
class EvaluatorResult:
    """Evaluator 的评估结果"""
    is_correct: bool
    feedback: str
    confidence: float = 1.0
    
    def as_str(self) -> str:
        """格式化为字符串"""
        status = "✓ 正确" if self.is_correct else "✗ 错误"
        return f"""{status}
反馈：{self.feedback}
置信度：{self.confidence:.0%}"""


@dataclass
class StrategyTag:
    """策略标记"""
    id: str
    tag: str  # "helpful" | "harmful" | "neutral"
    justification: str


@dataclass
class ReflectionResult:
    """Reflector 的反思结果"""
    reasoning: str
    error_identification: str
    error_location: str
    root_cause_analysis: str
    correct_approach: str
    key_insight: str
    confidence_in_analysis: float
    strategy_tags: List[StrategyTag]
    
    def as_str(self) -> str:
        """格式化为字符串，供 Curator 提示词使用"""
        tags_str = "\n".join([
            f"  - [{tag.id}] {tag.tag}: {tag.justification}"
            for tag in self.strategy_tags
        ])
        
        return f"""
## 系统化分析
{self.reasoning}

## 诊断详情
- **错误识别**: {self.error_identification}
- **错误位置**: {self.error_location}
- **根因分析**: {self.root_cause_analysis}

## 正确方法
{self.correct_approach}

## 关键见解
{self.key_insight}

## 置信度
{self.confidence_in_analysis}

## 策略标记
{tags_str if tags_str else "（无策略标记）"}
""".strip()

@dataclass
class DeltaOperation:
    """Playbook 更新操作"""
    type: str  # "ADD" | "UPDATE" | "REMOVE"
    content: Optional[str] = None
    category: Optional[str] = None
    strategy_id: Optional[str] = None
    justification: str = ""
    applied: bool = False
    skip_reason: Optional[str] = None

@dataclass
class CuratorResult:
    """Curator 的策展结果"""
    reasoning: str
    operations: List[DeltaOperation]
    added_count: int = 0
    updated_count: int = 0
    removed_count: int = 0
    marked_count: int = 0  # 标记的策略数量

    def as_str(self) -> str:
        """格式化为字符串"""
        ops_summary = []
        for op in self.operations:
            status = "已执行" if op.applied else f"已跳过：{op.skip_reason or '原因未知'}"
            if op.type == "ADD":
                ops_summary.append(f"  + 新增：{op.content[:80]}...（{status}）")
            elif op.type == "UPDATE":
                ops_summary.append(f"  ↻ 更新：[{op.strategy_id}] {op.content[:60]}...（{status}）")
            elif op.type == "REMOVE":
                ops_summary.append(f"  - 移除：[{op.strategy_id}]（{status}）")

        ops_str = "\n".join(ops_summary) if ops_summary else "  （无操作）"

        return f"""推理：{self.reasoning[:200]}{"..." if len(self.reasoning) > 200 else ""}

统计：
  新增策略：{self.added_count} 个
  更新策略：{self.updated_count} 个
  移除策略：{self.removed_count} 个
  标记策略：{self.marked_count} 个

操作详情：
{ops_str}"""