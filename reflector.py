"""
Reflector - 分析 Agent 的推理过程

反思器负责分析 Agent 的执行过程，找出成功模式或失败原因。
基于 ACE prompts_v2.py 的系统化分析协议。
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import json
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 统一使用 ace_langgraph 前缀导入
from ace_langgraph.prompts import REFLECTOR_PROMPT_V2
from ace_langgraph.types import ReactQuestion, ReactAgentResult, EvaluatorResult, ReflectionResult, StrategyTag
from ace_langgraph.playbook import Playbook


class Reflector:
    """
    反思器 - 高级诊断分析系统
    """
    
    def __init__(
        self,
        playbook: Optional[Playbook] = None,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
    ):
        """
        初始化反思器。
        
        参数：
            playbook: Playbook 实例（用于获取策略内容）
            model_name: LLM 模型名称
            temperature: 温度参数（较高的温度产生更多样化的分析）
        """
        self.playbook = playbook
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.model_name = model_name
    
    def reflect(
        self,
        react_question: ReactQuestion,
        react_result: ReactAgentResult,
        evaluator_result: Optional[EvaluatorResult] = None
    ) -> ReflectionResult:
        """
        反思 Agent 的执行过程。
        
        参数：
            react_question: 输入问题
            react_result: Agent 的执行结果
            evaluator_result: 评估结果（可选，生产模式下为 None）
            
        返回：
            ReflectionResult 包含完整的诊断分析
        """
        # 提取信息
        question = react_question.question
        reasoning = react_result.reasoning
        final_answer = react_result.answer
        used_strategy_ids = react_result.used_strategies
        ground_truth = react_question.ground_truth  # 从问题中获取
        
        # 提取评估信息
        if evaluator_result:
            feedback = evaluator_result.feedback
        else:
            feedback = None
        
        # 构建策略摘录（包含策略的完整内容）
        playbook_excerpt = "(无策略被引用)"
        if used_strategy_ids and self.playbook:
            strategy_lines = []
            for sid in used_strategy_ids:
                strategy = self.playbook.get_strategy(sid)
                if strategy:
                    strategy_lines.append(f"[{sid}] {strategy.content}")
                else:
                    strategy_lines.append(f"[{sid}] (策略未找到)")
            playbook_excerpt = "\n".join(strategy_lines)
        elif used_strategy_ids:
            # 如果没有 playbook，至少显示 ID
            playbook_excerpt = "\n".join(
                f"[{sid}] (策略内容不可用)" for sid in used_strategy_ids
            )
        
        # 填充提示词
        prompt = REFLECTOR_PROMPT_V2.format(
            current_date=datetime.now().strftime("%Y-%m-%d"),
            question=question,
            reasoning=reasoning,
            prediction=final_answer,
            ground_truth=ground_truth or "(未提供)",
            feedback=feedback or "(无反馈)",
            playbook_excerpt=playbook_excerpt
        )
        
        # 调用 LLM
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        # 解析 JSON 响应
        try:
            # 提取 JSON（可能被包裹在```json```中）
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            data = json.loads(content)
            
            # 构建策略标记
            strategies_to_mark = []
            for strategy_tag in data.get("strategy_tags", []):
                tag = strategy_tag.get("tag", "neutral").lower()
                strategy_id = strategy_tag.get("id", "")
                
                if tag == "helpful":
                    strategies_to_mark.append({
                        "strategy_id": strategy_id,
                        "helpful": True,
                        "justification": strategy_tag.get("justification", "")
                    })
                elif tag == "harmful":
                    strategies_to_mark.append({
                        "strategy_id": strategy_id,
                        "harmful": True,
                        "justification": strategy_tag.get("justification", "")
                    })
                else:  # neutral
                    strategies_to_mark.append({
                        "strategy_id": strategy_id,
                        "neutral": True,
                        "justification": strategy_tag.get("justification", "")
                    })
            
            # 将策略标记转换为 StrategyTag 对象
            strategy_tag_objects = [
                StrategyTag(
                    id=tag["strategy_id"],
                    tag="helpful" if tag.get("helpful") else ("harmful" if tag.get("harmful") else "neutral"),
                    justification=tag.get("justification", "")
                )
                for tag in strategies_to_mark
            ]
            
            return ReflectionResult(
                reasoning=data.get("reasoning", ""),
                error_identification=data.get("error_identification", "none"),
                error_location=data.get("error_location", "N/A"),
                root_cause_analysis=data.get("root_cause_analysis", ""),
                correct_approach=data.get("correct_approach", ""),
                key_insight=data.get("key_insight", ""),
                confidence_in_analysis=float(data.get("confidence_in_analysis", 0.8)),
                strategy_tags=strategy_tag_objects
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # 反思失败，返回空结果
            print(f"⚠️  反思解析失败，返回空结果: {e}")
            return ReflectionResult(
                reasoning="反思服务暂时不可用，跳过本次反思。",
                error_identification="none",
                error_location="N/A",
                root_cause_analysis="",
                correct_approach="",
                key_insight="",
                confidence_in_analysis=0.0,
                strategy_tags=[]
            )
