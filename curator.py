"""
Curator - 从反思中提取和管理策略（增强版 v2.0）

策展器负责从反思中提取可复用的策略，并管理 Playbook。
基于 ACE prompts_v2.py 的优先级决策树和质量控制协议。
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
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
from ace_langgraph.playbook import Playbook
from ace_langgraph.prompts import CURATOR_PROMPT_V2
from ace_langgraph.types import (
    ReactQuestion, ReactAgentResult, EvaluatorResult, 
    ReflectionResult, CuratorResult, DeltaOperation
)


# 保留旧的 CurationResult 作为兼容性别名
CurationResult = CuratorResult


class Curator:
    """
    策展器 v2.0 - 高级 Playbook 管理系统
    
    基于 ACE prompts_v2.py 的优先级决策树，提供：
    1. 5级优先级决策（关键错误/缺失能力/策略改进/矛盾解决/成功强化）
    2. 策略去重协议（70%相似度检查）
    3. 严格的质量控制标准
    4. Playbook大小管理
    
    示例：
        >>> curator = Curator(model_name="gpt-4o-mini", use_v2_prompt=True)
        >>> result = curator.curate(
        ...     reflection="Agent 成功使用了工具...",
        ...     is_correct=True,
        ...     playbook=playbook,
        ...     strategies_to_mark=[...]
        ... )
        >>> print(f"提取了 {result.added_count} 个新策略")
        >>> print(f"执行了 {len(result.operations)} 个操作")
    """
    
    def __init__(
        self,
        playbook: Playbook,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_strategies_per_reflection: int = 5
    ):
        """
        初始化策展器 v2.0。
        
        参数：
            playbook: Playbook 实例（用于管理策略）
            model_name: LLM 模型名称
            temperature: 温度参数
            max_strategies_per_reflection: 每次反思最多提取的策略数
        """
        self.playbook = playbook
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.model_name = model_name
        self.max_strategies = max_strategies_per_reflection
    
    def curate(
        self,
        react_question: ReactQuestion,
        react_result: ReactAgentResult,
        evaluator_result: Optional[EvaluatorResult],
        reflection_result: ReflectionResult
    ) -> CuratorResult:
        """
        策展：提取策略并更新 Playbook（v2.0 版本）。
        
        基于 ACE prompts_v2.py 的增强特性：
        - 5 级优先级决策树（关键错误/缺失能力/策略改进/矛盾解决/成功强化）
        - 自动去重检查（70% 相似度）
        - 返回详细操作记录和推理说明
        - Playbook 大小管理
        - JSON Schema 输出验证
        
        参数：
            react_question: 输入问题
            react_result: Agent 的执行结果
            evaluator_result: 评估结果（可选）
            reflection_result: 反思结果
            
        返回：
            CuratorResult 包含提取策略、操作记录、推理说明
        """
        # 1. 先标记已用策略（从 ReflectionResult 中获取）
        marked_count = 0
        if reflection_result.strategy_tags:
            for strategy_tag in reflection_result.strategy_tags:
                strategy_id = strategy_tag.id
                tag = strategy_tag.tag  # helpful, harmful, neutral
                
                if strategy_id and tag:
                    strategy = self.playbook.get_strategy(strategy_id)
                    if strategy:
                        if tag == "helpful":
                            self.playbook.mark_helpful(strategy_id)
                        elif tag == "harmful":
                            self.playbook.mark_harmful(strategy_id)
                        elif tag == "neutral":
                            self.playbook.mark_neutral(strategy_id)
                        marked_count += 1
        
        # 2. 构建问题上下文字符串
        question_context_parts = [f"问题: {react_question.question}"]

        if react_question.context:
            question_context_parts.append(f"上下文: {react_question.context}")
        
        if react_question.ground_truth:
            question_context_parts.append(f"正确答案: {react_question.ground_truth}")
        
        question_context_parts.append(f"输出结果: {react_result.answer}")

        if evaluator_result:
            question_context_parts.append(f"是否正确: {'正确' if evaluator_result.is_correct else '错误'}")
            question_context_parts.append(f"反馈: {evaluator_result.feedback}")
        
        question_context_str = "\n".join(question_context_parts)
        
        # 3. 格式化 v2.0 提示词
        prompt_content = CURATOR_PROMPT_V2.format(
            playbook=self.playbook.as_str(),
            reflection=reflection_result.as_str(),
            stats=self.playbook.stats_str(),
            question_context=question_context_str
        )
        
        try:
            # 4. 调用 LLM
            response = self.llm.invoke([
                {"role": "system", "content": "你是一个专业的策略策展专家，负责管理 Playbook 的质量和效率。请严格按照 JSON Schema 返回结果。"},
                {"role": "user", "content": prompt_content}
            ])
            
            # 5. 解析 JSON 响应
            response_text = response.content.strip()
            
            # 移除可能的 markdown 代码块标记
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            result_json = json.loads(response_text)
            
            # 6. 提取字段
            operations = result_json.get("operations", [])
            reasoning = result_json.get("reasoning", "")
            
            # 7. 执行操作并统计
            added_count = 0
            updated_count = 0
            removed_count = 0
            
            for op in operations:
                op_type = op.get("type")
                
                if op_type == "ADD":
                    # 添加新策略
                    strategy_text = op.get("content", "")
                    category = op.get("category", "general")  # LLM 推理出的分类
                    
                    # 去重检查（70% 相似度）
                    if not self._is_duplicate(strategy_text, self.playbook):
                        self.playbook.add_strategy(
                            content=strategy_text,
                            category=category
                        )
                        added_count += 1
                    else:
                        print(f"⚠️ 跳过重复策略: {strategy_text[:50]}...")
                
                elif op_type == "UPDATE":
                    # 更新现有策略的内容（增量追加）
                    strategy_id = op.get("strategy_id")
                    additional_content = op.get("content", "")
                    
                    if strategy_id and additional_content:
                        success = self.playbook.update_strategy_content(strategy_id, additional_content)
                        if success:
                            print(f"✅ 已更新策略: {strategy_id}")
                            updated_count += 1
                        else:
                            print(f"⚠️ 策略不存在，无法更新: {strategy_id}")
                    else:
                        print(f"⚠️ UPDATE 操作缺少 strategy_id 或 content")
                
                elif op_type == "REMOVE":
                    # 移除策略
                    strategy_id = op.get("strategy_id")
                    if strategy_id is not None:
                        success = self.playbook.remove_strategy(strategy_id)
                        if success:
                            print(f"✅ 已移除策略: {strategy_id}")
                            removed_count += 1
                        else:
                            print(f"⚠️ 策略不存在，无法移除: {strategy_id}")
            
            # 转换为 DeltaOperation 对象
            delta_operations = [
                DeltaOperation(
                    type=op.get("type", ""),
                    content=op.get("content"),
                    category=op.get("category"),
                    strategy_id=op.get("strategy_id"),
                    justification=op.get("justification", "")
                )
                for op in operations
            ]
            
            return CuratorResult(
                reasoning=reasoning,
                operations=delta_operations,
                added_count=added_count,
                updated_count=updated_count,
                removed_count=removed_count,
                marked_count=marked_count
            )
        
        except json.JSONDecodeError as e:
            # JSON 解析失败，返回空结果并记录错误
            print(f"⚠️ Curator v2.0 JSON 解析失败: {e}")
            print(f"响应内容前 200 字符: {response_text[:200]}...")
            print("提示：请检查 LLM 是否正确返回了 JSON 格式")
            
            return CuratorResult(
                reasoning=f"JSON 解析失败: {str(e)}",
                operations=[],
                marked_count=marked_count
            )
        
        except Exception as e:
            print(f"⚠️ Curator v2.0 执行失败: {e}")
            import traceback
            traceback.print_exc()
            
            return CuratorResult(
                reasoning=f"执行失败: {str(e)}",
                operations=[],
                marked_count=marked_count
            )
    
    def _is_duplicate(self, new_strategy: str, playbook: Playbook, threshold: float = 0.7) -> bool:
        """
        检测策略是否重复。
        
        优先使用向量检索（如果启用），否则回退到文本相似度计算。
        
        基于 ACE 的去重协议：
        - 向量检索模式：使用语义相似度（余弦相似度）
        - 文本匹配模式：使用 difflib.SequenceMatcher 计算相似度
        - 阈值为 70%（0.7）
        - 如果与任何已有策略相似度 >= 70%，视为重复
        
        参数：
            new_strategy: 新策略文本
            playbook: 当前 Playbook
            threshold: 相似度阈值（默认 0.7）
        
        返回：
            True 如果是重复，False 如果不重复
        """
        # 如果 Playbook 为空，直接返回 False
        if len(playbook) == 0:
            return False
        
        # 优先使用向量检索
        if playbook.enable_retrieval:
            try:
                # 生成新策略的嵌入
                new_embedding = playbook.embeddings.embed_query(new_strategy)
                
                # 从 ChromaDB 检索最相似的策略
                results = playbook.collection.query(
                    query_embeddings=[new_embedding],
                    n_results=1  # 只检索最相似的一个
                )
                
                # 检查是否有结果
                if results["distances"] and len(results["distances"][0]) > 0:
                    # ChromaDB 返回的是距离（越小越相似），需要转换为相似度
                    # 对于余弦距离：similarity = 1 - distance
                    distance = results["distances"][0][0]
                    similarity = 1 - distance
                    
                    # 如果相似度达到阈值，视为重复
                    if similarity >= threshold:
                        strategy_id = results["ids"][0][0] if results["ids"] else "unknown"
                        print(f"🔍 向量检索检测到重复策略（相似度: {similarity:.2f}，ID: {strategy_id}）")
                        return True
                
                return False
                
            except Exception as e:
                print(f"⚠️  向量检索失败，回退到文本相似度计算: {e}")
                # 继续执行下面的文本相似度计算
        
        # 回退到文本相似度计算（原实现）
        from difflib import SequenceMatcher
        
        all_strategies = playbook.get_all_strategies()
        for existing in all_strategies:
            # 计算相似度
            similarity = SequenceMatcher(
                None,
                new_strategy.lower(),
                existing.content.lower()
            ).ratio()
            
            # 如果相似度达到阈值，视为重复
            if similarity >= threshold:
                print(f"🔍 文本匹配检测到重复策略（相似度: {similarity:.2f}，ID: {existing.id}）")
                return True
        
        return False
