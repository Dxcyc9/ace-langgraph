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

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from playbook import Playbook
from prompts import CURATOR_PROMPT_V2
from agent_types import (
    ReactQuestion, ReactAgentResult, EvaluatorResult,
    ReflectionResult, CuratorResult, DeltaOperation
)

CurationResult = CuratorResult

class Curator:
    def __init__(
        self,
        playbook: Playbook,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_strategies_per_reflection: int = 2
    ):
        self.playbook = playbook
        self.llm = ChatOpenAI(model=model_name, temperature=temperature, base_url="https://api.moonshot.cn/v1")
        self.model_name = model_name
        self.max_strategies = max_strategies_per_reflection

    def curate(
        self,
        react_question: ReactQuestion,
        react_result: ReactAgentResult,
        evaluator_result: Optional[EvaluatorResult],
        reflection_result: ReflectionResult
    ) -> CuratorResult:
        marked_count = 0
        if reflection_result.strategy_tags:
            for strategy_tag in reflection_result.strategy_tags:
                strategy_id = strategy_tag.id
                tag = strategy_tag.tag
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

        prompt_content = CURATOR_PROMPT_V2.format(
            playbook=self.playbook.as_str(),
            reflection=reflection_result.as_str(),
            stats=self.playbook.stats_str(),
            question_context=question_context_str,
            current_date=datetime.now().strftime("%Y-%m-%d")
        )

        try:
            response = self.llm.invoke([
                {"role": "system", "content": "你是一个专业的策略策展专家，负责管理 Playbook 的质量和效率。请严格按照 JSON Schema 返回结果。"},
                {"role": "user", "content": prompt_content}
            ])
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            result_json = json.loads(response_text)
            operations = result_json.get("operations", [])
            reasoning = result_json.get("reasoning", "")

            # 术语规范：在执行前统一将专有术语规范为英文（防止歧义）
            def _norm(text: Optional[str]) -> str:
                if not text:
                    return ""
                return self._normalize_terms_to_english(text)

            operations = [
                {
                    **op,
                    "content": _norm(op.get("content")),
                    "justification": _norm(op.get("justification")),
                }
                for op in operations
            ]

            # 增长控制：限制每轮新增数量并去重（超预算的 ADD 转为 UPDATE 合并到最相似策略）
            max_add = max(1, min(self.max_strategies, 2))
            add_count = 0
            seen_add_contents = set()
            filtered_ops = []
            for op in operations:
                if op.get("type") == "ADD":
                    content_norm = (op.get("content", "").strip().lower())
                    if not content_norm:
                        continue
                    if content_norm in seen_add_contents:
                        # 跳过完全重复的新增
                        continue
                    if add_count < max_add:
                        filtered_ops.append(op)
                        seen_add_contents.add(content_norm)
                        add_count += 1
                    else:
                        # 超预算：转为 UPDATE 合并到最相似策略
                        similar_id = self._find_most_similar_strategy_id(op.get("content", ""))
                        if similar_id:
                            op["type"] = "UPDATE"
                            op["strategy_id"] = similar_id
                            op["justification"] = (
                                        op.get("justification", "") + "（超过新增预算→更新最相似策略）").strip()
                            filtered_ops.append(op)
                        # 若找不到相似策略，直接跳过该操作
                else:
                    filtered_ops.append(op)

            # 错误案例强制增量修正（保持原逻辑）
            enforced_ops = list(filtered_ops)

            # 来自反思的一般策略，直接作为 ADD 预置（避免丢失）
            try:
                for gs in getattr(reflection_result, "general_strategies", []) or []:
                    enforced_ops.insert(0, {
                        "type": "ADD",
                        "category": gs.category or "general",
                        "content": gs.content,
                        "justification": "来自反思 general_strategies"
                    })
            except Exception:
                pass

            need_enforce = evaluator_result and (not evaluator_result.is_correct)
            if need_enforce:
                existing_update_ids = {op.get("strategy_id") for op in enforced_ops if op.get("type") == "UPDATE"}
                target_id = None
                try:
                    harmful_ids = [t.id for t in (reflection_result.strategy_tags or []) if t.tag in ("harmful", "neutral")]
                    target_id = harmful_ids[0] if harmful_ids else None
                except Exception:
                    target_id = None
                if not target_id:
                    try:
                        used_ids = react_result.used_strategies or []
                        target_id = used_ids[0] if used_ids else None
                    except Exception:
                        target_id = None
                if target_id and target_id not in existing_update_ids:
                    supplemental = reflection_result.key_insight or reflection_result.correct_approach or "根据案例修正列归属、JOIN 键与引号规范。"
                    enforced_ops.append({
                        "type": "UPDATE",
                        "strategy_id": target_id,
                        "content": ("补充：" + str(supplemental))[:200],
                        "justification": "自动规则：错误案例触发针对已引用策略的增量修正"
                    })

            added_count = 0
            updated_count = 0
            removed_count = 0
            delta_operations = []

            for op in enforced_ops:
                op_type = op.get("type")
                applied = False
                skip_reason = None

                if op_type == "ADD":
                    strategy_text = op.get("content", "")
                    category = op.get("category", "general")
                    if not self._is_duplicate(strategy_text, self.playbook):
                        self.playbook.add_strategy(content=strategy_text, category=category)
                        added_count += 1
                        applied = True
                    else:
                        skip_reason = "与现有策略相似度过高"
                        applied = False
                        similar_id = self._find_most_similar_strategy_id(strategy_text)
                        if similar_id:
                            success = self.playbook.update_strategy_content(similar_id, strategy_text)
                            if success:
                                print(f"✅ 已更新策略: {similar_id}（重复新增→更新）")
                                updated_count += 1
                                applied = True
                                op["type"] = "UPDATE"
                                op["strategy_id"] = similar_id
                                op["justification"] = (op.get("justification", "") + "（重复→更新最相似策略）").strip()

                            else:
                                skip_reason = "not_found"
                        else:
                            print(f"⚠️ 跳过重复策略: {strategy_text[:50]}...")
                            skip_reason = "duplicate"

                elif op_type == "UPDATE":
                    strategy_id = op.get("strategy_id")
                    additional_content = op.get("content", "")
                    if strategy_id and additional_content:
                        success = self.playbook.update_strategy_content(strategy_id, additional_content)
                        if success:
                            print(f"✅ 已更新策略: {strategy_id}")
                            updated_count += 1
                            applied = True
                        else:
                            print(f"⚠️ 策略不存在，无法更新: {strategy_id}")
                            skip_reason = "not_found"
                    else:
                        print(f"⚠️ UPDATE 操作缺少 strategy_id 或 content")
                        skip_reason = "invalid_op"

                elif op_type == "REMOVE":
                    strategy_id = op.get("strategy_id")
                    if strategy_id is not None:
                        success = self.playbook.remove_strategy(strategy_id)
                        if success:
                            print(f"✅ 已移除策略: {strategy_id}")
                            removed_count += 1
                            applied = True
                        else:
                            print(f"⚠️ 策略不存在，无法移除: {strategy_id}")
                            skip_reason = "not_found"
                    else:
                        print(f"⚠️ REMOVE 操作缺少 strategy_id")
                        skip_reason = "invalid_op"

                delta_operations.append(
                    DeltaOperation(
                        type=op.get("type", ""),
                        content=op.get("content"),
                        category=op.get("category"),
                        strategy_id=op.get("strategy_id"),
                        justification=op.get("justification", ""),
                        applied=applied,
                        skip_reason=skip_reason
                    )
                )

            return CuratorResult(
                reasoning=reasoning,
                operations=delta_operations,
                added_count=added_count,
                updated_count=updated_count,
                removed_count=removed_count,
                marked_count=marked_count
            )

        except json.JSONDecodeError as e:
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

    def _is_duplicate(self, new_strategy: str, playbook: Playbook, threshold: float = 0.90) -> bool:
        if len(playbook) == 0:
            return False
        size = len(playbook)
        threshold = 0.90 if size < 50 else (0.85 if size < 100 else 0.80)
        if playbook.enable_retrieval:
            try:
                new_embedding = playbook.embeddings.embed_query(new_strategy)
                results = playbook.collection.query(query_embeddings=[new_embedding], n_results=1)
                if results.get("distances") and len(results["distances"][0]) > 0:
                    distance = results["distances"][0][0]
                    similarity = 1 - distance
                    if similarity >= threshold:
                        strategy_id = results["ids"][0][0] if results["ids"] else "unknown"
                        print(f"🔍 向量检索检测到重复策略（相似度: {similarity:.2f}，ID: {strategy_id}）")
                        return True
                return False
            except Exception as e:
                print(f"⚠️  向量检索失败，回退到文本相似度计算: {e}")
        from difflib import SequenceMatcher
        for existing in playbook.get_all_strategies():
            similarity = SequenceMatcher(None, new_strategy.lower(), existing.content.lower()).ratio()
            if similarity >= threshold:
                print(f"🔍 文本匹配检测到重复策略（相似度: {similarity:.2f}，ID: {existing.id}）")
                return True
        return False

    def _find_most_similar_strategy_id(self, content: str) -> Optional[str]:
        try:
            if self.playbook.enable_retrieval and hasattr(self.playbook, "embeddings"):
                emb = self.playbook.embeddings.embed_query(content)
                results = self.playbook.collection.query(query_embeddings=[emb], n_results=1)
                if results and results.get("ids") and results["ids"][0]:
                    return results["ids"][0][0]
        except Exception:
            pass
        from difflib import SequenceMatcher
        best_id = None
        best_sim = -1.0
        for s in self.playbook.get_all_strategies():
            sim = SequenceMatcher(None, content.lower(), s.content.lower()).ratio()
            if sim > best_sim:
                best_sim = sim
                best_id = s.id
        return best_id

    def _sanitize_proprietary_terms(self, text: str) -> str:
        return text  # 该类不做写入清洗

    def _normalize_terms_to_english(self, text: str) -> str:
        """将常见专有术语的中文表述规范为英文，避免歧义。仅替换非 SQL 标识的中文词汇。"""
        if not text:
            return text
        replacements = {
            "完全虚拟": "exclusively virtual",
            "全虚拟": "exclusively virtual",
            "虚拟学校": "virtual school",
            "小学学区": "Elementary School District",
            "统一学区": "Unified School District",
        }
        normed = text
        for zh, en in replacements.items():
            normed = normed.replace(zh, en)
        return normed