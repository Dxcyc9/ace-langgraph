"""
Evaluator - 检查 agent 的答案是否正确

使用 LLM 进行智能评估，替代简单的字符串匹配。
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import json
import sys
import os
import sqlite3
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 统一使用 ace_langgraph 前缀导入
from agent_types import ReactQuestion, ReactAgentResult, EvaluatorResult



# LLM 评估提示词（Text-to-SQL 语义等价优先）
EVALUATOR_PROMPT = """你是 Text-to-SQL 评估专家，基于语义等价判定 AI Agent 的答案是否正确。

【评估准则】
- 语义优先：候选 SQL 与期望 SQL 等价或返回相同类型与集合的数据，即判为正确。
- 结构容忍：允许 T1/T2 别名方向不同、JOIN 两侧翻转、ABS(A-B)≡ABS(B-A)、CAST(... AS REAL)≡* 1.0 等价。
- 引号与样式：允许大小写、空格、引号风格差异（列名含空格/括号通常双引号），不据此判错；字符串常量一般用单引号。
- 列与顺序：仅当问题明确要求排序/Top-K/特定列时严格检查 ORDER BY/LIMIT/必需列；否则顺序可忽略，附加列不影响正确性。
- 结果意图：以问题意图和期望列头为准，判断候选 SQL 是否满足同一信息需求。

【输出格式】
{
  "is_correct": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "简短说明判断依据（1-2句话）",
  "feedback": "给 Agent 的反馈信息"
}

【示例】
1) 等价的别名与列归属翻转：
期望：SELECT T1.School, T1.Street FROM schools AS T1 INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode WHERE ABS(T2."Enrollment (K-12)" - T2."Enrollment (Ages 5-17)") > 30
实际：SELECT T2.School, T2.Street FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE ABS(T1."Enrollment (K-12)" - T1."Enrollment (Ages 5-17)") > 30
评估：{
  "is_correct": true,
  "confidence": 0.95,
  "reasoning": "别名方向与列归属翻转但语义一致，满足同一信息需求",
  "feedback": "正确。注意统一别名风格可提升可读性。"
}

2) 浮点写法差异：
期望：... WHERE CAST(A AS REAL) / B > 0.3
实际：... WHERE A * 1.0 / B > 0.3
评估：{
  "is_correct": true,
  "confidence": 0.95,
  "reasoning": "两种写法语义等价，均确保浮点除法",
  "feedback": "正确。保持一种写法风格即可。"
}

3) 缺失排序/Top-K：
期望：... ORDER BY X DESC LIMIT 3
实际：... （无排序与限制）
评估：{
  "is_correct": false,
  "confidence": 0.9,
  "reasoning": "问题要求 Top-K 或排序，候选 SQL 未满足输出约束",
  "feedback": "请补充 ORDER BY 与 LIMIT 以满足题目。"
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
            temperature=temperature, base_url="https://api.moonshot.cn/v1"
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

        # 优先尝试执行结果对比（Text-to-SQL 语义评估）
        try:
            exec_result = self._evaluate_by_execution(
                question=react_question,
                answer_sql=react_result.answer,
                ground_truth_sql=ground_truth
            )
            if exec_result is not None:
                return exec_result
        except Exception:
            pass

        # 回退到 LLM 文本评估
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

    def _resolve_db_path(self, db_path: str) -> str:
        """解析 SQLite 路径，兼容项目内/容器挂载/环境变量。"""
        raw = db_path or os.getenv("SQLITE_DB_PATH") or "data/sqlite/california_schools.sqlite"
        if os.path.isabs(raw) and os.path.isfile(raw):
            return raw
        candidate1 = os.path.join(str(project_root), raw)
        if os.path.isfile(candidate1):
            return candidate1
        candidate2 = os.path.join(str(project_root), raw.lstrip('/'))
        if os.path.isfile(candidate2):
            return candidate2
        env_path = os.getenv("SQLITE_DB_PATH")
        if env_path and os.path.isfile(env_path):
            return env_path
        fallback = os.path.join(str(project_root), "data/sqlite/california_schools.sqlite")
        return fallback if os.path.isfile(fallback) else raw

    def _run_sql(self, sql: str, limit: int = 50) -> Optional[dict]:
        """只读执行 SQL，返回 {"headers": [...], "rows": [...]}，失败返回 None。"""
        try:
            sql_text = (sql or "").strip()
            if not sql_text:
                return None
            head = sql_text.split()[0].upper()
            if head not in {"SELECT", "WITH", "EXPLAIN"}:
                return None
            path = self._resolve_db_path(os.getenv("SQLITE_DB_PATH") or "data/sqlite/california_schools.sqlite")
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            cur = conn.cursor()
            cur.execute(sql_text)
            rows = cur.fetchmany(limit)
            headers = [d[0] for d in cur.description] if cur.description else []
            conn.close()
            return {"headers": headers, "rows": rows}
        except Exception:
            return None

    def _evaluate_by_execution(self, question: ReactQuestion, answer_sql: str, ground_truth_sql: str) -> Optional[EvaluatorResult]:
        """执行候选与期望 SQL，基于结果语义判断正确性；不可用时返回 None。"""
        ans = self._run_sql(answer_sql, limit=50)
        gt = self._run_sql(ground_truth_sql, limit=50)
        if ans is None or gt is None:
            return None

        gt_headers = gt["headers"] or []
        ans_headers = ans["headers"] or []

        def norm_val(v):
            try:
                return round(float(v), 8)
            except Exception:
                return str(v)

        def project_rows(rows, indices):
            return [tuple(norm_val(r[i]) for i in indices) for r in rows]

        # 1) 首选严格映射（按列名）
        try:
            idx_map = [ans_headers.index(h) for h in gt_headers]
            ans_proj = set(project_rows(ans["rows"], idx_map))
            gt_set = set(tuple(norm_val(v) for v in r) for r in gt["rows"])
            if ans_proj == gt_set:
                return EvaluatorResult(is_correct=True, feedback="执行结果与期望一致（忽略行顺序）。", confidence=0.95)
        except ValueError:
            pass

        # 2) 列名不一致时的宽松比较
        ans_cols = len(ans_headers)
        gt_cols = len(gt_headers)

        # 单列场景：忽略列名，直接比较值集合
        if gt_cols == 1 and ans_cols >= 1:
            ans_vals = set(project_rows(ans["rows"], [0]))
            gt_vals = set(project_rows(gt["rows"], [0]))
            if ans_vals == gt_vals:
                return EvaluatorResult(is_correct=True, feedback="执行结果与期望一致（忽略列名差异）。", confidence=0.93)

        # 多列场景：列数一致时按位置比较（忽略列名）
        if ans_cols == gt_cols and gt_cols > 0:
            ans_set = set(project_rows(ans["rows"], list(range(ans_cols))))
            gt_set = set(project_rows(gt["rows"], list(range(gt_cols))))
            if ans_set == gt_set:
                return EvaluatorResult(is_correct=True, feedback="执行结果与期望一致（忽略列名差异与行序）。", confidence=0.92)

        return EvaluatorResult(is_correct=False, feedback="执行结果与期望不一致。请检查过滤条件、JOIN 与排序。", confidence=0.9)
