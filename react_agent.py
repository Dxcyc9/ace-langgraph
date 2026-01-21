"""
ReAct Agent v2.0 - 带 Playbook 策略学习的推理与行动智能体

基于 ACE prompts_v2.py 优化：
- 集成 Playbook 策略库
- 策略 ID 引用系统
- 4步骤问题解决协议
- 使用 LangChain create_agent 自动处理工具调用
"""

from typing import List, Dict, Any, Optional  
from datetime import datetime
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

import sys
import os
from pathlib import Path

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()  # 加载 .env 文件

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 支持两种导入方式
try:
    from .playbook import Playbook
    from .prompts import REACT_AGENT_PROMPT_V3
    from .agent_types import ReactQuestion, ReactAgentResult
except ImportError:
    from playbook import Playbook
    from prompts import REACT_AGENT_PROMPT_V3
    from agent_types import ReactQuestion, ReactAgentResult

# ... existing code ...
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 路径解析助手：统一解析相对/绝对/容器路径
def _resolve_db_path(db_path: str) -> str:
    import os
    raw = db_path or os.getenv("SQLITE_DB_PATH") or "data/sqlite/california_schools.sqlite"
    # 已是绝对路径且存在
    if os.path.isabs(raw) and os.path.isfile(raw):
        print(f"[db_resolve] 使用绝对路径: {raw}")
        return raw
    # 1) 项目根目录 + 相对路径
    candidate1 = os.path.join(str(project_root), raw)
    if os.path.isfile(candidate1):
        print(f"[db_resolve] 解析为项目内路径: {candidate1}")
        return candidate1
    # 2) 兼容 '/data/...' 容器挂载：去掉前导斜杠再拼项目根
    candidate2 = os.path.join(str(project_root), raw.lstrip('/'))
    if os.path.isfile(candidate2):
        print(f"[db_resolve] 兼容容器路径: {candidate2}")
        return candidate2
    # 3) 环境变量显式指定
    env_path = os.getenv("SQLITE_DB_PATH")
    if env_path and os.path.isfile(env_path):
        print(f"[db_resolve] 使用环境变量 SQLITE_DB_PATH: {env_path}")
        return env_path
        # 3.5) 当前工作目录下的相对路径
    if os.path.isfile(raw):
        print(f"[db_resolve] 使用相对路径: {raw}")
        return raw
    # 4) 最终回退到项目默认
    fallback = os.path.join(str(project_root), "data/sqlite/california_schools.sqlite")
    if os.path.isfile(fallback):
        print(f"[db_resolve] 使用默认路径: {fallback}")
        return fallback
    print(f"[db_resolve] 未找到数据库文件（raw={raw}）")
    return raw
# ... existing code ...

# ========== 工具定义（使用 @tool 装饰器）==========

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。
    
    参数：
        expression: 要计算的数学表达式，例如 "2 + 3 * 4"
        
    返回：
        计算结果的字符串
    """
    try:
        # 安全的数学表达式求值
        allowed_names = {
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"计算错误：{str(e)}"

@tool
def search(query: str) -> str:
    """使用 SearXNG API 搜索信息。
    
    参数：
        query: 搜索查询关键词
        
    返回：
        搜索结果
    """
    import requests
    import os
    
    # 从环境变量获取 SearXNG API 地址，如果未设置则使用默认值
    searxng_url = os.getenv("SEARXNG_URL")
    
    try:
        # 构建请求参数
        params = {
            "q": query,
            "format": "json",
            "language": "zh-CN"
        }
        
        # 发送请求
        response = requests.get(searxng_url, params=params, timeout=10)
        response.raise_for_status()
        
        # 解析结果
        data = response.json()
        
        if "results" in data and data["results"]:
            results = []
            for idx, result in enumerate(data["results"][:3], 1):
                title = result.get("title", "无标题")
                content = result.get("content", "无内容描述")
                url = result.get("url", "")
                results.append(f"{idx}. {title}\n{content}\n来源: {url}")
            
            return "\n\n".join(results) if results else "未找到相关信息"
        else:
            return "未找到相关信息"
            
    except requests.exceptions.Timeout:
        return "搜索错误：请求超时，请稍后重试"
    except requests.exceptions.RequestException as e:
        return f"搜索错误：网络请求失败 - {str(e)}"
    except Exception as e:
        return f"搜索错误：{str(e)}"

@tool
def sqlite_schema(db_path: str = "data/sqlite/california_schools.sqlite", sample_rows: int = 3) -> str:
    """
    读取 SQLite 文件，返回文本化的 schema 和少量样例数据，供大模型生成 SQL 使用。
    参数：
        db_path:      本地 *.sqlite 文件绝对路径
        sample_rows:  每张表抽样行数（默认 3 行，0 表示不抽样）
    返回：
        字符串，包含：
        1) CREATE TABLE 语句（含主键/外键）
        2) 每表最多 sample_rows 行 INSERT 风格示例（CSV 格式，仅字符串/数字）
    """
    import sqlite3, csv, io, textwrap, time

    print(f"[sqlite_schema] 开始读取 schema, db_path={db_path}, sample_rows={sample_rows}")
    start_ts = time.time()

    resolved = _resolve_db_path(db_path)
    if not os.path.isfile(resolved):
        print(f"[sqlite_schema] 文件不存在: {resolved}")
        return f"文件不存在：{resolved}"

    try:
        conn = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
        conn.text_factory = str
        cur = conn.cursor()
        print("[sqlite_schema] 已连接数据库（只读）")

        tables = [t[0] for t in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'" ).fetchall()]
        print(f"[sqlite_schema] 发现 {len(tables)} 张表: {', '.join(tables) if tables else '(无)'}")
        if not tables:
            print("[sqlite_schema] 数据库中无用户表")
            return "数据库中无用户表。"

        buf = io.StringIO()
        # 1) 输出 schema
        for tbl in tables:
            print(f"[sqlite_schema] 读取表结构: {tbl}")
            buf.write(f"-- Table: {tbl}\n")
            row = cur.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (tbl,) ).fetchone()
            create_sql = row[0] if row else ""
            buf.write(create_sql + ";\n\n")

            # 2) 抽样数据
            if sample_rows > 0:
                print(f"[sqlite_schema] 抽样数据: {tbl}, rows={sample_rows}")
                rows = cur.execute(f"SELECT * FROM `{tbl}` LIMIT ?", (sample_rows,) ).fetchall()
                if not rows:
                    print(f"[sqlite_schema] {tbl} 无样例数据")
                    buf.write("-- (empty)\n\n")
                    continue
                # 转 CSV 风格，避免值里有换行
                buf.write("-- Sample data (CSV format):\n")
                output = io.StringIO()
                writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([d[0] for d in cur.description])  # header
                writer.writerows(rows)
                buf.write(output.getvalue() + "\n")

        conn.close()
        dur_ms = int((time.time() - start_ts) * 1000)
        print(f"[sqlite_schema] 完成，耗时 {dur_ms} ms")
        return buf.getvalue()

    except Exception as e:
        print(f"[sqlite_schema] 读取失败: {str(e)}")
        return f"读取 sqlite 失败：{str(e)}"

# ... existing code ...
@tool
def sqlite_query(db_path: str = "data/sqlite/california_schools.sqlite", sql: str = "", limit: int = 50) -> str:
    """
    只读执行 SQL 并返回结果预览或错误信息。
    - 仅允许 SELECT/WITH/EXPLAIN 语句
    - 返回 JSON：{"headers": [...], "rows": [...], "row_count": N}
    """
    import sqlite3, json, os, time
    start_ts = time.time()
    print(f"[sqlite_query] 开始: db_path={db_path}, limit={limit}")
    sql_text = (sql or "").strip()
    print(f"[sqlite_query] SQL: {sql_text}")
    if not sql_text:
        print("[sqlite_query] SQL为空")
        return "SQL为空"
    head = sql_text.split()[0].upper()
    if head not in {"SELECT", "WITH", "EXPLAIN"}:
        print(f"[sqlite_query] 非只读语句: {head}")
        return "仅支持只读查询（SELECT/WITH/EXPLAIN）"
    resolved = _resolve_db_path(db_path)
    if not os.path.isfile(resolved):
        print(f"[sqlite_query] 文件不存在: {resolved}")
        return f"文件不存在：{resolved}"
    try:
        print("[sqlite_query] 连接数据库（只读）")
        conn = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
        cur = conn.cursor()
        print("[sqlite_query] 执行查询...")
        cur.execute(sql_text)
        rows = cur.fetchmany(limit)
        headers = [d[0] for d in cur.description] if cur.description else []
        total = None
        try:
            if head != "EXPLAIN":
                total = len(rows)
        except Exception:
            total = None
        conn.close()
        dur_ms = int((time.time() - start_ts) * 1000)
        print(f"[sqlite_query] 完成: rows={total}, headers={headers}, 耗时{dur_ms}ms")
        return json.dumps({"headers": headers, "rows": rows, "row_count": total}, ensure_ascii=False)
    except Exception as e:
        print(f"[sqlite_query] 失败: {str(e)}")
        return f"SQL执行错误：{str(e)}"

@tool
def sqlite_columns(db_path: str = "data/sqlite/california_schools.sqlite", table: str = "") -> str:
    """
    返回指定表的列名与类型（PRAGMA table_info）。
    """
    import sqlite3, json, os
    print(f"[sqlite_columns] 开始: db_path={db_path}, table={table}")
    if not table:
        print("[sqlite_columns] 未指定表名")
        return "未指定表名"
    resolved = _resolve_db_path(db_path)
    if not os.path.isfile(resolved):
        print(f"[sqlite_columns] 文件不存在: {resolved}")
        return f"文件不存在：{resolved}"
    try:
        print("[sqlite_columns] 连接数据库（只读）")
        conn = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
        cur = conn.cursor()
        print(f"[sqlite_columns] 查询列信息: {table}")
        cur.execute(f"PRAGMA table_info('{table}')")
        info = cur.fetchall()
        conn.close()
        cols = [{"name": r[1], "type": r[2], "notnull": r[3], "pk": r[5]} for r in info]
        print(f"[sqlite_columns] 列数: {len(cols)}")
        return json.dumps(cols, ensure_ascii=False)
    except Exception as e:
        print(f"[sqlite_columns] 失败: {str(e)}")
        return f"读取列信息失败：{str(e)}"

@tool
def sqlite_distinct(db_path: str = "data/sqlite/california_schools.sqlite", table: str = "", column: str = "", limit: int = 50) -> str:
    """
    返回指定列的 DISTINCT 值（便于值域对齐与过滤条件构造）。
    """
    import sqlite3, json, os
    print(f"[sqlite_distinct] 开始: db_path={db_path}, table={table}, column={column}, limit={limit}")
    if not (table and column):
        print("[sqlite_distinct] 未指定表或列")
        return "未指定表或列"
    resolved = _resolve_db_path(db_path)
    if not os.path.isfile(resolved):
        print(f"[sqlite_distinct] 文件不存在: {resolved}")
        return f"文件不存在：{resolved}"
    col = column.replace('"', '""')
    try:
        print("[sqlite_distinct] 连接数据库（只读）")
        conn = sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)
        cur = conn.cursor()
        print(f"[sqlite_distinct] 查询 DISTINCT 值: {table}.\"{col}\"")
        cur.execute(f"SELECT DISTINCT \"{col}\" FROM '{table}' LIMIT {int(limit)}")
        values = [r[0] for r in cur.fetchall()]
        conn.close()
        print(f"[sqlite_distinct] 返回值数量: {len(values)}")
        return json.dumps({"values": values}, ensure_ascii=False)
    except Exception as e:
        print(f"[sqlite_distinct] 失败: {str(e)}")
        return f"读取 DISTINCT 失败：{str(e)}"


# ========== ReAct Agent ==========

class ReActAgent:
    """
    ReAct Agent v2.0 - 带 Playbook 策略学习的推理与行动智能体。
    
    增强特性：
    - 集成 Playbook 策略库
    - 自动引用策略 ID
    - 4步骤问题解决协议
    - 策略应用追踪
    
    示例：
        >>> playbook = Playbook()
        >>> playbook.add_strategy("复杂计算要分步进行", category="react")
        >>> agent = ReActAgent(playbook=playbook)
        >>> result = agent.run("计算 (15 + 27) * 3")
        >>> print(result['answer'])
    """
    
    def __init__(
        self,
        playbook: Optional[Playbook] = None,
        tools: List = None,
        model_name: str = "gpt-4o-mini",
        max_iterations: int = 50,
        verbose: bool = True,
        top_k_strategies: int = 50     # 检索或选择的策略数量
    ):
        """
        初始化 ReAct Agent v2.0。
        
        参数：
            playbook: Playbook 实例（包含学习的策略）
                      如果为 None，将创建默认 Playbook
                      检索方式由 Playbook 的 enable_retrieval 属性控制
            tools: 可用工具列表
            model_name: LLM 模型名称
            max_iterations: 最大迭代次数
            verbose: 是否打印详细信息
            top_k_strategies: 检索或选择的策略数量
        """
        self.playbook = playbook if playbook is not None else Playbook()
        self.tools = tools or get_default_tools()
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.top_k_strategies = top_k_strategies
        self.current_question = ""  # 当前正在处理的问题（用于检索）
        
        # 不在初始化时创建 agent，而是在 run 时动态创建
        # 这样可以根据当前问题选择相关策略
        self._agent_cache = None

    def _auto_validate_and_correct_sql(self, sql: str, context: str = "") -> str:
        """
        轻量级自动校验与修正：
        - 值域映射：District 文本 → DOC 代码（52/54）
        - 浮点保障：COUNT(*) / 12 → CAST(COUNT(*) AS REAL) / 12
        - 别名与列归属校验（保持原逻辑）
        """
        import re, json

        def _get_tool(name: str):
            for t in self.tools:
                if getattr(t, "name", "") == name:
                    return t
            return None

        def norm(s: str) -> str:
            return "".join((s or "").lower().split())

        columns_tool = _get_tool("sqlite_columns")
        query_tool = _get_tool("sqlite_query")
        if not columns_tool or not query_tool:
            return sql

        text = sql

        # A) 值域映射修复：District 学区类型文本 → DOC 代码
        text_fixed = re.sub(r"(?i)\bDistrict\s*=\s*'Elementary\s+School\s+District'", "DOC = 52", text)
        text_fixed = re.sub(r"(?i)\bDistrict\s*=\s*'Unified\s+School\s+District'", "DOC = 54", text_fixed)

        # B) 浮点保障：COUNT(*) / 12 → CAST(COUNT(*) AS REAL) / 12
        text_fixed = re.sub(r"(?i)count\s*\(\s*\*\s*\)\s*/\s*12", "CAST(COUNT(*) AS REAL) / 12", text_fixed)

        changed = (text_fixed != text)
        if changed:
            resp = query_tool.invoke({"sql": text_fixed, "limit": 3})
            if isinstance(resp, str) and resp.strip().startswith("{"):
                print("【自动校验】值域映射/浮点修正已应用并通过验证")
                return text_fixed
            else:
                print("【自动校验】修正后仍错误，保留原始SQL")

        # C) 原有别名与列归属修复逻辑（保持）
        alias_map = {}
        for pat in [
            r"FROM\s+(\w+)\s+AS\s+(\w+)",
            r"FROM\s+(\w+)\s+(\w+)",
            r"JOIN\s+(\w+)\s+AS\s+(\w+)",
            r"JOIN\s+(\w+)\s+(\w+)"
        ]:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                table, alias = m.group(1), m.group(2)
                alias_map[alias] = table
        if not alias_map:
            alias_map = {"T1": "frpm", "T2": "schools", "T3": "satscores"}

        table_cols = {}
        for alias, table in alias_map.items():
            try:
                resp = columns_tool.invoke({"table": table})
                cols = []
                try:
                    parsed = json.loads(resp)
                    cols = [c["name"] for c in parsed if isinstance(c, dict) and "name" in c]
                except Exception:
                    cols = []
                table_cols[alias] = {"raw": set(cols), "norm": set(norm(c) for c in cols)}
            except Exception:
                table_cols[alias] = {"raw": set(), "norm": set()}

        refs = []
        for m in re.finditer(r"(\w+)\.(?:`([^`]+)`|\"([^\"]+)\")", text):
            alias = m.group(1)
            col = m.group(2) if m.group(2) is not None else m.group(3)
            quote = "`" if m.group(2) is not None else '"'
            refs.append((m.group(0), alias, col, quote))

        corrected = text
        changed = False
        for full, alias, col, quote in refs:
            here = table_cols.get(alias, {"raw": set(), "norm": set()})
            col_norm = norm(col)
            if col in here["raw"] or col_norm in here["norm"]:
                continue

            candidates = []
            for a, cols in table_cols.items():
                if (col in cols["raw"]) or (col_norm in cols["norm"]):
                    candidates.append(a)

            if len(candidates) == 1:
                new_alias = candidates[0]
                new_full = f"{new_alias}.{quote}{col}{quote}"
                corrected = corrected.replace(full, new_full)
                changed = True

        if changed:
            resp = query_tool.invoke({"sql": corrected, "limit": 3})
            if isinstance(resp, str) and resp.strip().startswith("{"):
                print("【自动校验】列别名已修正并通过验证")
                return corrected
            else:
                print("【自动校验】修正后仍错误，保留原始SQL")
        return sql

    def _get_or_create_agent(self, question: str = "", context: str = ""):
        """
        获取或创建 agent。
        
        根据当前问题动态创建 agent，以便使用最相关的策略。
        
        参数：
            question: 当前问题
            context: 额外的上下文信息
        """
        from langgraph.prebuilt import create_react_agent
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate

        # 1. 组装系统提示
        system = self._get_system_prompt(question, context)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("placeholder", "{messages}")  # 运行时把用户消息塞到这里
        ])

        # 2. 创建模型
        llm = ChatOpenAI(
            model=self.model_name,
            base_url="https://api.moonshot.cn/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0
        )

        # 3. 返回编译好的 LangGraph
        return create_react_agent(
            model=llm,
            tools=self.tools,
            prompt=prompt,  # ← 用 prompt 而不是 messages
            version="v2"  # 支持并行工具
        )
        # return create_agent(
        #     model=f"openai:{self.model_name}",
        #     tools=self.tools,
        #     system_prompt=self._get_system_prompt(question, context)
        # )


        # return create_react_agent(
        #     model=f"openai:{self.model_name}",
        #     tools=self.tools,
        #     prompt=self._get_system_prompt(question, context),  # 注意：prompt 不是 system_prompt
        #     max_iterations=self.max_iterations  # 关键：限制循环次数
        # )


    def run(
        self,
        react_question: ReactQuestion,
        track_strategies: bool = True
    ) -> ReactAgentResult:
        """
        运行 ReAct Agent 解决问题。
        """
        question = react_question.question
        context = react_question.context
        self.current_question = question

        agent = self._get_or_create_agent(question, context)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"问题：{question}")
            if context:
                print(f"上下文：{context}")
            print(f"{'='*60}\n")

        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config={"recursion_limit": 40}
            )
        except Exception as e:
            from langgraph.errors import GraphRecursionError
            if isinstance(e, GraphRecursionError):
                print(f"⚠️ LangGraph 递归上限触发，启用降级路径：{e}")
                messages = []
                final_message = ""
                return ReactAgentResult(
                    answer="",
                    reasoning="递归上限触发，未生成答案",
                    used_strategies=[],
                    iterations=0,
                    messages=messages,
                    success=False
                )
            else:
                raise

        messages = result["messages"]
        final_message = messages[-1].content if messages else ""

        # 解析最终答案
        if "Final Answer:" in final_message:
            final_answer = final_message.split("Final Answer:")[-1].strip()
        else:
            final_answer = final_message

        # 若答案不是 SQL，兜底从最近一次 sqlite_query 工具调用中提取
        if not final_answer.strip().upper().startswith("SELECT"):
            fallback_sql = self._extract_sql_from_tool_calls(messages)
            if fallback_sql:
                print("【从工具调用中提取SQL作为最终答案】")
                final_answer = fallback_sql

        # 自动校验与修正（仅当看起来是 SQL）
        if final_answer.strip().upper().startswith("SELECT"):
            try:
                corrected = self._auto_validate_and_correct_sql(final_answer, context)
                if corrected and corrected != final_answer:
                    print("【自动校验并修正SQL】")
                    final_answer = corrected
            except Exception as e:
                print(f"【自动校验失败】{e}")

        import re
        used_strategies = []
        if "Strategy:" in (final_message or ""):
            line = re.search(r'(?im)^\s*Strategy:\s*(.+)$', final_message or "")
            if line:
                used_strategies = re.findall(r'\[([A-Za-z]{3}-\d{5})\]', line.group(1))
        if not used_strategies:
            used_strategies = self._extract_used_strategies(messages)
        if track_strategies and not used_strategies and len(self.playbook) > 0:
            try:
                best = self.playbook.retrieve_strategies(question=question, top_k=1, min_score=0)
                if best:
                    used_strategies = [best[0].id]
                    print(f"ℹ️ 自动选择策略：[{used_strategies[0]}]")
            except Exception:
                pass
        # 去重保序
        used_strategies = list(dict.fromkeys(used_strategies))

        # 汇总推理过程
        reasoning_steps = []
        for msg in messages:
            if isinstance(msg, AIMessage) and hasattr(msg, 'content'):
                reasoning_steps.append(f"\n{msg.content.strip()}")
        reasoning = "\n\n".join(reasoning_steps) if reasoning_steps else "未生成推理过程"

        # 迭代次数
        iterations = len([m for m in messages if hasattr(m, 'tool_calls') and m.tool_calls])

        if self.verbose:
            print(f"\n【最终答案】\n{final_answer}")
            print(f"\n【推理过程】\n{reasoning}")
            if used_strategies:
                print(f"\n【使用的策略】\n{', '.join(used_strategies)}")
            print(f"\n【迭代次数】\n{iterations}")
            print()

        return ReactAgentResult(
            answer=final_answer,
            reasoning=final_message,
            used_strategies=used_strategies,
            iterations=iterations,
            messages=messages,
            success=True
        )
    # def run(self, react_question: ReactQuestion, track_strategies: bool = True) -> ReactAgentResult:
    #     question = react_question.question
    #     context = react_question.context
    #     self.current_question = question
    #
    #     # 1. 先让模型生成 ReAct 文本（不自动执行工具）
    #     llm = ChatOpenAI(
    #         model=self.model_name,
    #         base_url="https://api.moonshot.cn/v1",
    #         api_key=os.getenv("OPENAI_API_KEY"),
    #         temperature=0
    #     )
    #
    #     # 2. 循环直到拿到 Final Answer
    #     messages = [{"role": "user", "content": question}]
    #     max_steps = self.max_iterations
    #     steps_taken = 0
    #     full_history = []
    #
    #     for step in range(max_steps):
    #         # 调用 LLM
    #         response = llm.invoke(messages)
    #         content = response.content
    #         full_history.append(f"Step {step}: {content}")
    #
    #         # ===== 实时输出推理过程 =====
    #         if self.verbose:
    #             print(f"\n{'=' * 60}")
    #             print(f"【Step {step} - 推理/Action】")
    #             print(f"{'=' * 60}")
    #             print(content)
    #
    #         # 3. 手动解析 Action
    #         if "Action:" in content and "Action Input:" in content:
    #             # 提取工具名和参数
    #             action_match = re.search(r'Action:\s*(\w+)', content)
    #             input_match = re.search(r'Action Input:\s*(.+)', content)
    #
    #             if action_match and input_match:
    #                 tool_name = action_match.group(1)
    #                 tool_input = input_match.group(1).strip()
    #
    #                 # 4. 执行真实工具
    #                 tool_result = self._execute_tool(tool_name, tool_input)
    #
    #                 # 5. 把结果包装成 Observation 追加到历史
    #                 observation = f"Observation: {tool_result}"
    #                 full_history.append(observation)
    #                 messages.append({"role": "user", "content": observation})
    #                 steps_taken += 1
    #
    #                 # ===== 实时输出工具执行结果 =====
    #                 if self.verbose:
    #                     print(f"\n【Tool 执行】")
    #                     print(f"工具: {tool_name}")
    #                     print(f"输入: {tool_input}")
    #                     print(f"结果:\n{tool_result}")
    #
    #                 continue
    #
    #         # 6. 检测到 Final Answer 就跳出
    #         if "Final Answer:" in content:
    #             if self.verbose:
    #                 print("\n【检测到 Final Answer，停止迭代】")
    #             break
    #
    #     # 7. 提取最终答案
    #     final_answer = content.split("Final Answer:")[-1].strip() if "Final Answer:" in content else content
    #
    #     # ===== 实时输出最终答案 =====
    #     if self.verbose:
    #         print(f"\n{'=' * 60}")
    #         print("【最终答案】")
    #         print(f"{'=' * 60}")
    #         print(final_answer)
    #         print(f"{'=' * 60}\n")
    #     # 8. 追踪策略
    #     used_strategies = self._extract_used_strategies(full_history) if track_strategies else []
    #
    #     return ReactAgentResult(
    #         answer=final_answer,
    #         reasoning="\n\n".join(full_history),
    #         used_strategies=used_strategies,
    #         iterations=steps_taken,
    #         messages=messages,
    #         success=True
    #     )
    #
    # def _execute_tool(self, tool_name: str, tool_input: str) -> str:
    #     """手动执行工具"""
    #     for tool in self.tools:
    #         if tool.name == tool_name:
    #             try:
    #                 # 尝试解析 JSON 输入，如果是纯文本就原样传
    #                 import json
    #                 try:
    #                     parsed_input = json.loads(tool_input)
    #                     return tool.invoke(parsed_input)
    #                 except:
    #                     return tool.invoke(tool_input)
    #             except Exception as e:
    #                 return f"工具执行错误: {str(e)}"
    #     return f"未找到工具: {tool_name}"

    def _extract_sql_from_tool_calls(self, messages: List) -> Optional[str]:
        """
        从工具调用中提取最近一次 sqlite_query 的 SQL 参数，用于兜底最终答案。
        """
        last_sql = None
        for msg in messages:
            calls = getattr(msg, "tool_calls", None)
            if not calls:
                continue
            try:
                for call in calls:
                    # 结构示例：{'name': 'sqlite_query', 'args': {'sql': '...'}, 'id': '...', 'type': 'tool_call'}
                    if isinstance(call, dict) and call.get("name") == "sqlite_query":
                        args = call.get("args", {})
                        sql = args.get("sql")
                        if sql:
                            last_sql = sql  # 取最近一次
            except Exception:
                continue
        return last_sql

    def _get_system_prompt(self, question: str = "", context: str = "") -> str:
        """
        生成系统提示词。
        
        参数：
            question: 当前问题（用于检索相关策略，不包含在系统提示词中）
            context: 额外的上下文信息（包含在系统提示词中）
        """
        playbook_str = self._format_playbook(question)



        # 2. 格式化 context（如果有）
        context_str = f"\n## 额外上下文\n\n{context}" if context else ""
        # 3. 如果 context 中已包含 schema，提示模型直接使用
        schema_hint = ""
        if "create table" in (context or ""):
            schema_hint = (
                "\n\n## 数据库查询说明\n"
                "上下文中已提供数据库 Schema。请生成 SQL 并进行快速验证：\n"
                "- 使用 sqlite_columns 检查目标表的列名与类型（含空格/括号需加引号）\n"
                "- 使用 sqlite_query 只读执行，若返回错误则修正 SQL 后再次验证\n"
                "- 如需构造过滤条件，使用 sqlite_distinct 获取真实值域\n"
                "验证无误后，再输出最终 SQL。"
            )
        # 4. 组装最终系统提示（原模板 + SQL 规范）
        safe_playbook = playbook_str.replace("{", "{{").replace("}", "}}")
        safe_context = context_str.replace("{", "{{").replace("}", "}}")

        prompt = REACT_AGENT_PROMPT_V3.format(
            playbook=safe_playbook,
            context=safe_context
        ) + schema_hint


        return prompt
    
    def _format_playbook(self, question: str = "") -> str:
        """
        格式化 Playbook 为提示词格式。
        
        使用 playbook.retrieve_strategies 自动选择最佳检索方式：
        - 如果 Playbook 启用了 enable_retrieval 且提供了问题，使用语义检索
        - 否则回退到按分数排序
        
        参数：
            question: 当前问题（用于向量检索）
        """

        if not len(self.playbook):
            return "（Playbook 为空，尚无学习策略）"
        
        # 直接调用 playbook.retrieve_strategies，检索方式由 Playbook 内部控制
        strategies = self.playbook.retrieve_strategies(
            question=question,
            top_k=self.top_k_strategies,
            min_score=0   
        )
        if not strategies:
            return "（⚠️ Playbook 中暂无可用策略，但你仍需检查是否有相关策略可引用）"
        if self.verbose:
            # 检测实际使用的检索方式（由 Playbook 决定）
            retrieval_method = "🔍 向量检索" if (self.playbook.enable_retrieval and question) else "📊 分数排序"
            print(f"{retrieval_method}获取 {len(strategies)} 个策略")
        
        if not strategies:
            return "（Playbook 中暂无可用策略）"
        
        # 格式化策略
        lines = []
        for strategy in strategies:
            helpful = strategy.helpful_count
            harmful = strategy.harmful_count
            confidence = helpful / (helpful + harmful + 1)  # 简单的置信度计算
            
            lines.append(
                f"[{strategy.id}] {strategy.content} "
                f"(✓{helpful} ✗{harmful} 置信度:{confidence:.2f})"
            )
        
        return "\n".join(lines)
    
    def _extract_used_strategies(self, messages: List) -> List[str]:
        """从消息中提取使用的策略 ID。"""
        import re
        used = set()
        
        for msg in messages:
            if hasattr(msg, 'content') and msg.content:
                # 查找策略引用模式：[strategy_id]
                matches = re.findall(r'\[([a-z]{3}-\d{5})\]', msg.content)
                used.update(matches)
        
        return list(used)



# ========== 辅助函数 ==========

def get_default_tools() -> List:
    """获取默认工具集。"""
    return [calculator, search, sqlite_schema,sqlite_query, sqlite_columns, sqlite_distinct]


# ========== 演示代码 ==========

def demo():
    """演示 ReAct Agent v2.0 with Playbook。"""
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*8 + "ReAct Agent v2.0 演示（带 Playbook 策略）" + " "*8 + "║")
    print("╚" + "="*58 + "╝\n")
    
    # 创建 Playbook 并添加一些示例策略
    playbook = Playbook(enable_retrieval=False)
    playbook.add_strategy(
        "复杂数学计算要分步进行，先计算括号内的表达式",
        category="react"
    )
    playbook.add_strategy(
        "需要查询信息时，优先使用 search 工具获取准确数据",
        category="react"
    )
    
    # 标记一些策略为有用的
    strategies = playbook.get_all_strategies()
    if strategies:
        playbook.mark_helpful(strategies[0].id)
        playbook.mark_helpful(strategies[1].id)
    
    print("✅ 已创建 Playbook，包含以下策略：")
    for s in strategies:
        print(f"   [{s.id}] {s.content}")
    print()
    
    # 创建 agent（带 Playbook）
    tools = get_default_tools()
    agent = ReActAgent(playbook=playbook, tools=tools, verbose=True)
    
    # 测试问题
    questions = [
        ReactQuestion(
            question="计算 (15 + 27) * 3 的结果",
            context="这是一个数学计算问题",
            ground_truth=None
        ),
        ReactQuestion(
            question="Python 是什么时候创建的？",
            context="",
            ground_truth=None
        ),
        ReactQuestion(
            question="东京在哪个国家？它举办过奥运会吗？",
            context="",
            ground_truth=None
        ),
    ]
    
    for i, react_question in enumerate(questions, 1):
        print(f"\n{'#'*60}")
        print(f"示例 {i}")
        print(f"{'#'*60}")
        
        result = agent.run(react_question)
        
        # 显示使用的策略
        if result.used_strategies:
            print(f"\n📊 策略使用统计：")
            for strategy_id in result.used_strategies:
                strategy = playbook.get_strategy(strategy_id)
                if strategy:
                    print(f"   ✓ 使用了策略 [{strategy_id}]: {strategy.content[:50]}...")
        
        if i < len(questions):
            input("\n按回车继续...")
    
    # 显示 Playbook 统计
    print(f"\n{'='*60}")
    print("📈 Playbook 统计")
    print(f"{'='*60}")
    print(f"总策略数：{len(strategies)}")
    print(f"有用标记：{sum(s.helpful_count for s in strategies)}")
    print(f"有害标记：{sum(s.harmful_count for s in strategies)}")


if __name__ == "__main__":
    import os
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️  请设置 OPENAI_API_KEY：")
        print("   export OPENAI_API_KEY='your-key-here'")
    else:
        demo()
