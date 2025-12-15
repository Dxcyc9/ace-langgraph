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
project_root = Path(__file__).parent.parent
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
def sqlite_schema(db_path: str, sample_rows: int = 3) -> str:
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
    import sqlite3, csv, io, textwrap

    if not os.path.isfile(db_path):
        return f"文件不存在：{db_path}"

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.text_factory = str
        cur = conn.cursor()

        tables = [t[0] for t in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()]
        if not tables:
            return "数据库中无用户表。"

        buf = io.StringIO()
        # 1) 输出 schema
        for tbl in tables:
            buf.write(f"-- Table: {tbl}\n")
            create_sql = cur.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (tbl,)).fetchone()[0]
            buf.write(create_sql + ";\n\n")

            # 2) 抽样数据
            if sample_rows > 0:
                rows = cur.execute(f"SELECT * FROM `{tbl}` LIMIT ?", (sample_rows,)).fetchall()
                if not rows:
                    buf.write("-- (empty)\n\n")
                    continue
                # 转 CSV 风格，避免值里有换行
                buf.write("-- Sample data (CSV format):\n")
                output = io.StringIO()
                writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.MINIMAL)
                writer.writerow([d[0] for d in cur.description])  # header
                writer.writerows(rows)
                buf.write(output.getvalue() + "\n")

        conn.close()
        return buf.getvalue()

    except Exception as e:
        return f"读取 sqlite 失败：{str(e)}"

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

        参数：
            react_question: 输入问题（ReactQuestion 对象）
            track_strategies: 是否追踪使用的策略

        返回：
            ReactAgentResult 对象
        """
        question = react_question.question
        context = react_question.context

        # 保存当前问题（用于检索策略）
        self.current_question = question

        # 动态创建 agent（使用当前问题检索相关策略）
        agent = self._get_or_create_agent(question, context)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"问题：{question}")
            if context:
                print(f"上下文：{context}")
            print(f"{'='*60}\n")

        # 调用 agent
        result = agent.invoke({
            "messages": [{"role": "user", "content": question}]
        })
        messages = result["messages"]
        final_message = messages[-1].content if messages else ""

        # === 解析策略行（与可选提示词兼容） ===
        used_strategies = []
        lines = final_message.strip().split('\n')

        # 检查第一行是否是策略引用
        if lines and "Strategy:" in lines[0]:
            # 提取策略ID
            import re
            matches = re.findall(r'Strategy: \[([a-z]{3}-\d{5})\]', lines[0])
            used_strategies = matches

            # SQL 内容从第二行开始
            sql_lines = lines[1:] if len(lines) > 1 else []
            final_answer = '\n'.join(sql_lines).strip()

            # 仅在策略引用格式错误时警告（不是缺失）
            if not matches and track_strategies:
                print("⚠️ 策略引用格式错误，应为: Strategy: [sql-xxxxx]")
        else:
            # Agent 未引用策略（符合新的可选规则）
            final_answer = final_message.strip()

            # 仅在 Playbook 非空时提示（可关闭）
            if track_strategies and len(self.playbook) > 0:
                print("ℹ️ Agent 未使用策略（直接生成SQL）")


        # === 调试：看所有 AI 消息的 tool_calls ===
        for idx, m in enumerate(result["messages"]):
            if hasattr(m, "type") and m.type == "ai":
                calls = getattr(m, "tool_calls", None)
                if calls:
                    print(f"【步骤 {idx} 调用了工具】", calls)
                    break  # 找到第一个就停
        else:
            print("【全程未调用任何工具】")

        # ==========================================
        # print("原始消息:", result["messages"][-1])
        # 提取答案和推理过程
        messages = result["messages"]
        final_message = messages[-1].content if messages else "未能生成答案"
        # 提取 Final Answer
        if "Final Answer:" in final_message:
            final_answer = final_message.split("Final Answer:")[-1].strip()
        else:
            final_answer = final_message

        # 提取完整推理过程（所有 AI 消息的拼接）
        # 提取推理过程：遍历所有AI消息，构建编号的推理步骤
        reasoning_steps = []
        step_num = 1

        for msg in messages:
            if isinstance(msg, AIMessage) and hasattr(msg, 'content'):
                reasoning_steps.append(f"\n{msg.content.strip()}")
                step_num += 1
        reasoning = "\n\n".join(reasoning_steps) if reasoning_steps else "未生成推理过程"

        # 追踪使用的策略
        used_strategies = []
        if track_strategies:
            used_strategies = self._extract_used_strategies(messages)

        if self.verbose:
            print(f"\n【最终答案】\n{final_answer}")
            print(f"\n【推理过程】\n{reasoning}")
            if used_strategies:
                print(f"\n【使用的策略】\n{', '.join(used_strategies)}")
            print(f"\n【迭代次数】\n{len([m for m in messages if hasattr(m, 'tool_calls') and m.tool_calls])}")
            print()

        return ReactAgentResult(
            answer=final_answer,
            reasoning=final_message,
            used_strategies=used_strategies,
            iterations=len([m for m in messages if hasattr(m, 'tool_calls') and m.tool_calls]),
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
        if "CREATE TABLE" in (context or ""):
            schema_hint = "\n\n## 数据库查询说明\n上下文中已提供数据库 Schema，请直接生成 SQL，不要调用任何工具。"
        # 4. 组装最终系统提示（原模板 + SQL 规范）
        prompt = REACT_AGENT_PROMPT_V3.format(
            playbook=playbook_str,
            context=context_str
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
    return [calculator, search, sqlite_schema]


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
