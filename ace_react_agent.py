"""
ACE + ReAct Agent - 使用 ACE 范式提升 ReAct Agent 的性能（简化版）

使用简化的 ReAct Agent（基于 LangGraph 内置组件）与 ACE 框架整合。

核心思路：
1. ReAct Agent 作为 Generator（使用 ToolNode 和 tools_condition）
2. Evaluator 评估 Agent 的答案是否正确
3. Reflector 分析 Agent 的推理过程
4. Curator 将模式存储为策略，供后续使用
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
import sys
from pathlib import Path



# 统一使用 ace_langgraph 前缀导入
from  playbook import Playbook
from  react_agent import ReActAgent, get_default_tools
from  reflector import Reflector
from  curator import Curator
from  evaluator import Evaluator
from  agent_types import (
    ReactQuestion, ReactAgentResult, EvaluatorResult,
    ReflectionResult, CuratorResult
)
    
from typing_extensions import TypedDict


class ACEReActState(TypedDict):
    """ACE + ReAct 工作流状态。
    
    所有数据都以类型对象形式存储，避免冗余字段。
    需要的信息都可以从各个 Result 对象中获取。
    """
    # 输入对象
    react_question: ReactQuestion
    
    # 各组件输出对象
    react_result: ReactAgentResult
    evaluation: EvaluatorResult
    reflection: ReflectionResult
    curator_result: CuratorResult  # 新增：保存 curator 结果

class ACEReActWorkflow:
    """
    ACE + ReAct 工作流（简化版）。
    
    将简化的 ReAct Agent 集成到 ACE 框架中。
    自动管理 Playbook 持久化（加载/保存）。
    """
    
    # 默认的 Playbook 文件路径
    DEFAULT_PLAYBOOK_PATH = "ace_react_playbook.json"
    
    def __init__(
        self,
        tools: List = None,
        model_name: str = "qwen-plus",
        max_iterations: int = 50, #暂时没用
        use_vector_retrieval: bool = True,
        playbook_path: str = None,
        auto_save: bool = True
    ):
        """
        初始化 ACE + ReAct 工作流。
        
        参数：
            tools: ReAct Agent 可用的工具列表
            model_name: LLM 模型名称
            max_iterations: Agent 最大迭代次数
            use_vector_retrieval: 是否使用向量检索（推荐）
            playbook_path: Playbook 文件路径（默认：ace_react_playbook.json）
                          设置为 None 可创建不保存的临时 Playbook
            auto_save: 是否在每次运行后自动保存 Playbook
        """
        self.tools = tools or get_default_tools()
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.use_vector_retrieval = use_vector_retrieval
        self.playbook_path = playbook_path if playbook_path is not None else self.DEFAULT_PLAYBOOK_PATH
        self.auto_save = auto_save
        
        # 自动加载或创建 Playbook
        self.playbook = self._load_or_create_playbook()
        
        # 创建 ReAct Agent（会引用 self.playbook，能获取最新策略）
        self.agent = ReActAgent(
            playbook=self.playbook,  # 传递引用，agent 会访问最新的 playbook
            tools=self.tools,
            model_name=self.model_name,
            max_iterations=self.max_iterations,
            verbose=False,
            top_k_strategies=5
        )

        self.evaluator = Evaluator(model_name=model_name)
        self.reflector = Reflector(playbook=self.playbook, model_name=model_name)
        self.curator = Curator(playbook=self.playbook, model_name=model_name)
        
        # 构建工作流图（使用条件边）
        self.graph = self._build_graph()
    
    def _load_or_create_playbook(self) -> Playbook:
        """
        加载或创建 Playbook。
        
        返回：
            Playbook 实例
        """
        from pathlib import Path
        
        # 如果指定了路径且文件存在，尝试加载
        if self.playbook_path and Path(self.playbook_path).exists():
            try:
                print(f"📂 从 {self.playbook_path} 加载已有的 Playbook...")
                playbook = Playbook.load_from_file(self.playbook_path,enable_retrieval=self.use_vector_retrieval)
                stats = playbook.stats()
                print(f"   ✓ 成功加载 {stats['total_strategies']} 个策略")
                return playbook
            except Exception as e:
                print(f"   ⚠️  加载失败: {e}")
                print(f"   创建新的 Playbook")
        elif self.playbook_path:
            print(f"📝 未找到 {self.playbook_path}，创建新的 Playbook")
        else:
            print(f"📝 创建临时 Playbook（不会保存到文件）")
        
        # 创建新 Playbook（根据配置选择类型）
        return Playbook(
            enable_retrieval=self.use_vector_retrieval
        )
    
    def _save_playbook(self):
        """
        保存 Playbook 到文件。
        """
        if not self.playbook_path:
            # 没有指定路径，不保存（临时 Playbook）
            return
            
        try:
            self.playbook.save_to_file(self.playbook_path)
            print(f"💾 已自动保存 Playbook 到 {self.playbook_path}")
        except Exception as e:
            print(f"⚠️  保存 Playbook 失败: {e}")
    
    def _should_evaluate(self, state: ACEReActState) -> str:
        """
        条件函数：判断是否需要评估。
        
        根据 ground_truth 是否存在决定路由：
        - 有 ground_truth → "evaluate" (训练模式)
        - 无 ground_truth → "skip_evaluate" (生产模式)
        """
        react_question = state.get("react_question")
        has_ground_truth = react_question and react_question.ground_truth
        return "evaluate" if has_ground_truth else "skip_evaluate"
    
    def _build_graph(self) -> StateGraph:
        """
        构建 LangGraph 工作流（使用条件边）。
        
        工作流结构：
        - react_agent → 条件判断
          - 有 ground_truth → evaluator → reflector → curator
          - 无 ground_truth → reflector → curator
        """
        workflow = StateGraph(ACEReActState)
        
        # 添加所有节点
        workflow.add_node("react_agent", self._react_agent_node)
        workflow.add_node("evaluator", self._evaluator_node)
        workflow.add_node("reflector", self._reflector_node)
        workflow.add_node("curator", self._curator_node)
        
        # 设置入口
        workflow.set_entry_point("react_agent")
        
        # 使用条件边：react_agent 后根据是否有 ground_truth 决定路由
        workflow.add_conditional_edges(
            "react_agent",
            lambda state: "evaluator" if state["react_question"].ground_truth else "reflector",
            {"evaluator": "evaluator", "reflector": "reflector"}
        )
        
        # evaluator 后续路径
        workflow.add_edge("evaluator", "reflector")
        
        # reflector 和 curator 的路径
        workflow.add_edge("reflector", "curator")
        workflow.add_edge("curator", END)
        
        return workflow.compile()
    
    def _react_agent_node(self, state: ACEReActState) -> ACEReActState:
        """
        ReAct Agent 节点 - 作为 Generator。
        
        使用初始化时创建的 Agent 实例。
        Agent 内部会动态获取最新的 playbook 策略（通过引用）。
        """
        # 直接使用 state 中的 ReactQuestion 对象
        react_question = state["react_question"]
        
        # 调用 agent，直接返回 ReactAgentResult 对象
        react_result = self.agent.run(react_question)
        
        # 保存 ReactAgentResult 对象到 state
        state["react_result"] = react_result
        
        return state
    
    def _evaluator_node(self, state: ACEReActState) -> ACEReActState:
        """评估器节点 - 使用 LLM 检查答案正确性。"""
        # 直接使用 state 中的类型对象
        react_question = state["react_question"]
        react_result = state["react_result"]
        
        # 调用新接口
        evaluation = self.evaluator.evaluate(react_question, react_result)
        state["evaluation"] = evaluation
        return state
    
    def _reflector_node(self, state: ACEReActState) -> ACEReActState:
        """
        反思器节点 - 分析推理过程。
        
        使用 Reflector 模块分析完整的消息历史，找出成功/失败模式。
        支持训练模式（有评估结果）和生产模式（无评估结果）。
        """
        # 直接使用 state 中的类型对象
        react_question = state["react_question"]
        react_result = state["react_result"]
        evaluator_result = state.get("evaluation")
        
        # 调用新接口
        reflection_result = self.reflector.reflect(
            react_question,
            react_result,
            evaluator_result
        )
        
        # 只保存结果对象
        state["reflection"] = reflection_result
        
        return state
    
    def _curator_node(self, state: ACEReActState) -> ACEReActState:
        """
        策展器节点 - 提取新策略。
        
        使用 Curator 模块从反思中提取可复用的策略模式。
        支持训练模式（有评估结果）和生产模式（无评估结果）。
        """
        # 直接使用 state 中的类型对象
        react_question = state["react_question"]
        react_result = state["react_result"]
        evaluator_result = state.get("evaluation")
        reflection_result = state["reflection"]
        
        # 调用新接口（playbook 已在 curator 初始化时传入）
        curation_result = self.curator.curate(
            react_question,
            react_result,
            evaluator_result,
            reflection_result
        )
        
        # 只保存结果对象
        state["curator_result"] = curation_result
        
        return state
    
    def run(
        self,
        question: ReactQuestion,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        运行 ACE + ReAct 工作流。
        
        根据是否提供 ground_truth 自动选择训练模式或生产模式：
        - 训练模式（有 ground_truth）：包含评估节点
        - 生产模式（无 ground_truth）：跳过评估节点
        
        使用条件边自动路由，无需手动判断。
        
        参数：
            question: ReactQuestion 对象
            verbose: 是否打印详细信息
            
        返回：包含答案、评估、反思等信息的字典
        """
        # 初始化状态（所有字段都是类型对象）
        initial_state: ACEReActState = {
            "react_question": question,
            "react_result": None,
            "evaluation": None,
            "reflection": None,
            "curator_result": None
        }
        
        if verbose:
            has_ground_truth = question.ground_truth is not None
            mode = "训练模式" if has_ground_truth else "生产模式"
            print(f"\n{'='*60}")
            print(f"问题：{question.question}")
            print(f"模式：{mode}")
            print(f"{'='*60}\n")
        
        # 执行工作流（条件边会自动处理路由）
        result = self.graph.invoke(initial_state)
        
        # 自动保存 Playbook（如果启用）
        if self.auto_save:
            self._save_playbook()
        
        if verbose:
            print("\n" + "="*60)
            print("执行结果")
            print("="*60)
            
            # 显示 ReAct Agent 结果
            react_result = result.get("react_result")
            if react_result:
                print("\n【ReAct Agent】")
                print(react_result.as_str())
            
            # 显示评估结果
            evaluation = result.get("evaluation")
            if evaluation:
                print("\n【评估结果】")
                print(evaluation.as_str())
            else:
                print("\n【评估结果】")
                print("(生产模式，未评估)")
            
            # 显示反思结果
            reflection_result = result.get('reflection')
            if reflection_result:
                print("\n【反思结果】")
                print(reflection_result.as_str())
            else:
                print("\n【反思结果】")
                print("(未生成)")
            
            # 显示策展结果
            curator_result = result.get('curator_result')
            if curator_result:
                print("\n【策展结果】")
                print(curator_result.as_str())
            else:
                print("\n【策展结果】")
                print("(未生成)")
            
            print(f"\n【Playbook】")
            print(f"总策略数：{len(self.playbook)}")
            print()
        
        return result
    
    def ask(
        self,
        question: ReactQuestion,
        verbose: bool = True
    ) -> ReactAgentResult:
        """
        快速询问模式 - 仅执行 ReAct Agent，不进行评估、反思和学习。
        
        适用场景：
        - 快速获取答案，不需要学习
        - 生产环境直接使用
        - 测试 Agent 性能
        
        参数：
            question: ReactQuestion 对象（可以不提供 ground_truth）
            verbose: 是否打印详细信息
            
        返回：ReactAgentResult 对象
        
        示例：
            >>> workflow = ACEReActWorkflow()
            >>> result = workflow.ask(ReactQuestion("2+2等于多少？"))
            >>> print(result.answer)
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"快速询问模式")
            print(f"问题：{question.question}")
            print(f"{'='*60}\n")
        
        # 只执行 react_agent 节点
        initial_state: ACEReActState = {
            "react_question": question,
            "react_result": None,
            "evaluation": None,
            "reflection": None,
            "curator_result": None
        }
        
        # 直接调用 react_agent 节点
        state = self._react_agent_node(initial_state)
        react_result = state["react_result"]
        
        if verbose:
            print("\n" + "="*60)
            print("执行结果")
            print("="*60)
            
            if react_result:
                print("\n【ReAct Agent】")
                print(react_result.as_str())
            
            print()
        
        return react_result

# ========== 主函数测试 ==========

def main():
    """测试 ACE + ReAct 工作流。"""
    import os
    
    # 检查 API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  请设置 OPENAI_API_KEY 环境变量")
        return
    
    print("\n" + "="*60)
    print("ACE + ReAct Agent 测试（简化版）")
    print("="*60)
    
    # 1. 创建工作流
    workflow = ACEReActWorkflow(
        tools=get_default_tools(),
        model_name="gpt-4o-mini",
        max_iterations=10,
        use_vector_retrieval=True
    )

    # 2. 训练问题
    questions = [
        ReactQuestion(
            question="计算 (25 + 17) * 3 - 8 的结果，并验证答案是否为偶数",
            ground_truth="118，是偶数",
            context=""
        ),
        ReactQuestion(
            question="搜索 Python 语言的创建者，并说明他创建 Python 的年份",
            ground_truth="Guido van Rossum，1991年",
            context=""
        ),
        ReactQuestion(
            question="搜索世界上最高的山峰名称和海拔高度，然后计算如果一个人每天爬升500米，需要多少天才能到达顶峰",
            ground_truth="珠穆朗玛峰，8849米，需要约18天",
            context=""
        ),
    ]
    
    # 3. 运行训练
    print("\n开始训练阶段...\n")

    for i, question in enumerate(questions, 1):
        print(f"\n{'='*60}")
        print(f"问题 {i}/{len(questions)}")
        print(f"{'='*60}")
        workflow.run(question, verbose=True)
    
    # 4. 查看学到的策略
    print("\n" + "="*60)
    print("学习成果")
    print("="*60)
    
    stats = workflow.playbook.stats()
    print(f"\n策略总数：{stats['total_strategies']}")
    print(f"分类数：{stats['categories']}")
    print(f"标记统计：✓{stats['tags']['helpful']} / ✗{stats['tags']['harmful']} / ~{stats['tags']['neutral']}")
    print(f"平均得分：{stats['avg_score']:.2f}")
    
    if len(workflow.playbook) > 0:
        print("\n前 5 个策略（按分数排序）：")
        top_strategies = workflow.playbook.get_top_strategies(n=5)
        
        for i, strategy in enumerate(top_strategies, 1):
            print(f"\n{i}. [{strategy.id}] (分数: {strategy.score})")
            print(f"   {strategy.content}")
            print(f"   ✓{strategy.helpful_count} / ✗{strategy.harmful_count} / ~{strategy.neutral_count}")
            print(f"   创建于: {strategy.created_at[:10]}")
        
    # 5. 测试应用（复杂问题 - 综合运用训练的能力）
    print("\n" + "="*60)
    print("测试阶段 - 复杂问题测试")
    print("="*60 + "\n")
    
    test_question = ReactQuestion(
        question="搜索世界上最深的海沟名称和深度，然后计算如果一个潜水器每小时下潜1000米，需要多少小时才能到达海沟最深处。最后验证计算结果是否为整数小时数。",
        context=""
    )
    print(f"问题：{test_question.question}")
    print("模式：综合测试（搜索 + 计算 + 验证）\n")
    
    result = workflow.run(test_question, verbose=True)
    
    print("\n💡 此问题测试了训练中学到的多个能力：")
    print("   ✓ 信息搜索能力")
    print("   ✓ 多步计算能力")
    print("   ✓ 数值验证能力")
    # 从 curator_result 获取新策略数量
    curator_result = result.get('curator_result')
    new_strategies_count = curator_result.added_count if curator_result else 0
    print(f"   📚 本次新增 {new_strategies_count} 个策略")
    

if __name__ == "__main__":
    import sys
    main()
