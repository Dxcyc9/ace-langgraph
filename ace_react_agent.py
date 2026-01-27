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
import os



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

# ========== 数据库 Schema 常量（用于训练阶段）==========
SCHEMA_CONTEXT = """
## 数据库 Schema (California Schools)
以下四张表已加载，可直接使用：

### Table: frpm
```sql
create table frpm
(
    CDSCode                                       TEXT not null
        primary key
        references schools,
    "Academic Year"                               TEXT,
    "County Code"                                 TEXT,
    "District Code"                               INTEGER,
    "School Code"                                 TEXT,
    "County Name"                                 TEXT,
    "District Name"                               TEXT,
    "School Name"                                 TEXT,
    "District Type"                               TEXT,
    "School Type"                                 TEXT,
    "Educational Option Type"                     TEXT,
    "NSLP Provision Status"                       TEXT,
    "Charter School (Y/N)"                        INTEGER,
    "Charter School Number"                       TEXT,
    "Charter Funding Type"                        TEXT,
    IRC                                           INTEGER,
    "Low Grade"                                   TEXT,
    "High Grade"                                  TEXT,
    "Enrollment (K-12)"                           REAL,
    "Free Meal Count (K-12)"                      REAL,
    "Percent (%) Eligible Free (K-12)"            REAL,
    "FRPM Count (K-12)"                           REAL,
    "Percent (%) Eligible FRPM (K-12)"            REAL,
    "Enrollment (Ages 5-17)"                      REAL,
    "Free Meal Count (Ages 5-17)"                 REAL,
    "Percent (%) Eligible Free (Ages 5-17)"       REAL,
    "FRPM Count (Ages 5-17)"                      REAL,
    "Percent (%) Eligible FRPM (Ages 5-17)"       REAL,
    "2013-14 CALPADS Fall 1 Certification Status" INTEGER
);


### Table: satscores
```sql
create table satscores
(
    cds         TEXT    not null
        primary key
        references schools,
    rtype       TEXT    not null,
    sname       TEXT,
    dname       TEXT,
    cname       TEXT,
    enroll12    INTEGER not null,
    NumTstTakr  INTEGER not null,
    AvgScrRead  INTEGER,
    AvgScrMath  INTEGER,
    AvgScrWrite INTEGER,
    NumGE1500   INTEGER
);


### Table: schools
```sql
create table schools
(
    CDSCode     TEXT not null
        primary key,
    NCESDist    TEXT,
    NCESSchool  TEXT,
    StatusType  TEXT not null,
    County      TEXT not null,
    District    TEXT not null,
    School      TEXT,
    Street      TEXT,
    StreetAbr   TEXT,
    City        TEXT,
    Zip         TEXT,
    State       TEXT,
    MailStreet  TEXT,
    MailStrAbr  TEXT,
    MailCity    TEXT,
    MailZip     TEXT,
    MailState   TEXT,
    Phone       TEXT,
    Ext         TEXT,
    Website     TEXT,
    OpenDate    DATE,
    ClosedDate  DATE,
    Charter     INTEGER,
    CharterNum  TEXT,
    FundingType TEXT,
    DOC         TEXT not null,
    DOCType     TEXT not null,
    SOC         TEXT,
    SOCType     TEXT,
    EdOpsCode   TEXT,
    EdOpsName   TEXT,
    EILCode     TEXT,
    EILName     TEXT,
    GSoffered   TEXT,
    GSserved    TEXT,
    Virtual     TEXT,
    Magnet      INTEGER,
    Latitude    REAL,
    Longitude   REAL,
    AdmFName1   TEXT,
    AdmLName1   TEXT,
    AdmEmail1   TEXT,
    AdmFName2   TEXT,
    AdmLName2   TEXT,
    AdmEmail2   TEXT,
    AdmFName3   TEXT,
    AdmLName3   TEXT,
    AdmEmail3   TEXT,
    LastUpdate  DATE not null
);



### Table: sqlite_master
```sql
create table sqlite_master
(
    type     TEXT,
    name     TEXT,
    tbl_name TEXT,
    rootpage INT,
    sql      TEXT
);
"""

class ACEReActState(TypedDict):
    """ACE + ReAct 工作流状态。"""
    react_question: ReactQuestion
    react_result: ReactAgentResult
    evaluation: EvaluatorResult
    reflection: ReflectionResult
    curator_result: CuratorResult

class ACEReActWorkflow:
    """
    ACE + ReAct 工作流（简化版）。

    将简化的 ReAct Agent 集成到 ACE 框架中。
    自动管理 Playbook 持久化（加载/保存）。
    """
    DEFAULT_PLAYBOOK_PATH = "ace_react_playbook.json"

    def __init__(
        self,
        tools: List = None,
        model_name: str = "qwen-plus",
        max_iterations: int = 50,
        use_vector_retrieval: bool = True,
        playbook_path: str = None,
        auto_save: bool = True
    ):
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
            playbook=self.playbook,
            tools=self.tools,
            model_name=self.model_name,
            max_iterations=self.max_iterations,
            verbose=False,
            top_k_strategies=5
        )
        self.evaluator = Evaluator(model_name=model_name)
        self.reflector = Reflector(playbook=self.playbook, model_name=model_name)
        self.curator = Curator(playbook=self.playbook, model_name=model_name)

        self.graph = self._build_graph()

    def _load_or_create_playbook(self) -> Playbook:
        """
        加载或创建 Playbook。
        """
        from pathlib import Path

        # 选取嵌入模型（可通过环境变量覆盖；默认多语言）
        embedding_model = os.getenv("ACE_EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

        if self.playbook_path and Path(self.playbook_path).exists():
            try:
                print(f"📂 从 {self.playbook_path} 加载已有的 Playbook...")
                playbook = Playbook.load_from_file(self.playbook_path, enable_retrieval=self.use_vector_retrieval)
                # 如果需要切换嵌入模型，重建索引
                if playbook.enable_retrieval and playbook.embedding_model != embedding_model:
                    print(f"🔁 切换嵌入模型为 {embedding_model} 并重建索引")
                    playbook.embedding_model = embedding_model
                    playbook.rebuild_index()
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

        return Playbook(
            enable_retrieval=self.use_vector_retrieval,
            embedding_model=embedding_model
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
        print(f"   标准答案: {react_question.ground_truth}")
        print(f"   问题回答: {react_result.answer}")
        
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

            # 执行工作流（条件边会自动处理路由），并设置递归上限与兜底降级
            try:
                result = self.graph.invoke(initial_state, config={"recursion_limit": 40})
            except Exception as e:
                from langgraph.errors import GraphRecursionError
                if isinstance(e, GraphRecursionError):
                    print(f"⚠️ Workflow 递归上限触发，启用降级直线流程：{e}")
                    # 手工串行执行各节点，避免整体中断
                    react_result = self.agent.run(question)
                    evaluation = self.evaluator.evaluate(question, react_result) if question.ground_truth else None
                    reflection = self.reflector.reflect(question, react_result, evaluation)
                    curator_result = self.curator.curate(question, react_result, evaluation, reflection)
                    result = {
                        "react_question": question,
                        "react_result": react_result,
                        "evaluation": evaluation,
                        "reflection": reflection,
                        "curator_result": curator_result,
                    }
                else:
                    raise
        
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
        model_name="kimi-k2-turbo-preview",
        max_iterations=10,
        use_vector_retrieval=True
    )

    # 2. 训练问题（train）
    questions = [
        ReactQuestion(
            question="What is the highest eligible free rate for K-12 students in the schools in Alameda County?",
            ground_truth="SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.",
            ground_truth="SELECT `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` FROM frpm WHERE `Educational Option Type` = 'Continuation School' AND `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` IS NOT NULL ORDER BY `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)` ASC LIMIT 3",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="Please list the zip code of all the charter schools in Fresno County Office of Education.",
            ground_truth="SELECT T2.Zip FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`District Name` = 'Fresno County Office of Education' AND T1.`Charter School (Y/N)` = 1",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="What is the unabbreviated mailing street address of the school with the highest FRPM count for K-12 students?",
            ground_truth="SELECT T2.MailStreet FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode ORDER BY T1.`FRPM Count (K-12)` DESC LIMIT 1",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="Please list the phone numbers of the direct charter-funded schools that are opened after 2000/1/1.",
            ground_truth="SELECT T2.Phone FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`Charter Funding Type` = 'Directly funded' AND T1.`Charter School (Y/N)` = 1 AND T2.OpenDate > '2000-01-01'",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="How many schools with an average score in Math greater than 400 in the SAT test are exclusively virtual?",
            ground_truth="SELECT COUNT(DISTINCT T2.School) FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T2.Virtual = 'F' AND T1.AvgScrMath > 400",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="Among the schools with the SAT test takers of over 500, please list the schools that are magnet schools or offer a magnet program.",
            ground_truth="SELECT T2.School FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T2.Magnet = 1 AND T1.NumTstTakr > 500",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="What is the phone number of the school that has the highest number of test takers with an SAT score of over 1500?",
            ground_truth="SELECT T2.Phone FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode ORDER BY T1.NumGE1500 DESC LIMIT 1",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="What is the number of SAT test takers of the schools with the highest FRPM count for K-12 students?",
            ground_truth="SELECT NumTstTakr FROM satscores WHERE cds = ( SELECT CDSCode FROM frpm ORDER BY `FRPM Count (K-12)` DESC LIMIT 1 )",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="Among the schools with the average score in Math over 560 in the SAT test, how many schools are directly charter-funded?",
            ground_truth="SELECT COUNT(T2.`School Code`) FROM satscores AS T1 INNER JOIN frpm AS T2 ON T1.cds = T2.CDSCode WHERE T1.AvgScrMath > 560 AND T2.`Charter Funding Type` = 'Directly funded'",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="For the school with the highest average score in Reading in the SAT test, what is its FRPM count for students aged 5-17?",
            ground_truth="SELECT T2.`FRPM Count (Ages 5-17)` FROM satscores AS T1 INNER JOIN frpm AS T2 ON T1.cds = T2.CDSCode ORDER BY T1.AvgScrRead DESC LIMIT 1",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="Please list the codes of the schools with a total enrollment of over 500.",
            ground_truth="SELECT T2.CDSCode FROM schools AS T1 INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.`Enrollment (K-12)` + T2.`Enrollment (Ages 5-17)` > 500",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="Among the schools with an SAT excellence rate of over 0.3, what is the highest eligible free rate for students aged 5-17?",
            ground_truth="SELECT MAX(CAST(T1.`Free Meal Count (Ages 5-17)` AS REAL) / T1.`Enrollment (Ages 5-17)`) FROM frpm AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE CAST(T2.NumGE1500 AS REAL) / T2.NumTstTakr > 0.3",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="Please list the phone numbers of the schools with the top 3 SAT excellence rate.",
            ground_truth="SELECT T1.Phone FROM schools AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds ORDER BY CAST(T2.NumGE1500 AS REAL) / T2.NumTstTakr DESC LIMIT 3",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="List the top five schools, by descending order, from the highest to the lowest, the most number of Enrollment (Ages 5-17). Please give their NCES school identification number.",
            ground_truth="SELECT T1.NCESSchool FROM schools AS T1 INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode ORDER BY T2.`Enrollment (Ages 5-17)` DESC LIMIT 5",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="Which active district has the highest average score in Reading?",
            ground_truth="SELECT T1.District FROM schools AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T1.StatusType = 'Active' ORDER BY T2.AvgScrRead DESC LIMIT 1",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="How many schools in merged Alameda have number of test takers less than 100?",
            ground_truth="SELECT COUNT(T1.CDSCode) FROM schools AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T1.StatusType = 'Merged' AND T2.NumTstTakr < 100 AND T1.County = 'Lake'",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="Rank schools by their average score in Writing where the score is greater than 499, showing their charter numbers.",
            ground_truth="SELECT CharterNum, AvgScrWrite, RANK() OVER (ORDER BY AvgScrWrite DESC) AS WritingScoreRank FROM schools AS T1  INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T2.AvgScrWrite > 499 AND CharterNum is not null",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="How many schools in Fresno (directly funded) have number of test takers not more than 250?",
            ground_truth="SELECT COUNT(T1.CDSCode) FROM frpm AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds WHERE T1.`Charter Funding Type` = 'Directly funded' AND T1.`County Name` = 'Fresno' AND T2.NumTstTakr <= 250",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="What is the phone number of the school that has the highest average score in Math?",
            ground_truth="SELECT T1.Phone FROM schools AS T1 INNER JOIN satscores AS T2 ON T1.CDSCode = T2.cds ORDER BY T2.AvgScrMath DESC LIMIT 1",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="How many schools in Amador which the Low Grade is 9 and the High Grade is 12?",
            ground_truth="SELECT COUNT(T1.`School Name`) FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.County = 'Amador' AND T1.`Low Grade` = 9 AND T1.`High Grade` = 12",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="In Los Angeles how many schools have more than 500 free meals but less than 700 free or reduced price meals for K-12?",
            ground_truth="SELECT COUNT(CDSCode) FROM frpm WHERE `County Name` = 'Los Angeles' AND `Free Meal Count (K-12)` > 500 AND `FRPM Count (K-12)`< 700",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="Which school in Contra Costa has the highest number of test takers?",
            ground_truth="SELECT sname FROM satscores WHERE cname = 'Contra Costa' AND sname IS NOT NULL ORDER BY NumTstTakr DESC LIMIT 1",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="List the names of schools with more than 30 difference in enrollements between K-12 and ages 5-17? Please also give the full street adress of the schools.",
            ground_truth="SELECT T1.School, T1.Street FROM schools AS T1 INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.`Enrollment (K-12)` - T2.`Enrollment (Ages 5-17)` > 30",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="Give the names of the schools with the percent eligible for free meals in K-12 is more than 0.1 and test takers whose test score is greater than or equal to 1500?",
            ground_truth="SELECT T2.`School Name` FROM satscores AS T1 INNER JOIN frpm AS T2 ON T1.cds = T2.CDSCode WHERE CAST(T2.`Free Meal Count (K-12)` AS REAL) / T2.`Enrollment (K-12)` > 0.1 AND T1.NumGE1500 > 0",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="Name schools in Riverside which the average of average math score for SAT is grater than 400, what is the funding type of these schools?",
            ground_truth="SELECT T1.sname, T2.`Charter Funding Type` FROM satscores AS T1 INNER JOIN frpm AS T2 ON T1.cds = T2.CDSCode WHERE T2.`District Name` LIKE 'Riverside%' GROUP BY T1.sname, T2.`Charter Funding Type` HAVING CAST(SUM(T1.AvgScrMath) AS REAL) / COUNT(T1.cds) > 400",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="State the names and full communication address of high schools in Monterey which has more than 800 free or reduced price meals for ages 15-17?",
            ground_truth="SELECT T1.`School Name`, T2.Street, T2.City, T2.State, T2.Zip FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.County = 'Monterey' AND T1.`Free Meal Count (Ages 5-17)` > 800 AND T1.`School Type` = 'High Schools (Public)'",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="What is the average score in writing for the schools that were opened after 1991 or closed before 2000? List the school names along with the score. Also, list the communication number of the schools if there is any.",
            ground_truth="SELECT T2.School, T1.AvgScrWrite, T2.Phone FROM schools AS T2 LEFT JOIN satscores AS T1 ON T2.CDSCode = T1.cds WHERE strftime('%Y', T2.OpenDate) > '1991' OR strftime('%Y', T2.ClosedDate) < '2000'",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="Consider the average difference between K-12 enrollment and 15-17 enrollment of schools that are locally funded, list the names and DOC type of schools which has a difference above this average.",
            ground_truth="SELECT T2.School, T2.DOC FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.FundingType = 'Locally funded' AND (T1.`Enrollment (K-12)` - T1.`Enrollment (Ages 5-17)`) > (SELECT AVG(T3.`Enrollment (K-12)` - T3.`Enrollment (Ages 5-17)`) FROM frpm AS T3 INNER JOIN schools AS T4 ON T3.CDSCode = T4.CDSCode WHERE T4.FundingType = 'Locally funded')",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="When did the first-through-twelfth-grade school with the largest enrollment open?",
            ground_truth="SELECT T2.OpenDate FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode ORDER BY T1.`Enrollment (K-12)` DESC LIMIT 1",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="Which cities have the top 5 lowest enrollment number for students in grades 1 through 12?",
            ground_truth="SELECT T2.City FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode GROUP BY T2.City ORDER BY SUM(T1.`Enrollment (K-12)`) ASC LIMIT 5",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="What is the eligible free rate of the 10th and 11th schools with the highest enrolment for students in grades 1 through 12?",
            ground_truth="SELECT CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)` FROM frpm ORDER BY `Enrollment (K-12)` DESC LIMIT 9, 2",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="What is the eligible free or reduced price meal rate for the top 5 schools in grades 1-12 with the highest free or reduced price meal count of the schools with the ownership code 66?",
            ground_truth="SELECT CAST(T1.`FRPM Count (K-12)` AS REAL) / T1.`Enrollment (K-12)` FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.SOC = 66 ORDER BY T1.`FRPM Count (K-12)` DESC LIMIT 5",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="If there are any, what are the websites address of the schools with a free meal count of 1,900-2,000 to students aged 5-17? Include the name of the school.",
            ground_truth="SELECT T2.Website, T1.`School Name` FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`Free Meal Count (Ages 5-17)` BETWEEN 1900 AND 2000 AND T2.Website IS NOT NULL",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="What is the free rate for students between the ages of 5 and 17 at the school run by Kacey Gibson?",
            ground_truth="SELECT CAST(T2.`Free Meal Count (Ages 5-17)` AS REAL) / T2.`Enrollment (Ages 5-17)` FROM schools AS T1 INNER JOIN frpm AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.AdmFName1 = 'Kacey' AND T1.AdmLName1 = 'Gibson'",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="What is the administrator's email address of the chartered school with the fewest students enrolled in grades 1 through 12?",
            ground_truth="SELECT T2.AdmEmail1 FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`Charter School (Y/N)` = 1 ORDER BY T1.`Enrollment (K-12)` ASC LIMIT 1",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="Under whose administration is the school with the highest number of students scoring 1500 or more on the SAT? Indicate their full names.",
            ground_truth="SELECT T2.AdmFName1, T2.AdmLName1, T2.AdmFName2, T2.AdmLName2, T2.AdmFName3, T2.AdmLName3 FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode ORDER BY T1.NumGE1500 DESC LIMIT 1",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="What is the complete address of the school with the lowest excellence rate? Indicate the Street, City, Zip and State.",
            ground_truth="SELECT T2.Street, T2.City, T2.State, T2.Zip FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode ORDER BY CAST(T1.NumGE1500 AS REAL) / T1.NumTstTakr ASC LIMIT 1",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="What are the webpages for the Los Angeles County school that has between 2,000 and 3,000 test takers?",
            ground_truth="SELECT T2.Website FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T1.NumTstTakr BETWEEN 2000 AND 3000 AND T2.County = 'Los Angeles'",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="What is the average number of test takers from Fresno schools that opened between 1/1/1980 and 12/31/1980?",
            ground_truth="SELECT AVG(T1.NumTstTakr) FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE strftime('%Y', T2.OpenDate) = '1980' AND T2.County = 'Fresno'",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="What is the telephone number for the school with the lowest average score in reading in Fresno Unified?",
            ground_truth="SELECT T2.Phone FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T2.District = 'Fresno Unified' AND T1.AvgScrRead IS NOT NULL ORDER BY T1.AvgScrRead ASC LIMIT 1",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="List the names of virtual schools that are among the top 5 in their respective counties based on average reading scores.",
            ground_truth="SELECT School FROM (SELECT T2.School,T1.AvgScrRead, RANK() OVER (PARTITION BY T2.County ORDER BY T1.AvgScrRead DESC) AS rnk FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T2.Virtual = 'F' ) ranked_schools WHERE rnk <= 5",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="What is the type of education offered in the school who scored the highest average in Math?",
            ground_truth="SELECT T2.EdOpsName FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode ORDER BY T1.AvgScrMath DESC LIMIT 1",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="What is the average math score of the school with the lowest average score for all subjects, and in which county is it located?",
            ground_truth="SELECT T1.AvgScrMath, T2.County FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T1.AvgScrMath IS NOT NULL ORDER BY T1.AvgScrMath + T1.AvgScrRead + T1.AvgScrWrite ASC LIMIT 1",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="What is the average writing score of the school who has the highest number of test takers whose total SAT sscores are greater or equal to 1500? Indicate the city to where the school is situated.",
            ground_truth="SELECT T1.AvgScrWrite, T2.City FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode ORDER BY T1.NumGE1500 DESC LIMIT 1",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="What is the average writing score of each of the schools managed by Ricci Ulrich? List the schools and the corresponding average writing scores.",
            ground_truth="SELECT T2.School, T1.AvgScrWrite FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T2.AdmFName1 = 'Ricci' AND T2.AdmLName1 = 'Ulrich'",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="Which state special schools have the highest number of enrollees from grades 1 through 12?",
            ground_truth="SELECT T2.School FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.DOC = 31 ORDER BY T1.`Enrollment (K-12)` DESC LIMIT 1",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="What is the monthly average number of schools that opened in Alameda County under the jurisdiction of the Elementary School District in 1980?",
            ground_truth="SELECT CAST(COUNT(School) AS REAL) / 12 FROM schools WHERE DOC = 52 AND County = 'Alameda' AND strftime('%Y', OpenDate) = '1980'",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="What is the ratio of merged Unified School District schools in Orange County to merged Elementary School District schools?",
            ground_truth="SELECT CAST(SUM(CASE WHEN DOC = 54 THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN DOC = 52 THEN 1 ELSE 0 END) FROM schools WHERE StatusType = 'Merged' AND County = 'Orange'",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="Which different county has the most number of closed schools? Please provide the name of each school as well as the closure date.",
            ground_truth="SELECT DISTINCT County, School, ClosedDate FROM schools WHERE County = ( SELECT County FROM schools WHERE StatusType = 'Closed' GROUP BY County ORDER BY COUNT(School) DESC LIMIT 1 ) AND StatusType = 'Closed' AND school IS NOT NULL",
            context=SCHEMA_CONTEXT
        ),
        ReactQuestion(
            question="What is the postal street address for the school with the 7th highest Math average? Indicate the school's name.",
            ground_truth="SELECT T2.MailStreet, T2.School FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode ORDER BY T1.AvgScrMath DESC LIMIT 6, 1",
            context=SCHEMA_CONTEXT
        )
    ]


    # questions = [
    #     ReactQuestion(
    #         question="In which mailing street address can you find the school that has the lowest average score in reading? Also give the school's name.",
    #         ground_truth="SELECT T2.MailStreet, T2.School FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T1.AvgScrRead IS NOT NULL ORDER BY T1.AvgScrRead ASC LIMIT 1",
    #         context=SCHEMA_CONTEXT
    #     ),
    #     ReactQuestion(
    #         question="What is the total number of schools whose total SAT scores are greater or equal to 1500 whose mailing city is Lakeport?",
    #         ground_truth="SELECT COUNT(T1.cds) FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T2.MailCity = 'Lakeport' AND (T1.AvgScrRead + T1.AvgScrMath + T1.AvgScrWrite) >= 1500",
    #         context=SCHEMA_CONTEXT
    #     ),
    #     ReactQuestion(
    #         question="How many test takers are there at the school/s whose mailing city address is in Fresno?",
    #         ground_truth="SELECT T1.NumTstTakr FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode WHERE T2.MailCity = 'Fresno'",
    #         context=SCHEMA_CONTEXT
    #     ),
    #     ReactQuestion(
    #         question="Please specify all of the schools and their related mailing zip codes that are under Avetik Atoian's administration.",
    #         ground_truth="SELECT School, MailZip FROM schools WHERE AdmFName1 = 'Avetik' AND AdmLName1 = 'Atoian'",
    #         context=SCHEMA_CONTEXT
    #     ),
    #     ReactQuestion(
    #         question="Of the schools with a mailing state address in California, what is the ratio of the schools located in the county of Colusa against the school located in the county of Humboldt?",
    #         ground_truth="SELECT CAST(SUM(CASE WHEN County = 'Colusa' THEN 1 ELSE 0 END) AS REAL) / SUM(CASE WHEN County = 'Humboldt' THEN 1 ELSE 0 END) FROM schools WHERE MailState = 'CA'",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="Of all the schools with a mailing state address in California, how many are active in San Joaquin city?",
    #         ground_truth="SELECT COUNT(CDSCode) FROM schools WHERE City = 'San Joaquin' AND MailState = 'CA' AND StatusType = 'Active'",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="What is the phone number and extension number for the school that had the 333rd highest average writing score?",
    #         ground_truth="SELECT T2.Phone, T2.Ext FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode ORDER BY T1.AvgScrWrite DESC LIMIT 332, 1",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="What is the phone number and extension number for the school with the zip code 95203-3704? Indicate the school's name.",
    #         ground_truth="SELECT Phone, Ext, School FROM schools WHERE Zip = '95203-3704'",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="What is the website for the schools under the administrations of Mike Larson and Dante Alvarez?",
    #         ground_truth="SELECT Website FROM schools WHERE (AdmFName1 = 'Mike' AND AdmLName1 = 'Larson') OR (AdmFName1 = 'Dante' AND AdmLName1 = 'Alvarez')",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="What are the websites for all the partially virtual chartered schools located in San Joaquin?",
    #         ground_truth="SELECT Website FROM schools WHERE County = 'San Joaquin' AND Virtual = 'P' AND Charter = 1",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="How many chartered schools located in the city of Hickman are owned by the Elementary School District?",
    #         ground_truth="SELECT COUNT(School) FROM schools WHERE DOC = 52 AND Charter = 1 AND City = 'Hickman'",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="What is the total number of non-chartered schools in the county of Los Angeles with a percent (%) of eligible free meals for grades 1 through 12 that is less than 0.18%?",
    #         ground_truth="SELECT COUNT(T2.School) FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.County = 'Los Angeles' AND T2.Charter = 0 AND CAST(T1.`Free Meal Count (K-12)` AS REAL) * 100 / T1.`Enrollment (K-12)` < 0.18",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="In chartered schools with charter number 00D2, what are the names of all the administrators? Include the name of the school and the city to which it belongs",
    #         ground_truth="SELECT AdmFName1, AdmLName1, School, City FROM schools WHERE Charter = 1 AND CharterNum = '00D2'",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="What is the total number of schools with a mailing city in Hickman belonging to the charter number 00D4?",
    #         ground_truth="SELECT COUNT(*) FROM schools WHERE CharterNum = '00D4' AND MailCity = 'Hickman'",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="What is the ratio in percentage of Santa Clara County schools that are locally funded compared to all other types of charter school funding?",
    #         ground_truth="SELECT CAST(SUM(CASE WHEN FundingType = 'Locally funded' THEN 1 ELSE 0 END) AS REAL) * 100 / SUM(CASE WHEN FundingType != 'Locally funded' THEN 1 ELSE 0 END) FROM schools WHERE County = 'Santa Clara' AND Charter = 1",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="Between 1/1/2000 to 12/31/2005, how many directly funded schools opened in the county of Stanislaus?",
    #         ground_truth="SELECT COUNT(School) FROM schools WHERE strftime('%Y', OpenDate) BETWEEN '2000' AND '2005' AND County = 'Stanislaus' AND FundingType = 'Directly funded'",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="What is the total amount of Community College District closure in 1989 in the city of San Francisco?",
    #         ground_truth="SELECT COUNT(School) FROM schools WHERE strftime('%Y', ClosedDate) = '1989' AND City = 'San Francisco' AND DOCType = 'Community College District'",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="Which county reported the most number of school closure in the 1980s with school wonership code belonging to Youth Authority Facilities (CEA)?",
    #         ground_truth="SELECT County FROM schools WHERE strftime('%Y', ClosedDate) BETWEEN '1980' AND '1989' AND StatusType = 'Closed' AND SOC = 11 GROUP BY County ORDER BY COUNT(School) DESC LIMIT 1",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="Please provide the National Center for Educational Statistics school district identification number for all schools with a School Ownership Code that are part of the State Special Schools.",
    #         ground_truth="SELECT NCESDist FROM schools WHERE SOC = 31",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="How many active and closed District Community Day Schools are there in the county of Alpine?",
    #         ground_truth="SELECT COUNT(School) FROM schools WHERE (StatusType = 'Closed' OR StatusType = 'Active') AND SOC = 69 AND County = 'Alpine'",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="What is the district code for the School that does not offer a magnet program in the city of Fresno?",
    #         ground_truth="SELECT T1.`District Code` FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.City = 'Fresno' AND T2.Magnet = 0",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="How many students from the ages of 5 to 17 are enrolled at the State Special School school in Fremont for the 2014-2015 academic year?",
    #         ground_truth="SELECT T1.`Enrollment (Ages 5-17)` FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.EdOpsCode = 'SSS' AND T2.City = 'Fremont' AND T1.`Academic Year` BETWEEN 2014 AND 2015",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="What is the free or reduced price meal count for ages 5 to 17 in the Youth Authority School with a mailing street address of PO Box 1040?",
    #         ground_truth="SELECT T1.`FRPM Count (Ages 5-17)` FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.MailStreet = 'PO Box 1040' AND T2.SOCType = 'Youth Authority Facilities'",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="What is the lowest grade for the District Special Education Consortia School with National Center for Educational Statistics school district identification number of 0613360?",
    #         ground_truth="SELECT MIN(T1.`Low Grade`) FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.NCESDist = '0613360' AND T2.EdOpsCode = 'SPECON'",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="What is the educational level name for the schools with Breakfast Provision 2 in county code 37? Indicate the name of the school.",
    #         ground_truth="SELECT T2.EILName, T2.School FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`NSLP Provision Status` = 'Breakfast Provision 2' AND T1.`County Code` = 37",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="What is the city location of the high school level school with Lunch Provision 2 whose lowest grade is 9 and the highest grade is 12 in the county of Merced?",
    #         ground_truth="SELECT T2.City FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`NSLP Provision Status` = 'Lunch Provision 2' AND T2.County = 'Merced' AND T1.`Low Grade` = 9 AND T1.`High Grade` = 12 AND T2.EILCode = 'HS'",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="Which schools served a grade span of Kindergarten to 9th grade in the county of Los Angeles and what is its Percent (%) Eligible FRPM (Ages 5-17)?",
    #         ground_truth="SELECT T2.School, T1.`FRPM Count (Ages 5-17)` * 100 / T1.`Enrollment (Ages 5-17)` FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.County = 'Los Angeles' AND T2.GSserved = 'K-9'",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="What is the most common type of grade span served in the city of Adelanto?",
    #         ground_truth="SELECT GSserved FROM schools WHERE City = 'Adelanto' GROUP BY GSserved ORDER BY COUNT(GSserved) DESC LIMIT 1",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="Between San Diego and Santa Barbara, which county offers the most number of schools that does not offer physical building? Indicate the amount.",
    #         ground_truth="SELECT County, COUNT(Virtual) FROM schools WHERE (County = 'San Diego' OR County = 'Santa Barbara') AND Virtual = 'F' GROUP BY County ORDER BY COUNT(Virtual) DESC LIMIT 1",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="What is the school type of the school with the highest latitude? Indicate the name of the school as well as the latitude coordinates.",
    #         ground_truth="SELECT T1.`School Type`, T1.`School Name`, T2.Latitude FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode ORDER BY T2.Latitude DESC LIMIT 1",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="In which city can you find the school in the state of California with the lowest latitude coordinates and what is its lowest grade? Indicate the school name.",
    #         ground_truth="SELECT T2.City, T1.`Low Grade`, T1.`School Name` FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.State = 'CA' ORDER BY T2.Latitude ASC LIMIT 1",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="What is the grade span offered in the school with the highest longitude?",
    #         ground_truth="SELECT GSoffered FROM schools ORDER BY ABS(longitude) DESC LIMIT 1",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="Of the schools that offers a magnet program serving a grade span of Kindergarten to 8th grade, how many offers Multiple Provision Types? List the number of cities that offers a Kindergarten to 8th grade span and indicate how many schools are there serving such grade span for each city.",
    #         ground_truth="SELECT T2.City, COUNT(T2.CDSCode) FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.Magnet = 1 AND T2.GSoffered = 'K-8' AND T1.`NSLP Provision Status` = 'Multiple Provision Types' GROUP BY T2.City",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="What are the two most common first names among the school administrators? Indicate the district to which they administer.",
    #         ground_truth="SELECT DISTINCT T1.AdmFName1, T1.District FROM schools AS T1 INNER JOIN ( SELECT admfname1 FROM schools GROUP BY admfname1 ORDER BY COUNT(admfname1) DESC LIMIT 2 ) AS T2 ON T1.AdmFName1 = T2.admfname1",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="What is the Percent (%) Eligible Free (K-12) in the school administered by an administrator whose first name is Alusine. List the district code of the school.",
    #         ground_truth="SELECT T1.`Free Meal Count (K-12)` * 100 / T1.`Enrollment (K-12)`, T1.`District Code` FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.AdmFName1 = 'Alusine'",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="What is the administrator's last name that oversees the school with Charter number 40? Indicate the district, the county where the school is situated, and the name of the school.",
    #         ground_truth="SELECT AdmLName1, District, County, School FROM schools WHERE CharterNum = '0040'",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="What are the valid e-mail addresses of the administrator of the school located in the San Bernardino county, City of San Bernardino City Unified that opened between 1/1/2009 to 12/31/2010 whose school types are public Intermediate/Middle Schools and Unified Schools?",
    #         ground_truth="SELECT T2.AdmEmail1, T2.AdmEmail2 FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.County = 'San Bernardino' AND T2.City = 'San Bernardino' AND T2.DOC = 54 AND strftime('%Y', T2.OpenDate) BETWEEN '2009' AND '2010' AND T2.SOC = 62",
    #         context=SCHEMA_CONTEXT
    #     ),
    #
    #     ReactQuestion(
    #         question="What is the administrator's email address for the school with the highest number of test takers who received SAT scores of at least 1500?Provide the name of the school.",
    #         ground_truth="SELECT T2.AdmEmail1, T2.School FROM satscores AS T1 INNER JOIN schools AS T2 ON T1.cds = T2.CDSCode ORDER BY T1.NumGE1500 DESC LIMIT 1",
    #         context=SCHEMA_CONTEXT
    #     )
    #
    # ]

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
        question="What is the total number of non-chartered schools in the county of Los Angeles with a percent (%) of eligible free meals for grades 1 through 12 that is less than 0.18%?",
        context='',
        ground_truth="SELECT COUNT(T2.School) FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.County = 'Los Angeles' AND T2.Charter = 0 AND CAST(T1.`Free Meal Count (K-12)` AS REAL) * 100 / T1.`Enrollment (K-12)` < 0.18"
    )
    print(f"问题：{test_question.question}")

    
    result = workflow.run(test_question, verbose=True)
    # print("result:", result)
    # 从 curator_result 获取新策略数量
    curator_result = result.get('curator_result')
    new_strategies_count = curator_result.added_count if curator_result else 0
    print(f"   📚 本次新增 {new_strategies_count} 个策略")
    

if __name__ == "__main__":
    import sys
    main()
