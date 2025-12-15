from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# 定义工具
@tool
def calculator(expression: str) -> str:
    """安全计算数学表达式"""
    try:
        allowed = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow}
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"计算错误：{e}"

@tool
def search(query: str) -> str:
    """简易搜索：这里直接返回固定文案，验证链路即可"""
    return (
        "1. Python 的创建者是 Guido van Rossum\n"
        "   Python 首次发布于 1991 年\n"
        "   来源: https://en.wikipedia.org/wiki/Python_(programming_language)"
    )

# 初始化 LLM 和工具
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [calculator, search]

# 创建 agent
agent = create_react_agent(llm, tools)

# 测试问题
question = "Python 是什么时候创建的？"
result = agent.invoke({"messages": [("human", question)]})

print(result)