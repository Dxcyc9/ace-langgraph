from langchain.agents import AgentExecutor

agent = create_agent(
    model=f"openai:{self.model_name}",
    tools=self.tools,
    system_prompt=self._get_system_prompt(question, context)
)
# 关键：把时间和次数都放大
agent_executor = AgentExecutor(
    agent=agent,
    tools=self.tools,
    max_iterations=50,
    max_execution_time=300,   # 5 分钟
    early_stopping_method="generate",  # 强制继续
    verbose=True
)
result = agent_executor.invoke({"input": question})