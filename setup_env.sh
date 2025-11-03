#!/bin/bash

# ACE Framework 环境变量设置脚本
# 使用方法：source setup_env.sh

echo "🔧 配置 ACE Framework 环境变量..."

# OpenAI API Key
read -p "请输入 OPENAI_API_KEY: " openai_key
export OPENAI_API_KEY="$openai_key"
echo "✅ OPENAI_API_KEY 已设置"

# 询问是否启用 LangSmith
read -p "是否启用 LangSmith 追踪？(y/n): " enable_langsmith

if [ "$enable_langsmith" = "y" ] || [ "$enable_langsmith" = "Y" ]; then
    # LangSmith 配置
    export LANGSMITH_TRACING=true
    export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
    
    read -p "请输入 LANGSMITH_API_KEY: " langsmith_key
    export LANGSMITH_API_KEY="$langsmith_key"
    
    read -p "请输入项目名称 (默认: ace-react-agent): " project_name
    project_name=${project_name:-ace-react-agent}
    export LANGSMITH_PROJECT="$project_name"
    
    echo "✅ LangSmith 追踪已启用"
    echo "   项目: $LANGSMITH_PROJECT"
    echo "   端点: $LANGSMITH_ENDPOINT"
    echo "   查看追踪: https://smith.langchain.com/"
else
    export LANGSMITH_TRACING=false
    echo "ℹ️  LangSmith 追踪未启用"
fi

echo ""
echo "✨ 环境配置完成！"
echo ""
echo "当前配置："
echo "  OPENAI_API_KEY: ${OPENAI_API_KEY:0:10}..."
echo "  LANGSMITH_TRACING: $LANGSMITH_TRACING"
if [ "$LANGSMITH_TRACING" = "true" ]; then
    echo "  LANGSMITH_PROJECT: $LANGSMITH_PROJECT"
fi
