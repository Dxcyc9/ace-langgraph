#!/bin/bash

# 快速启用 LangSmith 追踪
# 
# ⚠️  重要：必须使用 source 命令运行此脚本！
# 
# ✅ 正确用法：
#   source enable_langsmith.sh
#   或
#   . enable_langsmith.sh
# 
# ❌ 错误用法（环境变量不会生效）：
#   bash enable_langsmith.sh
#   ./enable_langsmith.sh

# 检查是否使用了 source 命令
if [ "$0" = "$BASH_SOURCE" ]; then
    echo "❌ 错误：请使用 'source enable_langsmith.sh' 运行此脚本"
    echo ""
    echo "正确用法："
    echo "  source enable_langsmith.sh"
    echo "  或"
    echo "  . enable_langsmith.sh"
    exit 1
fi

echo "🔍 启用 LangSmith 追踪..."

# 设置追踪配置
export LANGSMITH_TRACING=true
export LANGSMITH_ENDPOINT=https://api.smith.langchain.com

# 检查是否已设置 API key
if [ -z "$LANGSMITH_API_KEY" ]; then
    echo ""
    echo "⚠️  未检测到 LANGSMITH_API_KEY"
    echo ""
    echo "请先设置 API key："
    echo "  export LANGSMITH_API_KEY='your-api-key'"
    echo ""
    echo "获取 API key："
    echo "  https://smith.langchain.com/ → Settings → API Keys"
    echo ""
    return 1
fi

# 设置项目名称（如果未设置）
if [ -z "$LANGSMITH_PROJECT" ]; then
    export LANGSMITH_PROJECT="default"
    echo "✅ 使用默认项目: $LANGSMITH_PROJECT"
else
    echo "✅ 使用项目: $LANGSMITH_PROJECT"
fi

echo "✅ LangSmith 追踪已启用"
echo ""
echo "配置信息："
echo "  端点: $LANGSMITH_ENDPOINT"
echo "  项目: $LANGSMITH_PROJECT"
echo "  API Key: ${LANGSMITH_API_KEY:0:10}..."
echo ""
echo "查看追踪: https://smith.langchain.com/"
echo ""
echo "💡 提示：现在运行任何 ACE 代码都会自动追踪！"
