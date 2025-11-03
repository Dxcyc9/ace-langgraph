"""
ACE React Agent 训练程序

使用 train_questions.json 训练 ACE React Agent 的 Playbook，
自动学习策略并保存到 ace_react_playbook.json。
"""

import json
import sys
from pathlib import Path
from typing import List
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ace_langgraph.ace_react_agent import ACEReActWorkflow, get_default_tools
from ace_langgraph.types import ReactQuestion


def load_train_questions(filepath: str) -> List[ReactQuestion]:
    """
    从 JSON 文件加载训练问题。
    
    参数：
        filepath: JSON 文件路径
        
    返回：
        ReactQuestion 对象列表
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    questions = []
    for item in data:
        question = ReactQuestion(
            question=item['question'],
            ground_truth=item['ground_truth'],
            context=item.get('context', '')
        )
        questions.append(question)
    
    return questions


def train_agent(
    training_file: str = "train_questions.json",
    playbook_path: str = "ace_react_playbook.json",
    model_name: str = "gpt-4o-mini",
    max_iterations: int = 15,
    verbose: bool = False,
    show_progress: bool = True
):
    """
    训练 ACE React Agent。
    
    参数：
        training_file: 训练问题文件路径
        playbook_path: Playbook 保存路径
        model_name: LLM 模型名称
        max_iterations: Agent 最大迭代次数
        verbose: 是否显示详细过程
        show_progress: 是否显示训练进度
    """
    import os
    
    # 检查 API key
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  请设置 OPENAI_API_KEY 环境变量")
        return
    
    print("\n" + "="*70)
    print("ACE React Agent 训练程序")
    print("="*70)
    print(f"\n训练文件：{training_file}")
    print(f"Playbook 路径：{playbook_path}")
    print(f"模型：{model_name}")
    print(f"最大迭代：{max_iterations}")
    print(f"详细输出：{'是' if verbose else '否'}")
    print("="*70 + "\n")
    
    # 加载训练问题
    try:
        train_questions = load_train_questions(training_file)
        print(f"✓ 成功加载 {len(train_questions)} 个训练问题\n")
    except FileNotFoundError:
        print(f"❌ 找不到训练文件：{training_file}")
        return
    except Exception as e:
        print(f"❌ 加载训练文件失败：{e}")
        return
    
    # 创建 ACE React Workflow（会自动加载已有的 Playbook 或创建新的）
    print("初始化工作流...\n")
    workflow = ACEReActWorkflow(
        tools=get_default_tools(),
        model_name=model_name,
        max_iterations=max_iterations,
        playbook_path=playbook_path,  # 指定 Playbook 路径
        use_vector_retrieval=False,
        auto_save=True  # 每次运行后自动保存
    )
    
    # 显示初始状态
    initial_stats = workflow.playbook.stats()
    print(f"初始 Playbook 状态：")
    print(f"  - 策略数：{initial_stats['total_strategies']}")
    print(f"  - 分类数：{initial_stats['categories']}")
    print()
    
    # 训练统计
    training_stats = {
        'total_questions': len(train_questions),
        'correct': 0,
        'incorrect': 0,
        'errors': 0,
        'start_time': datetime.now()
    }
    
    # 开始训练
    print("="*70)
    print("开始训练...")
    print("="*70 + "\n")
    
    for i, question in enumerate(train_questions, 1):
        if show_progress and not verbose:
            print(f"\n{'='*70}")
            print(f"训练进度：{i}/{len(train_questions)}")
            print(f"{'='*70}")
            print(f"问题：{question.question}")
            print(f"正确答案：{question.ground_truth}")
            print("-" * 70)
        
        try:
            # 运行训练（会自动执行：Generator → Evaluator → Reflector → Curator）
            result = workflow.run(question, verbose=verbose)
            
            # 统计结果
            evaluation = result.get("evaluation")
            if evaluation:
                if evaluation.is_correct:
                    training_stats['correct'] += 1
                    status = "✓ 正确"
                else:
                    training_stats['incorrect'] += 1
                    status = "✗ 错误"
                
                if show_progress and not verbose:
                    react_result = result.get("react_result")
                    agent_answer = react_result.answer if react_result else "未生成答案"
                    print(f"Agent答案：{agent_answer[:80]}...")
                    print(f"评估：{status}")
                    print(f"反馈：{evaluation.feedback[:80]}...")
            
            # 显示学习到的新策略
            if show_progress and not verbose:
                curator_result = result.get('curator_result')
                if curator_result:
                    added = curator_result.added_count
                    updated = curator_result.updated_count
                    removed = curator_result.removed_count
                    marked = curator_result.marked_count
                    
                    changes = []
                    if added > 0:
                        changes.append(f"新增 {added} 个")
                    if updated > 0:
                        changes.append(f"更新 {updated} 个")
                    if removed > 0:
                        changes.append(f"移除 {removed} 个")
                    if marked > 0:
                        changes.append(f"标记 {marked} 个")
                    
                    if changes:
                        print(f"策略变化：{', '.join(changes)}")
                    else:
                        print(f"策略变化：无变化")
            
            # 显示当前进度
            if show_progress and not verbose:
                current_accuracy = (training_stats['correct'] / i * 100) if i > 0 else 0
                current_strategies = len(workflow.playbook)
                print(f"\n当前统计：")
                print(f"  - 正确率：{current_accuracy:.1f}% ({training_stats['correct']}/{i})")
                print(f"  - 策略数：{current_strategies}")
                
        except Exception as e:
            print(f"❌ 训练失败：{e}")
            training_stats['errors'] += 1
            if verbose:
                import traceback
                traceback.print_exc()
    
    # 训练结束，显示最终统计
    training_stats['end_time'] = datetime.now()
    training_stats['duration'] = (training_stats['end_time'] - training_stats['start_time']).total_seconds()
    
    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)
    
    # 显示训练统计
    print(f"\n训练统计：")
    print(f"  总问题数：{training_stats['total_questions']}")
    print(f"  正确数量：{training_stats['correct']} ✓")
    print(f"  错误数量：{training_stats['incorrect']} ✗")
    print(f"  异常数量：{training_stats['errors']} ⚠️")
    print(f"  正确率：{training_stats['correct']/training_stats['total_questions']*100:.1f}%")
    print(f"  训练时长：{training_stats['duration']:.1f} 秒")
    
    # 显示 Playbook 统计
    final_stats = workflow.playbook.stats()
    print(f"\n最终 Playbook 状态：")
    print(f"  策略总数：{final_stats['total_strategies']}")
    print(f"  分类数量：{final_stats['categories']}")
    print(f"  标记统计：✓{final_stats['tags']['helpful']} / ✗{final_stats['tags']['harmful']} / ~{final_stats['tags']['neutral']}")
    print(f"  平均得分：{final_stats['avg_score']:.2f}")
    
    strategies_gained = final_stats['total_strategies'] - initial_stats['total_strategies']
    print(f"\n本次训练：")
    print(f"  新增策略：{strategies_gained} 个")
    
    # 显示前 10 个高分策略
    if len(workflow.playbook) > 0:
        print(f"\n前 10 个高分策略：")
        print("-" * 70)
        top_strategies = workflow.playbook.get_top_strategies(n=10)
        
        for i, strategy in enumerate(top_strategies, 1):
            print(f"\n{i}. [{strategy.id}] 分数: {strategy.score}")
            print(f"   分类: {strategy.category}")
            print(f"   内容: {strategy.content[:100]}{'...' if len(strategy.content) > 100 else ''}")
            print(f"   标记: ✓{strategy.helpful_count} / ✗{strategy.harmful_count} / ~{strategy.neutral_count}")
    
    print(f"\n💾 Playbook 已保存到：{playbook_path}")
    print(f"\n✅ 训练完成！可以使用 test_ace_react.py 测试性能")
    print()
    
    return workflow, training_stats


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ACE React Agent 训练程序")
    parser.add_argument(
        "--training-file",
        default="train_questions.json",
        help="训练问题文件路径（默认：train_questions.json）"
    )
    parser.add_argument(
        "--playbook",
        default="ace_react_playbook.json",
        help="Playbook 保存路径（默认：ace_react_playbook.json）"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM 模型名称（默认：gpt-4o-mini）"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=15,
        help="Agent 最大迭代次数（默认：15）"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细输出"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，不显示进度"
    )
    
    args = parser.parse_args()
    
    train_agent(
        training_file=args.training_file,
        playbook_path=args.playbook,
        model_name=args.model,
        max_iterations=args.max_iterations,
        verbose=args.verbose,
        show_progress=not args.quiet
    )


if __name__ == "__main__":
    main()
