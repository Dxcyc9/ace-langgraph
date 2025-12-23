"""
Playbook - ACE 的策略知识库（带向量检索）

存储学习到的策略，帮助 agent 提升性能。
支持基于语义相似度的向量检索。
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import json


# 可选的向量检索依赖
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
@dataclass
class Strategy:
    """单个学习到的策略。"""
    
    id: str
    content: str
    category: str
    helpful_count: int = 0
    harmful_count: int = 0
    neutral_count: int = 0
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    
    def mark_helpful(self):
        """将此策略标记为有用。"""
        self.helpful_count += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()
    
    def mark_harmful(self):
        """将此策略标记为有害。"""
        self.harmful_count += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()
    
    def mark_neutral(self):
        """将此策略标记为中性。"""
        self.neutral_count += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()
    
    @property
    def score(self) -> int:
        """计算净有效性得分。"""
        return self.helpful_count - self.harmful_count
    
    @property
    def total_uses(self) -> int:
        """总使用次数。"""
        return self.helpful_count + self.harmful_count + self.neutral_count
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "category": self.category,
            "helpful_count": self.helpful_count,
            "harmful_count": self.harmful_count,
            "neutral_count": self.neutral_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Strategy":
        return cls(**data)

class Playbook:
    """
    积累学习策略的知识库（支持向量检索）。
    
    Playbook 是 ACE 系统的"记忆"，存储从过去经验中
    学到的见解。支持基于语义相似度的向量检索。
    
    示例：
        >>> # 创建 Playbook（自动启用向量检索）
        >>> playbook = Playbook()
        >>> strategy = playbook.add_strategy("使用工具前先分析问题", category="react")
        >>> 
        >>> # 标记策略
        >>> playbook.mark_helpful(strategy.id)
        >>> 
        >>> # 向量检索相关策略
        >>> strategies = playbook.retrieve_strategies("如何使用工具？", top_k=3)
        >>> 
        >>> # 保存到文件
        >>> playbook.save_to_file("my_playbook.json")
        >>> 
        >>> # 从文件加载
        >>> loaded = Playbook.load_from_file("my_playbook.json")
    """
    
    def __init__(
        self,
        enable_retrieval: bool = False,
        collection_name: str = "ace_strategies",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "doubao-embedding"
    ):
        """
        初始化 Playbook。
        
        参数：
            enable_retrieval: 是否启用向量检索（需要安装相关依赖）
            collection_name: ChromaDB 集合名称
            persist_directory: 向量数据库持久化目录
            embedding_model: 嵌入模型名称
        """
        self._strategies: Dict[str, Strategy] = {}
        self._categories: Dict[str, List[str]] = {}  # category -> strategy_ids
        self._next_id = 1
        
        # 向量检索相关
        self.enable_retrieval = enable_retrieval
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        # 初始化向量数据库（如果启用）
        if self.enable_retrieval:
            try:
                self._init_vector_db()
                if self._strategies:  # 如果已有策略，提示需要重建索引
                    print(f"💡 提示：Playbook 有现有策略，可能需要调用 rebuild_index()")
            except Exception as e:
                print(f"⚠️  向量检索初始化失败，将使用基础模式: {e}")
                self.enable_retrieval = False
        
        if not self.enable_retrieval and enable_retrieval:
            print(f"💡 向量检索未启用。安装依赖: pip install chromadb langchain-openai")
    
    def _init_vector_db(self):
        """初始化向量数据库。"""
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        self.embeddings = HuggingFaceEmbeddings(
            # model_name=r"G:\code\ace-langgraph\models\BAAI\bge-small-zh",  # 2. 本地绝对路径
            model_name="./models/BAAI/bge-small-zh",
            model_kwargs={"device": "cpu"},  # 3. 官方传参方式
            encode_kwargs={"normalize_embeddings": True}
        )

        
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print(f"✅ 向量检索已启用（集合: {self.collection_name}）")
    
    def add_strategy(
        self, 
        content: str, 
        category: str = "general",
        strategy_id: Optional[str] = None
    ) -> Strategy:
        """
        向 playbook 添加新策略。
        
        参数：
            content: 策略内容
            category: 策略分类（如 "react", "reasoning", "general"）
            strategy_id: 可选的自定义 ID（通常自动生成）
            
        返回：
            新创建的 Strategy 对象
        """
        if strategy_id is None:
            strategy_id = self._generate_id(category)
        
        strategy = Strategy(
            id=strategy_id,
            content=content,
            category=category,
            helpful_count=1
        )
        
        self._strategies[strategy_id] = strategy
        self._categories.setdefault(category, []).append(strategy_id)
        
        # 添加到向量数据库（如果启用）
        if self.enable_retrieval:
            self._add_to_vector_db(strategy)
        
        return strategy
    
    def get_strategy(self, strategy_id: str) -> Optional[Strategy]:
        """通过 ID 获取特定策略。"""
        return self._strategies.get(strategy_id)
    
    def mark_helpful(self, strategy_id: str) -> Optional[Strategy]:
        """将策略标记为有用。"""
        strategy = self.get_strategy(strategy_id)
        if strategy:
            strategy.mark_helpful()
        return strategy
    
    def mark_harmful(self, strategy_id: str) -> Optional[Strategy]:
        """将策略标记为有害。"""
        strategy = self.get_strategy(strategy_id)
        if strategy:
            strategy.mark_harmful()
        return strategy
    
    def mark_neutral(self, strategy_id: str) -> Optional[Strategy]:
        """将策略标记为中性。"""
        strategy = self.get_strategy(strategy_id)
        if strategy:
            strategy.mark_neutral()
        return strategy
    
    def remove_strategy(self, strategy_id: str) -> bool:
        """
        移除指定的策略。
        
        参数：
            strategy_id: 策略ID
            
        返回：
            True 如果成功移除，False 如果策略不存在
        """
        if strategy_id not in self._strategies:
            return False
        
        # 获取策略信息
        strategy = self._strategies[strategy_id]
        category = strategy.category
        
        # 从策略字典中移除
        del self._strategies[strategy_id]
        
        # 从分类索引中移除
        if category in self._categories:
            if strategy_id in self._categories[category]:
                self._categories[category].remove(strategy_id)
            # 如果分类为空，删除分类
            if not self._categories[category]:
                del self._categories[category]
        
        # 如果启用了向量检索，从向量数据库中删除
        if self.enable_retrieval:
            try:
                self.collection.delete(ids=[strategy_id])
            except Exception as e:
                print(f"⚠️ 从向量数据库删除策略失败: {e}")
        
        return True
    
    def update_strategy_content(
        self, 
        strategy_id: str, 
        additional_content: str
    ) -> bool:
        """
        向现有策略追加内容（增量更新）。
        
        参数：
            strategy_id: 策略ID
            additional_content: 要追加的内容
            
        返回：
            True 如果成功更新，False 如果策略不存在
        """
        strategy = self.get_strategy(strategy_id)
        if not strategy:
            return False
        
        # 追加内容
        strategy.content = f"{strategy.content}\n\n补充：{additional_content}"
        strategy.updated_at = datetime.now(timezone.utc).isoformat()
        
        # 如果启用向量检索，更新向量索引
        if self.enable_retrieval:
            try:
                # 删除旧的向量
                self.collection.delete(ids=[strategy_id])
                # 添加新的向量
                self._add_to_vector_db(strategy)
            except Exception as e:
                print(f"⚠️ 更新向量数据库失败: {e}")
        
        return True
    
    def get_all_strategies(self) -> List[Strategy]:
        """获取按有效性排序的所有策略。"""
        return sorted(
            self._strategies.values(),
            key=lambda s: s.score,
            reverse=True
        )
    
    def get_strategies_by_category(self, category: str) -> List[Strategy]:
        """获取指定分类的策略。"""
        strategy_ids = self._categories.get(category, [])
        return [self._strategies[sid] for sid in strategy_ids if sid in self._strategies]
    
    def get_top_strategies(self, n: int = 5, min_score: int = 0) -> List[Strategy]:
        """
        获取得分最高的前 N 个策略。
        
        参数：
            n: 返回的策略数量
            min_score: 最低分数阈值
            
        返回：
            得分最高的策略列表
        """
        all_strategies = self.get_all_strategies()
        filtered = [s for s in all_strategies if s.score >= min_score]
        return filtered[:n]
    
    # ------------------------------------------------------------------ #
    # 向量检索方法
    # ------------------------------------------------------------------ #
    
    def retrieve_strategies(
        self,
        question: str,
        top_k: int = 5,
        min_score: int = 0,
        category: Optional[str] = None
    ) -> List[Strategy]:
        """
        根据问题检索最相关的策略（使用向量检索）。
        
        参数：
            question: 当前问题
            top_k: 返回的策略数量
            min_score: 最低分数阈值（helpful_count - harmful_count）
            category: 可选的分类过滤
            
        返回：
            按相关性排序的策略列表
        """
        # 如果没有策略，返回空列表
        if len(self._strategies) == 0:
            return []
        
        # 如果向量检索未启用，回退到 get_top_strategies
        if not self.enable_retrieval:
            return self.get_top_strategies(n=top_k, min_score=min_score)
        
        try:
            # 生成问题的嵌入
            question_embedding = self.embeddings.embed_query(question)
            
            # 构建过滤条件
            where_filter = None
            if category:
                where_filter = {"category": category}
            
            # 从 ChromaDB 检索
            results = self.collection.query(
                query_embeddings=[question_embedding],
                n_results=min(top_k * 2, len(self._strategies)),  # 多检索一些，用于过滤
                where=where_filter
            )
            
            # 提取策略 ID
            strategy_ids = results["ids"][0] if results["ids"] else []
            
            # 获取完整的策略对象
            strategies = []
            for strategy_id in strategy_ids:
                strategy = self.get_strategy(strategy_id)
                if strategy and strategy.score >= min_score:
                    strategies.append(strategy)
            
            # 返回前 top_k 个
            return strategies[:top_k]
            
        except Exception as e:
            print(f"⚠️  向量检索失败，回退到按分数排序: {e}")
            return self.get_top_strategies(n=top_k, min_score=min_score)
    
    def rebuild_index(self) -> None:
        """
        重建向量索引。
        
        在以下情况使用：
        - 从文件加载 Playbook 后
        - 嵌入模型更换后
        - 数据库损坏需要修复
        """
        if not self.enable_retrieval:
            print("⚠️  向量检索未启用，无法重建索引")
            return
        
        print(f"🔨 重建向量索引（共 {len(self._strategies)} 个策略）...")
        
        # 清空现有集合
        try:
            self.chroma_client.delete_collection(self.collection.name)
        except Exception:
            pass
        
        # 重新创建集合
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # 批量添加所有策略
        for strategy in self._strategies.values():
            self._add_to_vector_db(strategy)
        
        print(f"✅ 向量索引重建完成")
    
    def _add_to_vector_db(self, strategy: Strategy) -> None:
        """
        将策略添加到向量数据库。
        
        内部方法，生成嵌入并存储到 ChromaDB。
        """
        try:
            # 生成嵌入
            embedding = self.embeddings.embed_query(strategy.content)
            
            # 添加到 ChromaDB
            self.collection.add(
                ids=[strategy.id],
                embeddings=[embedding],
                metadatas=[{
                    "category": strategy.category,
                    "score": strategy.score,
                    "helpful_count": strategy.helpful_count,
                    "harmful_count": strategy.harmful_count,
                }],
                documents=[strategy.content]
            )
        except Exception as e:
            print(f"⚠️  添加策略到向量数据库失败: {e}")
    
    # ------------------------------------------------------------------ #
    # 序列化和持久化
    # ------------------------------------------------------------------ #
    
    def to_dict(self) -> dict:
        """转换为字典格式。"""
        return {
            "strategies": {
                strategy_id: strategy.to_dict() 
                for strategy_id, strategy in self._strategies.items()
            },
            "categories": self._categories,
            "next_id": self._next_id,
            "version": "1.0",
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
    
    @classmethod
    def from_dict(
        cls, 
        data: dict,
        enable_retrieval: bool = True,
        collection_name: str = "ace_strategies",
        persist_directory: str = "./chroma_db"
    ) -> "Playbook":
        """
        从字典创建 Playbook。
        
        参数：
            data: 序列化的 Playbook 数据
            enable_retrieval: 是否启用向量检索
            collection_name: ChromaDB 集合名称
            persist_directory: 向量数据库持久化目录
        """
        playbook = cls(
            enable_retrieval=enable_retrieval,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        playbook._next_id = data.get("next_id", 1)
        
        # 加载策略（不触发向量索引）
        strategies_data = data.get("strategies", {})
        for strategy_id, strategy_data in strategies_data.items():
            strategy = Strategy.from_dict(strategy_data)
            playbook._strategies[strategy_id] = strategy
        
        # 加载分类索引
        playbook._categories = data.get("categories", {})
        
        # 如果启用了向量检索，重建索引
        if playbook.enable_retrieval and playbook._strategies:
            playbook.rebuild_index()
        
        return playbook
    
    def dumps(self) -> str:
        """序列化为 JSON 字符串。"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def loads(
        cls, 
        data: str,
        enable_retrieval: bool = True,
        collection_name: str = "ace_strategies",
        persist_directory: str = "./chroma_db"
    ) -> "Playbook":
        """
        从 JSON 字符串反序列化。
        
        参数：
            data: JSON 字符串
            enable_retrieval: 是否启用向量检索
            collection_name: ChromaDB 集合名称
            persist_directory: 向量数据库持久化目录
        """
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError("Playbook 序列化数据必须是 JSON 对象")
        return cls.from_dict(
            payload,
            enable_retrieval=enable_retrieval,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
    
    def save_to_file(self, path: str) -> None:
        """
        保存 Playbook 到 JSON 文件。
        
        参数：
            path: 文件路径
            
        示例：
            >>> playbook.save_to_file("trained_agent.json")
            >>> playbook.save_to_file("checkpoints/epoch_10.json")
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with file_path.open("w", encoding="utf-8") as f:
            f.write(self.dumps())
        
        print(f"✅ Playbook 已保存到: {path}")
    
    @classmethod
    def load_from_file(
        cls, 
        path: str,
        enable_retrieval: bool = False,
        collection_name: str = "ace_strategies",
        persist_directory: str = "./chroma_db"
    ) -> "Playbook":
        """
        从 JSON 文件加载 Playbook。
        
        参数：
            path: 文件路径
            enable_retrieval: 是否启用向量检索
            collection_name: ChromaDB 集合名称
            persist_directory: 向量数据库持久化目录
            
        返回：
            加载的 Playbook 实例
            
        示例：
            >>> playbook = Playbook.load_from_file("trained_agent.json")
            >>> print(f"加载了 {len(playbook)} 个策略")
            
        异常：
            FileNotFoundError: 文件不存在
            json.JSONDecodeError: JSON 格式错误
            ValueError: 数据格式无效
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Playbook 文件不存在: {path}")
        
        with file_path.open("r", encoding="utf-8") as f:
            playbook = cls.loads(
                f.read(),
                enable_retrieval=enable_retrieval,
                collection_name=collection_name,
                persist_directory=persist_directory
            )
        
        print(f"✅ 已从 {path} 加载 Playbook（{len(playbook)} 个策略）")
        return playbook
    
    # ------------------------------------------------------------------ #
    # 统计和信息
    # ------------------------------------------------------------------ #
    
    def as_str(self) -> str:
        """
        格式化 Playbook 为带编号和评分的字符串（供 LLM 提示词使用）。
        
        格式示例：
        ## category_name
        - [strategy_id] 策略内容 (helpful=3, harmful=0, neutral=1)
        
        返回：
            格式化的 Playbook 字符串
        """
        if not self._strategies:
            return "（Playbook 为空）"
        
        parts = []
        for category, strategy_ids in sorted(self._categories.items()):
            parts.append(f"## {category}")
            for strategy_id in strategy_ids:
                strategy = self._strategies.get(strategy_id)
                if strategy:
                    counters = f"(helpful={strategy.helpful_count}, harmful={strategy.harmful_count}, neutral={strategy.neutral_count})"
                    parts.append(f"- [{strategy.id}] {strategy.content} {counters}")
        
        return "\n".join(parts)
    
    def stats_str(self) -> str:
        """
        获取格式化的统计信息字符串（供 LLM 提示词使用）。
        
        格式示例：
        分类: react(5), reasoning(7), tool_usage(3), 策略总数: 15, 标记统计: ✓12 ✗3 ○2
        
        返回：
            格式化的统计字符串
        """
        stats = self.stats()
        
        # 格式化分类信息
        category_info = ", ".join([
            f"{cat}({count})" for cat, count in stats['categories'].items()
        ]) if stats['categories'] else "无"
        
        return (
            f"分类: {category_info}, "
            f"策略总数: {stats['total_strategies']}, "
            f"标记统计: ✓{stats['tags']['helpful']} "
            f"✗{stats['tags']['harmful']} "
            f"○{stats['tags']['neutral']}"
        )
    
    def stats(self) -> Dict[str, object]:
        """
        获取 Playbook 统计信息。
        
        返回：
            包含各种统计数据的字典
        """
        total_helpful = sum(s.helpful_count for s in self._strategies.values())
        total_harmful = sum(s.harmful_count for s in self._strategies.values())
        total_neutral = sum(s.neutral_count for s in self._strategies.values())
        
        stats = {
            "total_strategies": len(self._strategies),
            "categories": {
                cat: len(ids) for cat, ids in self._categories.items()
            },
            "tags": {
                "helpful": total_helpful,
                "harmful": total_harmful,
                "neutral": total_neutral,
                "total": total_helpful + total_harmful + total_neutral,
            },
            "avg_score": sum(s.score for s in self._strategies.values()) / len(self._strategies)
                if self._strategies else 0,
        }
        
        # 添加向量数据库信息
        if self.enable_retrieval:
            try:
                chroma_count = self.collection.count()
                stats["vector_db"] = {
                    "enabled": True,
                    "collection_name": self.collection_name,
                    "indexed_strategies": chroma_count,
                    "sync_status": "✓" if chroma_count == len(self._strategies) else "⚠️ 不同步"
                }
            except Exception as e:
                stats["vector_db"] = {"enabled": True, "error": str(e)}
        else:
            stats["vector_db"] = {"enabled": False}
        
        return stats
    
    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    
    def _generate_id(self, category: str) -> str:
        """生成策略 ID。"""
        self._next_id += 1
        category_prefix = category[:3].lower()
        return f"{category_prefix}-{self._next_id:05d}"
    
    def __len__(self) -> int:
        return len(self._strategies)
    