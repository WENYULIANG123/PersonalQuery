# PersonalQuery

基于 STaRK 基准测试的个性化查询生成与检索评估系统

## 📋 项目简介

PersonalQuery 是一个集成了 **STaRK (Semi-structured Retrieval Benchmark)** 基准测试框架和**个性化查询生成系统**的综合项目。该项目专注于：

1. **检索模型评估**：在文本和关系知识库上评估多种 LLM 检索模型的性能
2. **个性化查询生成**：从 Amazon 商品评论中提取用户偏好，生成个性化搜索查询
3. **查询变体生成**：使用多种策略生成查询变体，提升检索系统的鲁棒性

## 🏗️ 项目结构

```
.
├── stark/                          # STaRK基准测试核心代码
│   ├── stark_qa/                  # 核心Python包
│   │   ├── models/                # 检索模型实现
│   │   │   ├── bm25.py           # BM25关键词检索
│   │   │   ├── vss.py            # 向量相似度检索
│   │   │   ├── colbertv2.py      # ColBERTv2上下文检索
│   │   │   ├── gritlm.py         # GritLM多模态检索
│   │   │   └── llm_reranker.py   # LLM重排序
│   │   ├── skb/                   # 知识库加载器
│   │   │   ├── amazon.py         # Amazon商品知识库
│   │   │   ├── mag.py            # 学术论文知识库
│   │   │   └── prime.py          # 生物医学知识库
│   │   └── tools/                # 工具函数
│   ├── code/                      # 自定义代码
│   │   ├── generate_query/       # 个性化查询生成系统
│   │   │   ├── main.py           # 主流程控制
│   │   │   ├── product_extraction.py      # 商品实体提取
│   │   │   ├── user_preference_extraction.py  # 用户偏好提取
│   │   │   ├── entity_matching.py         # 实体匹配
│   │   │   └── query_generation.py       # 查询生成
│   │   ├── generate_strategy_variants.py  # 查询变体生成
│   │   └── analysis/             # 分析脚本
│   ├── eval.py                   # 主评估脚本
│   └── requirements.txt          # Python依赖
├── data/                         # 数据目录
│   └── Amazon-Reviews-2018/      # Amazon评论数据
├── script/                       # 批处理脚本
│   ├── evalscrpit/              # 评估脚本
│   ├── GenerateQueryScrpit/     # 查询生成脚本
│   └── analysisScript/          # 分析脚本
└── README.md                     # 本文件
```

## 🚀 快速开始

### 环境配置

项目使用 Conda 环境管理依赖：

```bash
# 激活环境
conda activate /home/wlia0047/ar57_scratch/wenyu/stark

# 验证环境
python --version
```

### 安装依赖

依赖已预装在 Conda 环境中，主要包括：

- PyTorch, Transformers
- ColBERT, BM25, GritLM
- LangChain, OpenAI, Anthropic
- Pandas, NumPy, scikit-learn

如需重新安装，可参考 `stark/requirements.txt`。

## 📖 主要功能

### 1. 检索模型评估

使用 STaRK 基准测试评估多种检索模型：

```bash
cd stark

# 评估 BM25 模型
python eval.py --dataset amazon --model BM25 --split test

# 评估向量检索模型
python eval.py --dataset amazon --model VSS --emb_model text-embedding-ada-002 --split test

# 评估 ColBERTv2
python eval.py --dataset amazon --model Colbertv2 --split test

# 评估 LLM 重排序
python eval.py --dataset amazon --model LLMReranker \
    --emb_model text-embedding-ada-002 \
    --llm_model gpt-4-1106-preview \
    --split test
```

**支持的模型：**
- `BM25`: 传统关键词检索
- `VSS`: 向量相似度检索
- `MultiVSS`: 多块向量检索
- `ColBERT`/`Colbertv2`: 上下文感知检索
- `GritLM`: 多模态检索
- `LLMReranker`: LLM 重排序

**支持的数据集：**
- `amazon`: Amazon 商品知识库
- `mag`: 学术论文知识库
- `prime`: 生物医学知识库

### 2. 个性化查询生成

从用户评论生成个性化搜索查询：

```bash
cd stark/code/generate_query

# 运行完整流程
python main.py
```

**工作流程：**
1. **商品实体提取**：从商品元数据中提取结构化实体
2. **用户偏好提取**：从用户评论中提取偏好实体
3. **实体匹配**：基于 7 维逻辑映射匹配商品实体与用户偏好
4. **查询生成**：生成自然语言查询语句

详细说明请参考 [`stark/code/generate_query/README.md`](stark/code/generate_query/README.md)

### 3. 查询变体生成

使用多种策略生成查询变体：

```bash
cd stark/code

# 生成查询变体
python generate_strategy_variants.py
```

**支持的策略：**
- `wordnet`: WordNet 同义词替换
- `embedding`: 嵌入相似度替换
- `character`: 字符级扰动
- `dependency`: 依存树变换
- `typo`: 拼写错误模拟
- `other`: 其他变换策略

## 📊 评估指标

系统支持多种检索评估指标：

- **MRR** (Mean Reciprocal Rank)
- **MAP** (Mean Average Precision)
- **R-Precision**
- **Recall@K** (K=5, 10, 20, 50, 100)
- **Hit@K** (K=1, 3, 5, 10, 20, 50)

## 🔧 配置说明

### 环境变量

项目使用 `.env` 文件管理 API 密钥：

```bash
# 复制环境变量文件（如果存在）
cp $ROOT_WORKTREE_PATH/.env .env
```

### Cursor Worktree 配置

项目已配置 Cursor worktree 自动设置：

```json
{
  "setup-worktree": [
    "conda activate /home/wlia0047/ar57_scratch/wenyu/stark",
    "cp $ROOT_WORKTREE_PATH/.env .env"
  ]
}
```

## 📝 使用示例

### 评估检索模型

```python
from stark_qa import load_qa, load_skb, load_model

# 加载数据集和知识库
qa_dataset = load_qa('amazon')
skb = load_skb('amazon', download_processed=True)

# 加载模型
model = load_model(args, skb)

# 评估
results = model.evaluate(pred_dict, answer_ids, metrics=['mrr', 'recall@10'])
```

### 生成个性化查询

```python
from generate_query.main import main

# 运行完整查询生成流程
main()
```

## 🛠️ 开发指南

### 添加新的检索模型

1. 在 `stark/stark_qa/models/` 中创建新模型类
2. 继承 `ModelForSTaRKQA` 基类
3. 实现 `forward()` 方法
4. 在 `load_model.py` 中注册模型

### 添加新的知识库

1. 在 `stark/stark_qa/skb/` 中创建知识库类
2. 实现知识库加载逻辑
3. 在 `load_skb.py` 中注册知识库

## 📚 相关资源

- **STaRK 官方网站**: https://stark.stanford.edu/
- **STaRK 论文**: https://arxiv.org/abs/2404.13207
- **Hugging Face 数据集**: https://huggingface.co/datasets/snap-stanford/stark
- **PyPI 包**: https://pypi.org/project/stark-qa/

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目基于 MIT 许可证开源。STaRK 基准测试遵循其原始许可证。

## 🙏 致谢

本项目基于 [STaRK](https://github.com/snap-stanford/stark) 基准测试框架构建，感谢 STaRK 团队的开源贡献。

---

**注意**: 本项目为研究用途，使用前请确保已正确配置所有 API 密钥和环境变量。
