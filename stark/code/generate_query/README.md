# Amazon Product Query Generation System

这个项目已经从单一文件拆分为模块化的架构，便于维护和扩展。

## 📁 文件结构

### 核心模块

1. **`main.py`** - 主入口文件
   - 串联所有模块的执行流程
   - 处理数据加载、保存和最终结果展示
   - 包含完整的pipeline执行逻辑

2. **`product_extraction.py`** - 商品实体提取模块
   - 负责从商品元数据中提取结构化实体
   - 包含商品实体提取的专用响应处理器
   - 处理HTML清理和元数据解析

3. **`user_preference_extraction.py`** - 用户偏好实体提取模块
   - 负责从用户评论中提取偏好实体
   - 支持并发处理多个评论
   - 包含用户偏好提取的专用响应处理器

4. **`entity_matching.py`** - 实体匹配和查询生成模块
   - 实现商品实体与用户偏好实体的语义匹配
   - 基于7维逻辑映射的智能匹配算法
   - 生成自然语言查询语句

5. **`utils.py`** - 共享工具模块
   - 包含所有模块共享的通用函数
   - API密钥管理、日志记录、LLM配置等
   - 提供统一的接口包装

## 🔄 执行流程

```
数据加载 → 商品实体提取 → 用户偏好提取 → 实体匹配 → 查询生成 → 结果保存
    ↓           ↓            ↓            ↓          ↓          ↓
 main.py → product_ → user_preference_ → entity_ → entity_ → main.py
           extraction  extraction       matching  matching
```

## 🔧 环境配置

### 自动环境激活
项目已配置自动激活conda环境 `/home/wlia0047/ar57_scratch/wenyu/stark`：

1. **Cursor规则**: 在 `.cursor/rules/` 中配置了自动激活规则
2. **Shell配置**: `.bashrc` 中配置了目录检测自动激活
3. **激活脚本**: 项目根目录提供便捷的激活脚本

### 手动激活环境
```bash
# 使用激活脚本
./activate_stark.sh

# 或直接激活
conda activate /home/wlia0047/ar57_scratch/wenyu/stark
```

### 环境检查
```bash
# 检查当前环境
conda info --envs

# 确认Python路径
which python3
```

## 🚀 使用方法

### 运行完整流程
```bash
python main.py
```

### 单独测试模块
```bash
# 测试商品实体提取
python -c "from product_extraction import process_product_extraction_response; print('OK')"

# 测试用户偏好提取
python -c "from user_preference_extraction import process_user_preference_extraction_response; print('OK')"

# 测试实体匹配
python -c "from entity_matching import process_entity_matching_response; print('OK')"
```

## 📋 响应处理器

每个场景都有专门的响应处理器：

| 场景 | 处理器函数 | 特点 |
|------|-----------|------|
| 商品实体提取 | `process_product_extraction_response` | 直接JSON解析，支持结构化输出 |
| 用户偏好提取 | `process_user_preference_extraction_response` | Chain of Thought智能提取 |
| 实体匹配 | `process_entity_matching_response` | 多维度语义匹配推理 |

## 🔧 配置

主要配置在`main.py`中：
- `TARGET_USER`: 目标用户ID
- `INPUT_DIR`: 输入数据目录
- `OUTPUT_FILE`: 输出结果文件
- `PRODUCT_METADATA_FILE`: 商品元数据文件

## 📊 输出格式

最终输出包含：
- 商品基本信息
- 提取的商品实体
- 用户偏好实体
- 匹配的实体对
- 生成的自然语言查询

## 🛠️ 依赖

- Python 3.7+
- langchain相关包
- 自定义model模块

## 🎯 优势

1. **模块化**: 每个功能独立，便于测试和维护
2. **可扩展**: 新功能可以轻松添加
3. **并发处理**: 支持高并发实体提取
4. **错误恢复**: 完善的API密钥fallback机制
5. **智能匹配**: 基于语义的实体匹配算法