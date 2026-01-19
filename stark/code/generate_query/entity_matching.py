#!/usr/bin/env python3
"""
实体匹配模块
负责匹配产品实体和用户偏好实体
"""

import os
import json
import sys
from typing import Dict, List
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import call_llm_with_retry, APIErrorException, ApiProvider
from utils import get_all_api_keys_in_order, create_llm_with_config, try_api_keys_with_fallback

def log_with_timestamp(message: str):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def create_llm_with_config(api_config):
    """Create LLM with config based on provider."""
    from langchain_openai import ChatOpenAI

    provider = api_config.get('provider', 'siliconflow')

    if provider == 'siliconflow':
        return ChatOpenAI(
            base_url="https://api.siliconflow.cn/v1",
            api_key=api_config['api_key'],
            model_name=api_config.get('model', 'Qwen/Qwen2.5-7B-Instruct'),
            temperature=0.1,
            max_tokens=4000,
            timeout=60
        )
    else:
        # Default to OpenAI
        return ChatOpenAI(
            api_key=api_config['api_key'],
            model_name=api_config.get('model', 'gpt-3.5-turbo'),
            temperature=0.1,
            max_tokens=4000,
            timeout=60
        )

def try_api_keys_with_fallback(api_keys: List[Dict], operation_func, context: str, success_message: str = None, error_message: str = None):
    """
    通用API key循环重试函数

    Args:
        api_keys: API key配置列表
        operation_func: 要执行的操作函数，参数为(api_config, provider_name, key_index)
        context: 上下文信息，用于日志
        success_message: 成功时的日志消息模板
        error_message: 错误时的日志消息模板

    Returns:
        (result, success) 元组，result是操作结果，success表示是否成功
    """
    for key_index, api_config in enumerate(api_keys):
        provider_name = "SiliconFlow" if api_config['provider'] == ApiProvider.SILICONFLOW else "Unknown"
        try:
            result = operation_func(api_config, provider_name, key_index)

            # 成功处理
            # if success_message:
            #     log_with_timestamp(success_message.format(
            #         context=context,
            #         provider=provider_name,
            #         key_num=api_config['key_index'] + 1,
            #     ))
            return result, True
        except APIErrorException as e:
            # API错误，继续下一个key
            if error_message:
                log_with_timestamp(error_message.format(
                    context=context,
                    provider=provider_name,
                    key_num=api_config['key_index'] + 1,
                    error=str(e)
                ))
            continue
        except Exception as e:
            # 其他错误，继续下一个key
            log_with_timestamp(f"❌ Unexpected error with {provider_name} Key #{api_config['key_index'] + 1}: {e}")
            continue

    # 所有key都失败了
    return None, False


def process_entity_matching_response(response_str: str) -> List[str]:
    """
    处理实体匹配的LLM响应

    Args:
        response_str: LLM返回的原始字符串

    Returns:
        处理后的实体列表

    Raises:
        APIErrorException: 当响应无效或无法解析时
    """
    # Debug: print raw response
    print(f"🔍 Entity matching raw response (first 500 chars): {response_str[:500]!r}", flush=True)

    # Check for markdown code blocks
    if response_str.startswith('```') and '```' in response_str:
        print("📦 Found markdown code block, extracting JSON...", flush=True)

    if not response_str:
        raise APIErrorException("No response from entity matching")

    try:
        # Clean the response
        response_str = response_str.strip()

        # Smart JSON extraction for Chain of Thought responses
        lines = response_str.strip().split('\n')
        json_found = False

        # Check if the last few lines contain valid JSON
        for i in range(len(lines) - 1, max(-1, len(lines) - 5), -1):  # Check last 5 lines
            line = lines[i].strip()
            if line.startswith('[') and line.endswith(']'):
                # Found JSON array at the end
                response_str = line
                json_found = True
                break
            elif line.startswith('{') and line.endswith('}'):
                # Found JSON object at the end
                response_str = line
                json_found = True
                break

        # If no JSON found at the end, look for code blocks
        if not json_found:
            # Find the LAST json code block (in case there are multiple)
            json_blocks = []
            start = 0
            while True:
                json_start = response_str.find('```json', start)
                if json_start == -1:
                    break
                json_end = response_str.find('```', json_start + 7)
                if json_end == -1:
                    break
                content_start = response_str.find('\n', json_start) + 1
                if content_start > 0:
                    content = response_str[content_start:json_end].strip()
                    if content:
                        json_blocks.append(content)
                start = json_end + 3

            # Also handle regular ``` blocks
            if not json_blocks:
                if '```' in response_str:
                    # Find the LAST code block
                    last_triple = response_str.rfind('```')
                    first_triple = response_str.rfind('```', 0, last_triple)
                    if first_triple != last_triple:
                        content_start = response_str.find('\n', first_triple) + 1
                        if content_start > 0:
                            response_str = response_str[content_start:last_triple].strip()
                    else:
                        # Single code block
                        content_start = response_str.find('\n', first_triple) + 1
                        if content_start > 0:
                            response_str = response_str[content_start:].strip()
                elif json_blocks:
                    response_str = json_blocks[-1]  # Use the last json block

        # Try to parse as JSON
        print(f"🔄 Attempting to parse JSON: {response_str[:200]!r}", flush=True)
        result = json.loads(response_str)
        print(f"✅ JSON parsed successfully: {result}", flush=True)

        # Handle different response formats
        if isinstance(result, list):
            # Array format - expected for entity matching
            flattened = []
            for item in result:
                if isinstance(item, str):
                    flattened.append(item)
                elif isinstance(item, list):
                    # Flatten nested list but only take string elements
                    for subitem in item:
                        if isinstance(subitem, str):
                            flattened.append(subitem)

            if flattened:
                return flattened
            else:
                raise APIErrorException("No valid entities extracted from entity matching (empty result)")

        elif isinstance(result, dict):
            # If somehow returns dict, try to extract matched entities
            flattened = []
            possible_keys = ["matched_entities", "matches", "results"]
            for key in possible_keys:
                if key in result and isinstance(result[key], list):
                    for item in result[key]:
                        if isinstance(item, str) and item.strip():
                            flattened.append(item.strip())

            if flattened:
                return flattened
            else:
                raise APIErrorException("No valid entities extracted from entity matching (empty result)")

        else:
            raise APIErrorException("Invalid result format from entity matching")

    except json.JSONDecodeError as e:
        print(f"JSON parsing error in entity matching: {e}", flush=True)
        raise APIErrorException("JSON parsing failed in entity matching")
    except Exception as e:
        print(f"Unexpected error processing entity matching response: {e}", flush=True)
        raise APIErrorException("Response processing failed in entity matching")

def match_product_and_user_entities_no_llm(product_entities: Dict[str, List[str]], user_entities: Dict[str, List[str]], llm_model) -> Dict[str, List[str]]:
    """
    使用LLM进行实体匹配：对每个用户偏好实体，在相同类别的产品实体中找到相似度最大的实体

    Args:
        product_entities: 商品实体字典 {category: [entities]}
        user_entities: 用户偏好实体字典 {category: [entities]}
        llm_model: LLM模型用于计算相似度

    Returns:
        匹配的实体字典 {category: [matched_entities]}
    """
    matched_entities = {}

    # 遍历用户偏好实体的所有类别
    for user_category, user_entity_list in user_entities.items():
        # 如果产品实体中也存在这个类别
        if user_category in product_entities:
            product_entity_list = product_entities[user_category]
            matched_in_category = []

            # 对每个用户偏好实体，在产品实体中找到最相似的
            matched_product_entities = set()  # 用于去重匹配的产品实体
            for user_entity in user_entity_list:
                best_match = find_most_similar_entity_with_llm(user_entity, product_entity_list, llm_model)
                if best_match and best_match not in matched_product_entities:
                    matched_in_category.append(best_match)
                    matched_product_entities.add(best_match)

            if matched_in_category:
                matched_entities[user_category] = matched_in_category

    return matched_entities

def find_most_similar_entity_with_llm(user_entity: str, product_entities: List[str], llm_model) -> str:
    """
    使用LLM在产品实体列表中找到与用户实体最相似的实体

    Args:
        user_entity: 用户偏好实体
        product_entities: 产品实体列表
        llm_model: LLM模型

    Returns:
        最相似的产品实体，如果没有找到则返回空字符串
    """
    if not product_entities:
        return ""

    # 如果只有一个产品实体，直接返回
    if len(product_entities) == 1:
        return product_entities[0]

    prompt = f"""
You are an expert at finding semantic similarity between product features.

Given:
- User preference entity: "{user_entity}"
- Product entities to compare: {product_entities}

Find the product entity that is most semantically similar to the user preference entity.
Consider synonyms, related concepts, and contextual similarity.

**OUTPUT REQUIREMENT:**
Return ONLY the most similar product entity as a JSON string. No explanations.

Example:
- User: "24 colors" → Product entities: ["24", "12", "36"] → Output: "24"
- User: "waterproof" → Product entities: ["water resistant", "durable", "lightweight"] → Output: "water resistant"

Output format:
"most_similar_entity"
"""

    # Retry up to 3 times
    for attempt in range(3):
        try:
            response_str, success = call_llm_with_retry(llm_model, prompt, context="entity_similarity")
            if success and response_str:
                # 尝试解析JSON字符串
                try:
                    # 移除可能的引号包装
                    if response_str.startswith('"') and response_str.endswith('"'):
                        result = response_str[1:-1]
                    else:
                        result = response_str.strip()

                    # 检查结果是否在产品实体列表中
                    if result in product_entities:
                        return result
                    else:
                        # 如果不在列表中，尝试找到最相似的
                        for product_entity in product_entities:
                            if result.lower() in product_entity.lower() or product_entity.lower() in result.lower():
                                return product_entity

                except Exception as e:
                    print(f"Error parsing LLM response for similarity: {e}", flush=True)

        except Exception as e:
            print(f"LLM error in entity similarity: {e}", flush=True)
            if attempt < 2:  # 不是最后一次尝试
                continue

    # 如果LLM失败，返回空字符串
    return ""

def match_product_and_user_entities(product_entities: List[str], user_entities: List[str], llm_model) -> List[str]:
    """使用LLM匹配产品实体和用户偏好实体，找出匹配的实体"""
    if not product_entities or not user_entities:
        return []

    # 简化的实体匹配prompt，直接要求JSON输出
    prompt = f"""
You are an expert at matching product features with user preferences.

Given:
- Product Entities: {product_entities}
- User Preferences: {user_entities}

Find entities that appear in both lists OR are semantically equivalent (synonyms or closely related).

**OUTPUT REQUIREMENT:**
Return ONLY a JSON array of matched entities. No explanations.

Examples:
- If "color" appears in both lists → ["color"]
- If "size" in products and "dimensions" in user preferences → ["size"]
- If no matches → []

```json

```json
[]
```

Begin your analysis now.
"""

    # Retry up to 5 times for JSON parsing errors in matching
    json_parse_retries = 5
    for attempt in range(json_parse_retries):
        try:
            response_str, success = call_llm_with_retry(llm_model, prompt, context="entity_matching")
            if success and response_str:
                entities = process_entity_matching_response(response_str)

                # Filter to ensure only strings and remove duplicates (specific to matching)
                matched_entities = []
                for item in entities:
                    if isinstance(item, str) and item.strip():
                        clean_item = item.strip()
                        if clean_item not in matched_entities:
                            matched_entities.append(clean_item)

                return matched_entities
        except APIErrorException as e:
            # Check if this is a JSON parsing error
            error_msg = str(e)
            if "JSON parsing failed" in error_msg or "JSON parsing error" in error_msg:
                if attempt < json_parse_retries - 1:
                    print(f"JSON parsing failed in matching (attempt {attempt + 1}/{json_parse_retries}), retrying...", flush=True)
                    continue
                else:
                    print(f"JSON parsing failed in matching after {json_parse_retries} attempts", flush=True)
            # For matching, we return empty list on error instead of raising
            return []
        except Exception as e:
            print(f"LLM error in entity matching: {e}", flush=True)
            raise  # 重新抛出异常，让API key循环处理

    return []


def perform_entity_matching(products: List[Dict], max_workers: int = 20) -> List[Dict]:
    """执行产品实体和用户偏好实体的匹配"""
    log_with_timestamp(f"🔗 Starting entity matching for {len(products)} products...")

    if not products:
        log_with_timestamp("⚠️ No products found for matching")
        return products

    # 获取API keys用于LLM匹配
    all_api_keys = get_all_api_keys_in_order()

    matched_count = 0
    total_products = len(products)

    for idx, product in enumerate(products):
        try:
            asin = product.get('asin', 'Unknown')
            product_entities = product.get('product_entities', {})
            user_entities = product.get('user_preference_entities', {})

            # 使用LLM进行实体相似度匹配
            def matching_operation(api_config, provider_name, key_index):
                llm_model = create_llm_with_config(api_config)
                return match_product_and_user_entities_no_llm(product_entities, user_entities, llm_model)

            matched_entities, success = try_api_keys_with_fallback(
                all_api_keys,
                matching_operation,
                f"{asin} entity matching"
            )

            if not success:
                matched_entities = {}

            # 检查是否有匹配的实体
            has_matches = any(matches for matches in matched_entities.values())

            # 添加匹配结果到产品数据
            product['matched_entities'] = matched_entities

            # 生成格式化的输出字符串
            formatted_output = generate_formatted_product_output(product, idx, total_products)
            product['formatted_output'] = formatted_output

            if has_matches:
                matched_count += 1

            # 每处理10个产品或最后一批时输出进度
            if (idx + 1) % 10 == 0 or idx + 1 == total_products:
                log_with_timestamp(f'📊 Entity matching progress: {idx + 1}/{total_products} products processed')

        except Exception as e:
            log_with_timestamp(f'❌ Exception in entity matching for {asin}: {e}')
            product['matched_entities'] = {}
            product['formatted_output'] = generate_formatted_product_output(product, idx, total_products)

    log_with_timestamp(f'✅ Entity matching completed! {matched_count}/{total_products} products have matched entities')
    return products



def generate_formatted_product_output(product, idx, total_products):
    """生成格式化的产品输出字符串"""
    asin = product.get('asin', 'Unknown')
    product_title = product.get('product_title', 'Unknown Product')
    product_entities = product.get('product_entities', [])
    user_entities = product.get('user_preference_entities', [])
    matched_entities = product.get('matched_entities', [])

    output_lines = [
        f"[{idx+1}/{total_products}] Product: {product_title}",
        f"ASIN: {asin}",
        f"Product Entities ({len(product_entities)}): {', '.join(product_entities) if product_entities else 'None'}",
        f"User Preference Entities ({len(user_entities)}): {', '.join(user_entities) if user_entities else 'None'}",
        f"Matched Entities ({len(matched_entities)}): {', '.join(matched_entities) if matched_entities else 'None'}",
        ""
    ]

    return "\n".join(output_lines)