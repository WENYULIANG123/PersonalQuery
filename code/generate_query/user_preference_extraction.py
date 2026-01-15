#!/usr/bin/env python3
"""
用户偏好实体提取模块
负责处理用户偏好实体的提取和处理
"""

import os, json, gzip, sys
from typing import Dict, List, Optional, Union
from datetime import datetime
from langchain_core.language_models.chat_models import BaseChatModel

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import call_llm_with_retry, APIErrorException
from utils import get_all_api_keys_in_order, create_llm_with_config, try_api_keys_with_fallback

def log_with_timestamp(message: str):
    """Log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)

def clean_html_content(text) -> str:
    """Remove HTML tags and clean up content for entity extraction."""
    import re
    if not text:
        return ""

    # Convert to string if it's not already
    if not isinstance(text, str):
        text = str(text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove JavaScript content
    text = re.sub(r'javascript:[^\'"\\s]*', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove common Amazon UI text patterns
    text = re.sub(r'Save \d+% on.*?(?=when|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Enter code [A-Z0-9]+ at checkout', '', text, flags=re.IGNORECASE)
    text = re.sub(r'restrictions apply', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Here\'s how', '', text, flags=re.IGNORECASE)

    # Clean up extra spaces again
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def load_data_from_gzip(file_path: str, data_type: str, filter_func=None, max_items: int = None) -> Union[List[Dict], Dict[str, Dict]]:
    """从gzip文件加载数据"""
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            if data_type == 'metadata':
                # Load metadata as dict keyed by asin
                data = {}
                for line_num, line in enumerate(f):
                    if max_items and line_num >= max_items:
                        break
                    try:
                        item = json.loads(line.strip())
                        asin = item.get('asin', '')
                        if asin:
                            if filter_func is None or filter_func(item):
                                data[asin] = item
                    except json.JSONDecodeError:
                        continue
                return data
            else:
                # Load reviews as list
                data = []
                for line_num, line in enumerate(f):
                    if max_items and line_num >= max_items:
                        break
                    try:
                        item = json.loads(line.strip())
                        if filter_func is None or filter_func(item):
                            data.append(item)
                    except json.JSONDecodeError:
                        continue
                return data
    except Exception as e:
        log_with_timestamp(f"Error loading {data_type} from {file_path}: {e}")
        return {} if data_type == 'metadata' else []

def load_data(data_type: str, filter_func=None, max_items: int = None, user_products: set = None):
    """加载数据的通用函数"""
    if data_type == 'metadata':
        file_path = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz"
    elif data_type == 'reviews':
        file_path = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw/Arts_Crafts_and_Sewing.json.gz"
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    return load_data_from_gzip(file_path, data_type, filter_func, max_items)

TARGET_USER = "AG7EF0SVBQOUX"
REVIEW_FILE = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw/Arts_Crafts_and_Sewing.json.gz"
META_FILE = "/home/wlia0047/ar57/wenyu/data/Amazon-Reviews-2018/raw/meta_Arts_Crafts_and_Sewing.json.gz"
OUTPUT_FILE = "/home/wlia0047/ar57_scratch/wenyu/user_preference_queries.json"

def load_user_reviews(target_user: str = None) -> List[Dict]:
    """加载指定用户的所有评论（保持向后兼容）"""
    if target_user:
        def filter_func(data):
            user_id = data.get('user_id') or data.get('reviewerID') or data.get('reviewer_id')
            return user_id == target_user
        return load_data('reviews', filter_func, max_items=None)  # Remove max_items limit when filtering for specific user
    return load_data('reviews', max_items=100)

def process_user_preference_extraction_response(response_str: str) -> tuple:
    """
    处理用户偏好实体提取的LLM响应

    Args:
        response_str: LLM返回的原始字符串

    Returns:
        (flattened_entities_list, categorized_entities_dict) 元组

    Raises:
        APIErrorException: 当响应无效或无法解析时
    """
    if not response_str:
        raise APIErrorException("No response from user preference extraction")

    try:
        # Clean the response
        response_str = response_str.strip()

        # Smart JSON extraction for Chain of Thought responses
        lines = response_str.strip().split('\n')
        json_found = False

        # Check if the last few lines contain valid JSON
        for i in range(len(lines) - 1, max(-1, len(lines) - 5), -1):  # Check last 5 lines
            line = lines[i].strip()
            if line.startswith('{') and line.endswith('}'):
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
        result = json.loads(response_str)

        # Handle different response formats
        if isinstance(result, dict):
            # New format: {category: [entities]}
            categorized_entities = {}
            flattened = []

            for category, entities in result.items():
                if isinstance(entities, list):
                    # Process list of entities for this category
                    category_entities = []
                    for entity in entities:
                        if isinstance(entity, str):
                            entity_text = entity.strip()
                            if entity_text:
                                # Apply atomic filtering
                                entity_lower = entity_text.lower()
                                if not (',' in entity_text or
                                        ' and ' in entity_lower or
                                        ' with ' in entity_lower or
                                        ' or ' in entity_lower or
                                        ' for ' in entity_lower or
                                        '&' in entity_text):
                                    category_entities.append(entity_text)
                                    flattened.append(entity_text)

                    if category_entities:
                        categorized_entities[category] = category_entities
                elif isinstance(entities, str):
                    # Handle single entity as string
                    entity_text = entities.strip()
                    if entity_text:
                        entity_lower = entity_text.lower()
                        if not (',' in entity_text or
                                ' and ' in entity_lower or
                                ' with ' in entity_lower or
                                ' or ' in entity_lower or
                                ' for ' in entity_lower or
                                '&' in entity_text):
                            categorized_entities[category] = [entity_text]
                            flattened.append(entity_text)

            if flattened:
                return flattened, categorized_entities
            else:
                raise APIErrorException("No valid entities extracted from user preference extraction")

        elif isinstance(result, list):
            # Legacy format: array of strings - convert to new format
            categorized_entities = {"General": []}
            flattened = []

            for item in result:
                if isinstance(item, str) and item.strip():
                    entity_text = item.strip()
                    # Apply atomic filtering
                    entity_lower = entity_text.lower()
                    if not (',' in entity_text or
                            ' and ' in entity_lower or
                            ' with ' in entity_lower or
                            ' or ' in entity_lower or
                            ' for ' in entity_lower or
                            '&' in entity_text):
                        categorized_entities["General"].append(entity_text)
                        flattened.append(entity_text)

            if flattened:
                return flattened, categorized_entities
            else:
                raise APIErrorException("No valid entities extracted from user preference extraction")

        else:
            raise APIErrorException("Invalid result format from user preference extraction")

    except json.JSONDecodeError as e:
        print(f"JSON parsing error in user preference extraction: {e}", flush=True)
        raise APIErrorException("JSON parsing failed in user preference extraction")
    except Exception as e:
        print(f"Unexpected error processing user preference extraction response: {e}", flush=True)
        raise APIErrorException("Response processing failed in user preference extraction")

def prepare_content_and_extract_entities(data_source, data_type: str, llm_model, asin: str = None) -> List[str]:
    """通用函数：准备内容并提取实体

    Args:
        data_source: 数据源（产品信息字典或评论列表）
        data_type: 数据类型 ('product' 或 'user preference')
        llm_model: LLM模型
        asin: 产品ASIN（可选）

    Returns:
        提取的实体列表
    """
    if data_type in ['user_preference', 'user preference']:
        # 处理用户评论
        user_reviews = data_source
        if not user_reviews:
            raise APIErrorException("No user reviews available for preference extraction")

        # Combine all review content
        content_parts = []
        for review in user_reviews:
            text = review.get('reviewText', '').strip()
            title = review.get('summary', '').strip()

            if text or title:
                review_parts = []
                if title:
                    review_parts.append(f"Title: {title}")
                if text:
                    review_parts.append(f"Review: {text}")
                content_parts.append(' '.join(review_parts))

        content = ' '.join(content_parts)
        # Clean content
        content = clean_html_content(content)

        return extract_user_preference_entities(content, llm_model)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

def extract_user_preference_entities(content: str, llm_model) -> List[str]:
    """Extract user preference entities using LLM - business logic implementation with JSON parsing retry."""
    # Ensure content is not None or empty
    if not content or not isinstance(content, str):
        raise APIErrorException("Invalid content for user preference extraction")

    prompt = f"""Extract all entities mentioned in the following product review text and categorize them.
Include any products, activities, techniques, materials, tools, brands, or other relevant entities that the user mentions, regardless of whether they like them or not.

**实体分类要求:**
对于每个提取的实体，必须将其归类为以下类别之一：
[Brand, Material, Dimensions, Quantity, Color/Finish, Design, Usage, Selling Point, Safety/Certification, Accessories, Activity, Technique]

**输出格式:**
返回一个JSON对象，其中键是类别名称，值是该类别对应的实体数组。相同类别的多个实体应该放在同一个数组中。

示例:
{{
  "Brand": ["Apple", "Samsung"],
  "Design": ["smartphone", "waterproof"],
  "Selling Point": ["battery life", "camera quality"],
  "Usage": ["wireless charging", "fast charging"],
  "Color/Finish": ["black", "blue"]
}}

只返回有效的JSON对象，不要其他解释。

Text: {content}"""

    # Retry up to 5 times for JSON parsing errors
    json_parse_retries = 5
    for attempt in range(json_parse_retries):
        try:
            response_str, success = call_llm_with_retry(llm_model, prompt, context="user_preference_extraction")
            if success and response_str:
                entities_result = process_user_preference_extraction_response(response_str)
                # Handle tuple return format (list, dict)
                if isinstance(entities_result, tuple):
                    entities_list, entities_dict = entities_result
                    # Return the dict if available (new format), otherwise return the list
                    return entities_dict if entities_dict else entities_list
                else:
                    # Backward compatibility
                    return entities_result

            # If we get here, no valid entities were extracted - treat as failure
            raise APIErrorException("No valid entities extracted from user preference")
        except APIErrorException as e:
            # Check if this is a JSON parsing error
            error_msg = str(e)
            if "JSON parsing failed" in error_msg or "JSON parsing error" in error_msg:
                if attempt < json_parse_retries - 1:
                    print(f"JSON parsing failed (attempt {attempt + 1}/{json_parse_retries}), retrying...", flush=True)
                    continue
                else:
                    print(f"JSON parsing failed after {json_parse_retries} attempts", flush=True)
            # Re-raise API errors (including JSON parsing errors after retries)
            raise
        except Exception as e:
            # Let the caller handle other API errors - they will trigger key switching
            raise