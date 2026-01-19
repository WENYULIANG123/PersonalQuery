#!/usr/bin/env python3
"""
主函数文件
整合商品实体提取、用户偏好实体提取和实体匹配模块
"""

import os
import json
import sys
import threading
import concurrent.futures

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import get_all_api_keys_in_order

# Import modules
from product_extraction import (
    log_with_timestamp, clean_html_content, load_product_metadata, extract_product_entities_only
)

from user_preference_extraction import (
    load_user_reviews, prepare_content_and_extract_entities, TARGET_USER, OUTPUT_FILE
)

from utils import (
    try_api_keys_with_fallback, create_llm_with_config
)

from entity_matching import (
    perform_entity_matching
)

from query_generation import generate_queries_for_matched_products

def report_progress(current, total, report_interval=10, message_template="📊 Progress: {current} / {total} {unit} processed"):
    """
    通用进度报告函数

    Args:
        current: 当前进度
        total: 总数
        report_interval: 报告间隔
        message_template: 消息模板
    """
    if current % report_interval == 0 or current == total:
        log_with_timestamp(message_template.format(current=current, total=total, unit="products"))

def print_entity_matching_results():
    """打印实体匹配的完整结果"""
    log_with_timestamp("📋 Printing complete entity matching results...")

    try:
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        log_with_timestamp(f"❌ Error reading results file for printing: {e}")
        return

    products = data.get('products', [])
    if not products:
        log_with_timestamp("⚠️ No products found in results file")
        return

    print(f'\\n📋 Complete Entity Matching Results ({len(products)} products):', flush=True)
    print('=' * 90, flush=True)

    sorted_products = sorted(products, key=lambda x: x.get('asin', ''))

    for idx, product in enumerate(sorted_products, 1):
        asin = product.get('asin', 'Unknown')
        product_title = product.get('product_title', 'Unknown')
        product_entities = product.get('product_entities', {})
        user_entities = product.get('user_preference_entities', {})
        matched_entities = product.get('matched_entities', {})
        reviews = product.get('reviews', [])
        metadata = product.get('metadata', {})

        # 打印产品信息
        progress_info = f" ({idx}/{len(products)})"
        print(f'Product {asin} ({product_title[:50]}...){progress_info}:', flush=True)

        # 打印review内容
        if reviews:
            # 去重review内容
            unique_reviews = []
            seen_contents = set()
            for review in reviews:
                title = review.get('summary', '').strip()
                text = review.get('reviewText', '').strip()
                # 移除文本中的换行符，用空格替换
                text = ' '.join(text.split())
                review_content = f"{title} {text}".strip()
                if review_content and review_content not in seen_contents:
                    seen_contents.add(review_content)
                    unique_reviews.append(review_content)

            print(f'  Reviews ({len(unique_reviews)} unique):', flush=True)
            for i, review_content in enumerate(unique_reviews[:3], 1):  # 只显示前3个unique review
                if review_content:
                    print(f'    Review {i}: {review_content}', flush=True)
            if len(unique_reviews) > 3:
                print(f'    ... and {len(unique_reviews) - 3} more unique reviews', flush=True)
        else:
            print('  Reviews: None found', flush=True)

        # 打印产品实体
        if product_entities:
            total_product_entities = sum(len(entities) for entities in product_entities.values())
            print(f'  Product Entities ({len(product_entities)} categories, {total_product_entities} total):', flush=True)
            for category, entities in product_entities.items():
                print(f'    {category}: {", ".join(entities)}', flush=True)
        else:
            print('  Product Entities: None extracted', flush=True)

        # 打印用户偏好实体
        if user_entities:
            total_user_entities = sum(len(entities) for entities in user_entities.values())
            print(f'  User Preference Entities ({len(user_entities)} categories, {total_user_entities} total):', flush=True)
            for category, entities in user_entities.items():
                print(f'    {category}: {", ".join(entities)}', flush=True)
        else:
            print('  User Preference Entities: None extracted', flush=True)

        # 打印匹配实体
        if matched_entities:
            total_matched = sum(len(entities) for entities in matched_entities.values())
            print(f'  Matched Entities ({len(matched_entities)} categories, {total_matched} total):', flush=True)
            for category, entities in matched_entities.items():
                print(f'    {category}: {", ".join(entities)}', flush=True)
        else:
            print('  Matched Entities: No matches found', flush=True)

        # 打印生成的查询
        generated_query = product.get('generated_query', '')
        if generated_query:
            print(f'  Generated Query: {generated_query}', flush=True)
        else:
            print('  Generated Query: None generated', flush=True)

        # 打印metadata
        print('  Metadata:', flush=True)
        for key, value in metadata.items():
            print(f'    {key}: {value}', flush=True)
        print()

    failed_products = [p for p in products if not p.get('matched_entities') or not any(matches for matches in p.get('matched_entities', {}).values())]
    if failed_products:
        print(f'\\n❌ Products with No Matches ({len(failed_products)}):', flush=True)
        for product in failed_products:
            asin = product.get('asin', 'Unknown')
            product_entities = product.get('product_entities', {})
            user_entities = product.get('user_preference_entities', {})
            product_count = sum(len(entities) for entities in product_entities.values())
            user_count = sum(len(entities) for entities in user_entities.values())
            print(f'  Product {asin}: Product entities ({len(product_entities)} categories, {product_count} total), User entities ({len(user_entities)} categories, {user_count} total)', flush=True)

def main():
    log_with_timestamp('Starting product entity extraction with API key fallback...')

    user_reviews = load_user_reviews(TARGET_USER)

    if not user_reviews:
        log_with_timestamp(f"❌ No reviews found for user {TARGET_USER}")
        return

    log_with_timestamp(f"✅ Found {len(user_reviews)} reviews for user {TARGET_USER}")

    user_asins = set(review.get('asin') for review in user_reviews if review.get('asin'))
    product_metadata = load_product_metadata(user_asins)

    if not product_metadata:
        log_with_timestamp(f"❌ No product metadata found for user {TARGET_USER}'s reviewed products")
        return

    log_with_timestamp(f"✅ Found metadata for {len(product_metadata)} products reviewed by user {TARGET_USER}")

    log_with_timestamp(f'Extracting product entities for {len(product_metadata)} products...')

    all_api_keys = get_all_api_keys_in_order()

    all_asins = list(product_metadata.keys())

    if not all_asins:
        log_with_timestamp("❌ No products found")
        return

    reviewed_asins = set()
    reviews_by_asin = {}
    for review in user_reviews:
        asin = review.get('asin')
        if asin and asin in product_metadata:
            reviewed_asins.add(asin)
            if asin not in reviews_by_asin:
                reviews_by_asin[asin] = []
            reviews_by_asin[asin].append(review)

    all_asins = sorted(list(reviewed_asins))
    total_products = len(all_asins)
    log_with_timestamp(f'🔍 Selected ASINs for processing: {all_asins}')

    if not all_asins:
        log_with_timestamp("❌ No reviewed products found")
        return

    all_results = []

    log_with_timestamp(f'🔄 Processing {len(all_asins)} products concurrently with 5 workers...')

    progress_counter = {'completed': 0}
    progress_lock = threading.Lock()

    def process_single_product(asin):
        try:
            # Get API keys for processing
            ordered_keys = get_all_api_keys_in_order()

            def extract_operation(api_config, provider_name, key_index):
                result = extract_product_entities_only(asin, product_metadata, api_config, total_products)
                if result and result.get('success', False):
                    return result
                else:
                    error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
                    raise Exception(f"Extraction failed: {error_msg}")

            product_result, success = try_api_keys_with_fallback(
                ordered_keys,
                extract_operation,
                f"product {asin}",
                "✅ Successfully processed {context} with {provider} Key #{key_num}"
            )

            if success:
                result = product_result
            else:
                result = {
                    'asin': asin,
                    'error': 'All API keys failed',
                    'success': False
                }

            # Update progress
            with progress_lock:
                progress_counter['completed'] += 1
                current_count = progress_counter['completed']
                if current_count % 10 == 0 or current_count == total_products:
                    log_with_timestamp(f'📊 Progress: {current_count}/{total_products} products processed')

            return result

        except Exception as e:
            log_with_timestamp(f'❌ Error processing product {asin}: {e}')
            return {
                'asin': asin,
                'error': str(e),
                'success': False
            }

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        future_to_asin = {executor.submit(process_single_product, asin): asin for asin in all_asins}

        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_asin):
            asin = future_to_asin[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                log_with_timestamp(f'❌ Exception processing {asin}: {e}')
                all_results.append({
                    'asin': asin,
                    'error': str(e),
                    'success': False
                })

    successful_count = len([r for r in all_results if r.get('success', False)])
    if successful_count == len(all_asins):
        log_with_timestamp('✅ All products processed successfully!')
    else:
        log_with_timestamp(f'⚠️  {successful_count}/{len(all_asins)} products processed successfully, {len(all_asins) - successful_count} failed')


    log_with_timestamp(f'🔍 Found reviews for {len(reviews_by_asin)} products')
    log_with_timestamp(f'🔍 Reviews by ASIN keys: {sorted(list(reviews_by_asin.keys()))[:10]}...')

    total_reviews_to_process = sum(len(reviews) for reviews in reviews_by_asin.values())
    log_with_timestamp(f'📊 Total user preference reviews to process: {total_reviews_to_process}')


    product_user_entities_map = {}  # asin -> user_entities

    log_with_timestamp(f'🔍 Starting user preference entity extraction for {len(all_results)} products concurrently...')

    successful_products = [result for result in all_results if 'error' not in result]

    user_pref_progress_counter = {'completed_reviews': 0}
    user_pref_progress_lock = threading.Lock()

    log_with_timestamp(f'🔍 Processing {len(successful_products)} successful products: {[p["asin"] for p in successful_products[:3]]}...')

    def process_user_preferences(result):
        asin = result['asin']
        log_with_timestamp(f'🔍 Starting user preference processing for {asin}')
        product_user_entities = []

        try:
            # Rebuild reviews_by_asin for this ASIN to avoid concurrency issues
            product_reviews = [r for r in user_reviews if r.get('asin') == asin]
            log_with_timestamp(f'🔍 Found {len(product_reviews)} reviews for {asin}')
            if product_reviews:
                # Check if reviews have actual content
                valid_reviews = []
                for review in product_reviews:
                    text = review.get('reviewText', '').strip()
                    title = review.get('summary', '').strip()
                    if text or title:
                        valid_reviews.append(review)

                if not valid_reviews:
                    log_with_timestamp(f'⚠️ No valid content in {len(product_reviews)} reviews for {asin}')
                    product_user_entities_map[asin] = []
                    with user_pref_progress_lock:
                        user_pref_progress_counter['completed_reviews'] += len(product_reviews)
                        current_reviews = user_pref_progress_counter['completed_reviews']
                        if current_reviews % 100 == 0 or current_reviews == total_reviews_to_process:
                            log_with_timestamp(f'📊 User preference progress: {current_reviews}/{total_reviews_to_process} reviews processed')
                    return asin, []

            # Get API keys for processing
            ordered_keys = all_api_keys
            log_with_timestamp(f'🔍 Using {len(ordered_keys)} API keys for {asin}')

            def preference_operation(api_config, provider_name, key_index):
                llm_model = create_llm_with_config(api_config)
                return prepare_content_and_extract_entities(valid_reviews, 'user preference', llm_model)

            try:
                raw_result = try_api_keys_with_fallback(
                    ordered_keys,
                    preference_operation,
                    f"{asin} user preference extraction"
                )
                log_with_timestamp(f'🔍 Raw result for {asin}: {raw_result} (type: {type(raw_result)})')
                if raw_result is None:
                    log_with_timestamp(f'❌ try_api_keys_with_fallback returned None for {asin}')
                    product_user_entities, success = {}, False
                elif isinstance(raw_result, tuple) and len(raw_result) == 2:
                    # Check what the tuple contains
                    first_elem, second_elem = raw_result
                    if isinstance(first_elem, dict) and isinstance(second_elem, bool):
                        # Format: ({category: [entities]}, success_bool)
                        product_user_entities = first_elem  # The dict
                        success = second_elem  # The bool
                    elif isinstance(first_elem, (list, dict)) and isinstance(second_elem, (list, dict)):
                        # Format: (list, dict) - old format
                        entities_list, entities_dict = first_elem, second_elem
                        product_user_entities = entities_dict if entities_dict else entities_list
                        success = True
                    else:
                        log_with_timestamp(f'❌ Unexpected tuple content: {type(first_elem)}, {type(second_elem)}')
                        product_user_entities, success = {}, False
                elif isinstance(raw_result, dict):
                    # Direct dict format
                    product_user_entities = raw_result
                    success = True
                elif isinstance(raw_result, list):
                    # Legacy list format
                    product_user_entities = raw_result
                    success = True
                else:
                    log_with_timestamp(f'❌ Unexpected result type from try_api_keys_with_fallback: {raw_result}')
                    product_user_entities, success = {}, False
            except Exception as api_error:
                log_with_timestamp(f'❌ Exception in user preference extraction for {asin}: {api_error}')
                product_user_entities, success = [], False

            if not success:
                product_user_entities = []
                user_entity_explanations = {}
            else:
                # Skip generating explanations for user preference entities
                user_entity_explanations = {}

            # Update global progress
            with user_pref_progress_lock:
                user_pref_progress_counter['completed_reviews'] += len(valid_reviews)
                current_reviews = user_pref_progress_counter['completed_reviews']
                if current_reviews % 100 == 0 or current_reviews == total_reviews_to_process:
                    log_with_timestamp(f'📊 User preference progress: {current_reviews}/{total_reviews_to_process} reviews processed')

            log_with_timestamp(f'✅ Completed user preference extraction for {asin}: {len(product_user_entities)} entities')

            # Skip generating explanations for user preference entities
            user_entity_explanations = {}

            return asin, product_user_entities, user_entity_explanations, valid_reviews

        except Exception as e:
            log_with_timestamp(f'❌ Error processing user preferences for {asin}: {e}')
            return asin, [], {}, []

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Submit all tasks
        future_to_result = {executor.submit(process_user_preferences, result): result for result in successful_products}

        # Collect results as they complete
        user_preferences_data = []
        for future in concurrent.futures.as_completed(future_to_result):
            result = future_to_result[future]
            try:
                future_result = future.result()
                if len(future_result) == 4:
                    asin, user_entities, user_explanations, review_content = future_result
                elif len(future_result) == 2:
                    asin, user_entities = future_result
                    review_content = []
                else:
                    raise ValueError(f"Unexpected result format: {future_result}")

                product_user_entities_map[asin] = user_entities

                # Store user preference data - now with categorized entities
                user_pref_item = {
                    'asin': asin,
                    'user_preference_entities': user_entities,
                    'review_content': review_content
                }
                user_preferences_data.append(user_pref_item)
            except Exception as e:
                asin = result['asin']
                log_with_timestamp(f'❌ Exception processing user preferences for {asin}: {e}')
                product_user_entities_map[asin] = []

    log_with_timestamp(f'✅ Completed entity extraction for {len(all_results)} products.')

    log_with_timestamp('💾 Saving extracted entity data...')

    # Get the workspace root directory (parent of stark directory)
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    result_dir = os.path.join(workspace_root, "result")
    os.makedirs(result_dir, exist_ok=True)
    
    product_entities_file = os.path.join(result_dir, "product_entities.json")
    product_entities_data = {
        'user_id': TARGET_USER,
        'products': []
    }

    def filter_product_info(product_info):
        relevant_fields = {
            'title': 'title',
            'brand': 'brand',
            'description': 'description',
            'feature': 'feature',
            'category': 'category',
            'main_cat': 'main_cat'
        }
        filtered = {}
        for key, new_key in relevant_fields.items():
            if key in product_info:
                value = product_info[key]
                # 对文本字段进行HTML清洗
                if key in ['description', 'feature'] and isinstance(value, list):
                    # 如果是列表，对每个元素进行清洗
                    filtered[new_key] = [clean_html_content(str(item)) for item in value if clean_html_content(str(item))]
                elif isinstance(value, str):
                    # 如果是字符串，直接清洗
                    filtered[new_key] = clean_html_content(value)
                else:
                    # 其他类型保持不变
                    filtered[new_key] = value
        return filtered

    for result in successful_products:
        asin = result['asin']

        # 处理产品实体，支持新格式（字典）和旧格式（列表）
        product_entities = result['product_entities']
        if isinstance(product_entities, dict):
            # 新格式：{category: [entities]}
            # 展平所有实体用于后续处理
            cleaned_entities = []
            for category_entities in product_entities.values():
                if isinstance(category_entities, list):
                    cleaned_entities.extend(category_entities)
                else:
                    cleaned_entities.append(str(category_entities))
        else:
            # 旧格式：实体列表
            # 过滤产品实体，移除包含类别前缀的实体
            cleaned_entities = []
            for entity in product_entities:
                # 如果实体包含冒号，提取冒号后的部分
                if ':' in entity and len(entity.split(':', 1)) == 2:
                    prefix, value = entity.split(':', 1)
                    cleaned_entity = value.strip()
                    if cleaned_entity:  # 确保不为空
                        cleaned_entities.append(cleaned_entity)
                else:
                    cleaned_entities.append(entity)

        # 应用原子化过滤
        atomic_entities = []
        for entity in cleaned_entities:
            entity_lower = entity.lower()
            if (',' in entity or
                ' and ' in entity_lower or
                ' with ' in entity_lower or
                ' or ' in entity_lower or
                ' for ' in entity_lower or
                '&' in entity):
                continue
            atomic_entities.append(entity)

        product_info = result['product_info']
        product_info_filtered = filter_product_info(product_info)

        # 保存字典格式的product_entities
        if isinstance(product_entities, dict):
            # 新格式：{category: [entities]} - 直接使用
            saved_product_entities = product_entities
        else:
            # 旧格式：实体列表 - 保持原样用于向后兼容
            saved_product_entities = product_entities

        product_data = {
            'asin': asin,
            'product_title': result['product_title'],
            'product_entities': saved_product_entities,
            'product_info': product_info_filtered,
            'metadata': {}
        }

        # 解析metadata_lines添加到metadata中
        for line in result['metadata_lines']:
            if line.startswith('    '):
                line = line.strip()
                if ': ' in line:
                    key, value = line.split(': ', 1)
                    product_data['metadata'][key.lower()] = value

        product_entities_data['products'].append(product_data)

    with open(product_entities_file, 'w', encoding='utf-8') as f:
        json.dump(product_entities_data, f, indent=2, ensure_ascii=False)
    log_with_timestamp(f'💾 Saved product entities to {product_entities_file}')

    # 保存用户偏好实体数据到新的JSON文件
    user_preferences_file = os.path.join(result_dir, "user_preference_entities.json")
    try:
        user_pref_save_data = {
            'user_id': TARGET_USER,
            'products': user_preferences_data
        }
        with open(user_preferences_file, 'w', encoding='utf-8') as f:
            json.dump(user_pref_save_data, f, indent=2, ensure_ascii=False)
        log_with_timestamp(f'💾 Saved user preference entities to {user_preferences_file}')
    except Exception as e:
        log_with_timestamp(f'❌ Error saving user preference entities: {e}')

    log_with_timestamp('✅ Product entity extraction completed. Proceeding to entity matching...')

    # 准备商品数据用于实体匹配
    save_data = {
        'user_id': TARGET_USER,
        'products': product_entities_data['products']  # 使用已提取的商品数据
    }

    log_with_timestamp('🎯 Entity extraction phase finished.')

    # 加载用户偏好数据并合并到商品数据中
    # user_preferences_file already defined above
    try:
        with open(user_preferences_file, 'r', encoding='utf-8') as f:
            user_pref_data = json.load(f)

        # 创建用户偏好数据的映射 {asin: user_entities}
        user_entities_map = {}
        for product in user_pref_data.get('products', []):
            asin = product.get('asin')
            user_entities = product.get('user_preference_entities', {})
            if asin:
                user_entities_map[asin] = user_entities

        # 将用户偏好数据合并到商品数据中
        for product in save_data['products']:
            asin = product.get('asin')
            if asin in user_entities_map:
                product['user_preference_entities'] = user_entities_map[asin]

        log_with_timestamp(f'✅ Loaded user preference data for {len(user_entities_map)} products')

    except Exception as e:
        log_with_timestamp(f'❌ Error loading user preference data: {e}')
        # 如果加载失败，继续处理，但没有用户偏好数据

    matched_products = perform_entity_matching(save_data['products'])

    # 保存实体匹配结果到新文件
    matched_entities_file = os.path.join(result_dir, "entity_matching_results.json")
    try:
        matched_data = {
            'user_id': TARGET_USER,
            'products': matched_products
        }
        with open(matched_entities_file, 'w', encoding='utf-8') as f:
            json.dump(matched_data, f, indent=2, ensure_ascii=False)
        log_with_timestamp(f'💾 Saved matched entity data to {matched_entities_file}')
    except Exception as e:
        log_with_timestamp(f'❌ Error saving matched entity data: {e}')

    # 为匹配实体类别大于等于3个的产品生成查询语句
    try:
        products_with_queries = generate_queries_for_matched_products(matched_data, get_all_api_keys_in_order())

        # 保存生成的查询到单独的文件
        if products_with_queries:
            generated_queries_file = os.path.join(result_dir, "generated_queries.json")
            queries_data = {
                'user_id': TARGET_USER,
                'products': products_with_queries
            }
            with open(generated_queries_file, 'w', encoding='utf-8') as f:
                json.dump(queries_data, f, indent=2, ensure_ascii=False)
            log_with_timestamp(f'💾 Saved generated queries for {len(products_with_queries)} products to {generated_queries_file}')
        else:
            log_with_timestamp('⚠️ No queries were generated')
    except Exception as e:
        log_with_timestamp(f'❌ Error generating queries: {e}')

    print_entity_matching_results()

    log_with_timestamp('🏁 All processing completed successfully!')

if __name__ == '__main__':
    main()