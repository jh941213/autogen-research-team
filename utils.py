import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timezone, timedelta
import json
import re

from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient, OpenAIChatCompletionClient
from autogen_core.models import ModelInfo

from configuration import Configuration, SearchAPI
from state import WebSearchResult, StateManager

MODEL_TOKEN_LIMITS = {

    "openai:gpt-4.1-mini": 1047576,
    "openai:gpt-4.1": 1047576,
    "openai:gpt-4.1-nano": 1047576,
    "openai:gpt-4o-mini": 128000,
    "openai:gpt-4o": 128000,
    "openai:o3-mini": 200000,
    
    "anthropic:claude-opus-4": 200000,
    "anthropic:claude-sonnet-4": 200000,
    "anthropic:claude-3-5-sonnet": 200000,
    "anthropic:claude-3-5-haiku": 200000,
    

    "google:gemini-1.5-pro": 2097152,
    "google:gemini-1.5-flash": 1048576,
    "google:gemini-2.0-flash": 1048576,
    "google:gemini-2.5-flash": 1048576,
    "google:gemini-2.5-pro": 2097152,
    

    "azure:gpt-4.1-mini": 1047576,
    "azure:gpt-4.1": 1047576,
    "azure:gpt-4.1-nano": 1047576,
    "azure:gpt-4o-mini": 128000,
    "azure:gpt-4o": 128000,
    "azure:gpt-4-turbo": 128000,
    "azure:gpt-35-turbo": 16384,
    "azure:o3-mini": 200000,
    

    "cohere:command-r-plus": 128000,
    "mistral:mistral-large": 32768,
}


def get_model_token_limit(model_string: str) -> Optional[int]:
    """Get token limit for a given model string"""
    model_string = model_string.lower()
    
    # Direct match
    if model_string in MODEL_TOKEN_LIMITS:
        return MODEL_TOKEN_LIMITS[model_string]
    
    # Partial match
    for key, token_limit in MODEL_TOKEN_LIMITS.items():
        if key in model_string or model_string in key:
            return token_limit
    
    # Default limits based on provider
    if "gpt-4" in model_string:
        return 128000
    elif "gpt-3" in model_string:
        return 16384
    elif "claude" in model_string:
        return 200000
    elif "gemini" in model_string:
        return 1048576
    
    return None


def is_token_limit_exceeded(exception: Exception, model_name: str = None) -> bool:
    error_str = str(exception).lower()
    provider = None
    
    # Identify provider from model name
    if model_name:
        model_str = str(model_name).lower()
        if model_str.startswith('openai:') or model_str.startswith('azure:'):
            provider = 'openai'
        elif model_str.startswith('anthropic:'):
            provider = 'anthropic'
        elif model_str.startswith('gemini:') or model_str.startswith('google:'):
            provider = 'gemini'
    

    if provider == 'openai':
        return _check_openai_token_limit(exception, error_str)
    elif provider == 'anthropic':
        return _check_anthropic_token_limit(exception, error_str)
    elif provider == 'gemini':
        return _check_gemini_token_limit(exception, error_str)
    

    return (_check_openai_token_limit(exception, error_str) or
            _check_anthropic_token_limit(exception, error_str) or
            _check_gemini_token_limit(exception, error_str))


def _check_openai_token_limit(exception: Exception, error_str: str) -> bool:
    """Check for OpenAI/Azure OpenAI token limit errors"""
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    is_openai_exception = ('openai' in exception_type.lower() or 
                          'openai' in module_name.lower() or
                          'azure' in module_name.lower())
    is_bad_request = class_name in ['BadRequestError', 'InvalidRequestError']
    
    if is_openai_exception and is_bad_request:
        token_keywords = ['token', 'context', 'length', 'maximum context', 
                         'reduce', 'too long', 'exceeds']
        if any(keyword in error_str for keyword in token_keywords):
            return True
    
    # Check for specific error codes
    if hasattr(exception, 'code'):
        code = getattr(exception, 'code', '')
        if code in ['context_length_exceeded', 'invalid_request_error']:
            return True
    
    # Check response body if available
    if hasattr(exception, 'response'):
        try:
            response_body = getattr(exception.response, 'json', lambda: {})()
            error_code = response_body.get('error', {}).get('code', '')
            if error_code == 'context_length_exceeded':
                return True
        except:
            pass
    
    return False


def _check_anthropic_token_limit(exception: Exception, error_str: str) -> bool:
    """Check for Anthropic Claude token limit errors"""
    exception_type = str(type(exception))
    class_name = exception.__class__.__name__
    module_name = getattr(exception.__class__, '__module__', '')
    
    is_anthropic_exception = ('anthropic' in exception_type.lower() or 
                             'anthropic' in module_name.lower())
    is_bad_request = class_name == 'BadRequestError'
    
    if is_anthropic_exception and is_bad_request:
        if 'prompt is too long' in error_str:
            return True
        if 'maximum number of tokens' in error_str:
            return True
    
    return False


def _check_gemini_token_limit(exception: Exception, error_str: str) -> bool:
    """Check for Google Gemini token limit errors"""
    exception_type = str(type(exception))
    exception_module = str(exception.__class__.__module__)
    
    is_gemini_exception = ('google' in exception_type.lower() or 
                          'gemini' in exception_module.lower())
    
    if is_gemini_exception:
        if 'token' in error_str and ('limit' in error_str or 'exceed' in error_str):
            return True
    
    # Generic token limit patterns
    if 'request too large' in error_str or 'content too long' in error_str:
        return True
    
    return False


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a given text
    
    This is a rough estimate: ~1 token per 4 characters for English text
    """
    return len(text) // 4


def truncate_to_token_limit(text: str, max_tokens: int, model_name: str = None) -> str:
    """
    Truncate text to fit within token limit
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
        model_name: Optional model name to get specific limit
        
    Returns:
        Truncated text
    """
    if model_name:
        model_limit = get_model_token_limit(model_name)
        if model_limit and max_tokens > model_limit:
            max_tokens = model_limit
    
    estimated_tokens = estimate_tokens(text)
    if estimated_tokens <= max_tokens:
        return text
    
    # Calculate character limit (rough approximation)
    char_limit = max_tokens * 4
    
    # Leave some buffer for safety
    char_limit = int(char_limit * 0.9)
    
    if len(text) <= char_limit:
        return text
    
    # Truncate and add ellipsis
    return text[:char_limit-3] + "..."

# Message History Management


def remove_up_to_last_ai_message(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Remove messages up to the last AI message to reduce token usage
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Truncated message list
    """
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            return messages[:i]
    return messages


def compress_message_history(messages: List[Dict[str, Any]], 
                           target_tokens: int,
                           model_name: str = None) -> List[Dict[str, Any]]:
    """
    Compress message history to fit within token limit
    
    Args:
        messages: List of message dictionaries
        target_tokens: Target token count
        model_name: Optional model name
        
    Returns:
        Compressed message list
    """
    # Always keep the system message and latest user message
    if not messages:
        return messages
    
    system_messages = [m for m in messages if m.get("role") == "system"]
    user_messages = [m for m in messages if m.get("role") == "user"]
    
    if not user_messages:
        return messages
    
    # Start with system message and latest user message
    compressed = system_messages + [user_messages[-1]]
    
    # Estimate current tokens
    current_text = " ".join([m.get("content", "") for m in compressed])
    current_tokens = estimate_tokens(current_text)
    
    if current_tokens > target_tokens:
        # Even minimal messages exceed limit - truncate content
        for msg in compressed:
            if msg.get("role") == "user":
                msg["content"] = truncate_to_token_limit(
                    msg["content"], 
                    target_tokens // 2, 
                    model_name
                )
    
    return compressed



# Error Handling and Recovery


class ResearchError(Exception):
    """Custom exception for research-related errors"""
    pass


class TokenLimitError(ResearchError):
    """Exception raised when token limit is exceeded"""
    def __init__(self, message: str, model_name: str = None, token_count: int = None):
        super().__init__(message)
        self.model_name = model_name
        self.token_count = token_count


async def handle_api_error(error: Exception, retry_count: int = 0, max_retries: int = 3) -> bool:
    """
    Handle API errors with appropriate retry logic
    
    Args:
        error: The exception to handle
        retry_count: Current retry attempt
        max_retries: Maximum number of retries
        
    Returns:
        True if should retry, False otherwise
    """
    error_str = str(error).lower()
    
    # Rate limit errors - use exponential backoff
    if "rate limit" in error_str or "429" in error_str:
        if retry_count < max_retries:
            wait_time = min(60, (2 ** retry_count) * 5)  # 5, 10, 20, 40, 60 seconds
            logging.warning(f"Rate limit hit, waiting {wait_time} seconds...")
            await asyncio.sleep(wait_time)
            return True
    
    # Temporary errors - quick retry
    if any(err in error_str for err in ["timeout", "connection", "temporary"]):
        if retry_count < max_retries:
            await asyncio.sleep(2)
            return True
    
    # Token limit errors - don't retry
    if is_token_limit_exceeded(error):
        raise TokenLimitError(f"Token limit exceeded: {error}")
    
    # Content filter errors - don't retry
    if is_content_filter_error(error):
        logging.warning(f"Content filter error: {error}")
        return False
    
    return False


def is_content_filter_error(error: Exception) -> bool:
    """
    Azure OpenAI 콘텐츠 필터 에러인지 확인
    
    Args:
        error: 확인할 예외 객체
        
    Returns:
        콘텐츠 필터 에러인 경우 True
    """
    error_str = str(error).lower()
    return (
        "content_filter" in error_str or 
        "responsibleai" in error_str or
        "content management policy" in error_str
    )


def handle_content_filter_error(topic: str, researcher_name: str, user_request: str = ""):
    """
    콘텐츠 필터 에러에 대한 대체 연구 결과 생성
    
    Args:
        topic: 연구 주제
        researcher_name: 연구자 이름  
        user_request: 원본 사용자 요청
        
    Returns:
        제한된 연구 결과 객체
    """
    from state import ResearchResult
    
    findings = f"""이 주제는 Azure OpenAI 콘텐츠 정책으로 인해 제한된 연구가 수행되었습니다.

**제한 사유**: 콘텐츠 필터 정책 (폭력, 혐오 표현 등)

**권장 사항**:
1. 더 중성적인 표현으로 질문을 다시 작성해보세요
2. 구체적인 기업명이나 민감한 표현을 피해보세요  
3. 일반적인 산업 동향으로 범위를 넓혀보세요

**예시 대체 표현**:
- "기업간 인재 경쟁" → "산업 내 인재 확보 전략"
- "빼돌리는" → "유치하는", "확보하는"
- "경쟁사 견제" → "시장 경쟁력 강화"
"""

    return ResearchResult(
        topic=topic,
        findings=findings,
        sources=["콘텐츠 필터 제한으로 인한 분석"],
        raw_notes=[{
            "note": "content_filter_limitation", 
            "timestamp": datetime.now().isoformat(),
            "original_request": user_request
        }],
        confidence=0.2,  # 매우 낮은 신뢰도
        researcher_id=researcher_name,
        iterations_used=1
    )


# Search Query Optimization


def generate_search_queries(topic: str, 
                          existing_notes: List[str] = None,
                          max_queries: int = 3) -> List[str]:
    """
    Generate optimized search queries for a research topic
    
    Args:
        topic: Research topic
        existing_notes: Previous research notes to avoid duplication
        max_queries: Maximum number of queries to generate
        
    Returns:
        List of search queries
    """
    queries = []
    
    # Extract key concepts from topic
    key_concepts = extract_key_concepts(topic)
    
    # Generate different query types
    if len(key_concepts) > 0:
        # Conceptual query
        queries.append(" ".join(key_concepts[:3]))
        
        # Current state query
        if len(queries) < max_queries:
            queries.append(f"{key_concepts[0]} latest 2024 developments")
        
        # Problem-focused query
        if len(queries) < max_queries and len(key_concepts) > 1:
            queries.append(f"{key_concepts[0]} {key_concepts[1]} challenges solutions")
    
    # If not enough queries, add the original topic
    if len(queries) < max_queries:
        queries.append(topic)
    
    # Remove duplicates and limit
    seen = set()
    unique_queries = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique_queries.append(q)
            if len(unique_queries) >= max_queries:
                break
    
    return unique_queries


def extract_key_concepts(text: str) -> List[str]:
    """
    Extract key concepts from text for search query generation
    
    Args:
        text: Input text
        
    Returns:
        List of key concepts
    """
    # Remove common words
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "this", "that", "these", "those",
        "what", "which", "who", "when", "where", "why", "how", "about"
    }
    
    # Simple tokenization and filtering
    words = text.lower().split()
    concepts = []
    
    for word in words:
        # Remove punctuation
        word = re.sub(r'[^\w\s-]', '', word)
        
        # Skip if stop word or too short
        if word not in stop_words and len(word) > 2:
            concepts.append(word)
    
    return concepts


def get_model_config(config: Configuration, model_type: str) -> Dict[str, Any]:
    """
    Get model configuration for a specific model type
    
    Args:
        config: Configuration object
        model_type: Type of model (summarization, research, compression, final_report)
        
    Returns:
        Model configuration dictionary
    """
    model_configs = {
        "summarization": config.summarization_model,
        "research": config.research_model,
        "compression": config.compression_model,
        "final_report": config.final_report_model
    }
    
    return model_configs.get(model_type, config.research_model)


def validate_configuration(config: Configuration) -> List[str]:
    """
    Validate configuration and return list of warnings/errors
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of validation messages
    """
    messages = []
    
    # Check API keys for search APIs
    if hasattr(config, 'search_apis') and config.search_apis:
        for search_api in config.search_apis:
            if search_api == SearchAPI.TAVILY and not config.search_api_key:
                messages.append(f"Warning: Tavily API selected but no TAVILY_API_KEY provided")
            # DuckDuckGo doesn't need API key, so no check needed
    
    # Check model configurations
    for model_type in ["summarization", "research", "compression", "final_report"]:
        model_config = getattr(config, f"{model_type}_model")
        if not model_config:
            messages.append(f"Error: {model_type}_model not configured")
    
    # Check concurrent limits
    if config.max_concurrent_research_units > 10:
        messages.append("Warning: max_concurrent_research_units > 10 may hit rate limits")
    
    # Check token limits
    for model_type in ["summarization", "research", "compression", "final_report"]:
        model_config = getattr(config, f"{model_type}_model")
        max_tokens = getattr(config, f"{model_type}_model_max_tokens", 0)
        
        if model_config and max_tokens:
            model_limit = get_model_token_limit(model_config.model_name)
            if model_limit and max_tokens > model_limit:
                messages.append(
                    f"Warning: {model_type}_model_max_tokens ({max_tokens}) "
                    f"exceeds model limit ({model_limit})"
                )
    
    return messages

async def run_with_timeout(coro, timeout_seconds: int, default_value: Any = None):
    """
    Run a coroutine with timeout
    
    Args:
        coro: Coroutine to run
        timeout_seconds: Timeout in seconds
        default_value: Value to return on timeout
        
    Returns:
        Coroutine result or default value
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except asyncio.TimeoutError:
        logging.warning(f"Operation timed out after {timeout_seconds} seconds")
        return default_value


async def batch_process(items: List[Any], 
                       process_func,
                       batch_size: int = 5,
                       delay_between_batches: float = 0.1):
    """
    Process items in batches with delay between batches
    
    Args:
        items: Items to process
        process_func: Async function to process each item
        batch_size: Number of items per batch
        delay_between_batches: Delay in seconds between batches
        
    Returns:
        List of results
    """
    results = []
    
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        
        # Process batch concurrently
        batch_results = await asyncio.gather(
            *[process_func(item) for item in batch],
            return_exceptions=True
        )
        
        results.extend(batch_results)
        
        # Delay before next batch (except for last batch)
        if i + batch_size < len(items):
            await asyncio.sleep(delay_between_batches)
    
    return results


# State Management Helpers

def create_checkpoint(state_manager: StateManager, 
                     workflow_id: str,
                     checkpoint_name: str) -> str:
    """
    Create a checkpoint of current state
    
    Args:
        state_manager: State manager instance
        workflow_id: Workflow ID
        checkpoint_name: Name for the checkpoint
        
    Returns:
        Checkpoint ID
    """
    state = state_manager.load_state(workflow_id)
    if not state:
        raise ValueError(f"No state found for workflow {workflow_id}")
    
    checkpoint_id = f"{workflow_id}_checkpoint_{checkpoint_name}_{datetime.now().timestamp()}"
    state.workflow_id = checkpoint_id
    state_manager.save_state(state)
    
    return checkpoint_id


def restore_checkpoint(state_manager: StateManager,
                      checkpoint_id: str,
                      target_workflow_id: str) -> bool:
    """
    Restore state from checkpoint
    
    Args:
        state_manager: State manager instance
        checkpoint_id: Checkpoint ID to restore from
        target_workflow_id: Target workflow ID
        
    Returns:
        True if successful
    """
    checkpoint_state = state_manager.load_state(checkpoint_id)
    if not checkpoint_state:
        return False
    
    checkpoint_state.workflow_id = target_workflow_id
    state_manager.save_state(checkpoint_state)
    return True


# =====================================================
# Formatting Utilities
# =====================================================

def format_research_results(results: List[Dict[str, Any]], 
                          include_sources: bool = True) -> str:
    """
    Format research results for display
    
    Args:
        results: List of research result dictionaries
        include_sources: Whether to include source URLs
        
    Returns:
        Formatted string
    """
    formatted = "# Research Results\n\n"
    
    for i, result in enumerate(results, 1):
        formatted += f"## Finding {i}: {result.get('topic', 'Research Finding')}\n\n"
        formatted += f"{result.get('findings', '')}\n\n"
        
        if include_sources and result.get('sources'):
            formatted += "### Sources:\n"
            for source in result['sources']:
                formatted += f"- {source}\n"
            formatted += "\n"
        
        formatted += "---\n\n"
    
    return formatted


def create_summary_table(data: List[Dict[str, Any]], 
                        columns: List[str],
                        max_width: int = 50) -> str:
    """
    Create a markdown table from data
    
    Args:
        data: List of dictionaries
        columns: Column names to include
        max_width: Maximum column width
        
    Returns:
        Markdown table string
    """
    if not data or not columns:
        return ""
    
    # Create header
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["-" * (max_width + 2) for _ in columns]) + "|"
    
    # Create rows
    rows = []
    for item in data:
        row_data = []
        for col in columns:
            value = str(item.get(col, ""))
            if len(value) > max_width:
                value = value[:max_width-3] + "..."
            row_data.append(value)
        rows.append("| " + " | ".join(row_data) + " |")
    
    return "\n".join([header, separator] + rows)



# Date and Time Utilities


def get_today_str() -> str:
    """Get today's date as a formatted string"""
    return datetime.now().strftime("%Y-%m-%d")


def calculate_duration(start_time: datetime, end_time: datetime = None) -> str:
    """
    Calculate duration between two times
    
    Args:
        start_time: Start time
        end_time: End time (defaults to now)
        
    Returns:
        Human-readable duration string
    """
    if not end_time:
        end_time = datetime.now(timezone.utc)
    
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    if end_time.tzinfo is None:
        end_time = end_time.replace(tzinfo=timezone.utc)
    
    duration = end_time - start_time
    
    hours, remainder = divmod(duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"