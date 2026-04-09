"""
招聘助手 - 辅助工具函数
========================
提供 LLM 实例创建和 JSON 解析等通用功能。
"""

import os
import json
import re
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

# 加载 .env（从项目根目录）
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


def get_llm(temperature: float = 0.3, json_mode: bool = False) -> ChatOpenAI:
    """
    创建 LLM 实例，从环境变量读取模型名和 API 地址。

    Args:
        temperature: 生成温度
        json_mode: 是否启用 JSON Mode，强制模型返回合法 JSON
    """
    kwargs = {}
    if json_mode:
        kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}

    return ChatOpenAI(
        model=os.getenv("DEFAULT_LLM_PROVIDER", "gpt-4o-mini"),
        base_url=os.getenv("BASE_URL") or None,
        temperature=temperature,
        **kwargs,
    )


def invoke_for_json(llm, messages: list, max_retries: int = 2) -> dict | list:
    """
    调用 LLM 并解析 JSON 响应，解析失败时自动重试。

    重试策略：将模型的错误输出追加到对话中，要求模型修正。
    最多重试 max_retries 次（默认 2 次，即总共最多调用 3 次）。

    Args:
        llm: ChatOpenAI 实例（建议开启 json_mode）
        messages: 消息列表
        max_retries: 最大重试次数
    """
    current_messages = list(messages)
    result = None

    for attempt in range(max_retries + 1):
        response = llm.invoke(current_messages)
        result = parse_json_from_llm(response.content)

        # 解析成功
        if not (isinstance(result, dict) and "_error" in result):
            return result

        # 最后一次尝试，不再重试
        if attempt >= max_retries:
            logger.warning(
                "JSON 解析在 %d 次尝试后仍然失败，原始文本: %s",
                attempt + 1,
                response.content[:200],
            )
            break

        # 重试：追加错误上下文，要求模型修正
        logger.info("JSON 解析失败，第 %d 次重试", attempt + 1)
        current_messages = current_messages + [
            AIMessage(content=response.content),
            HumanMessage(content=(
                "Your previous response was not valid JSON and could not be parsed. "
                "Please respond with ONLY the raw JSON object/array, "
                "no markdown code blocks, no explanatory text before or after."
            )),
        ]

    return result


def parse_json_from_llm(text: str) -> dict | list:
    """
    从 LLM 的回复文本中提取 JSON，处理各种常见的非标准输出。

    处理的场景：
    1. ```json ... ``` 代码块（含不完整块）
    2. 前后有说明文字，JSON 嵌在中间
    3. Python 风格布尔值 True/False/None
    4. 单引号代替双引号
    5. 尾部逗号（trailing comma）
    6. JSON 注释 // 和 /* */
    7. BOM 字符
    8. 值中未转义的控制字符（换行等）

    如果所有尝试都失败，返回包含原始文本的 dict。
    """
    if not text or not text.strip():
        return {"_raw": "", "_error": "空文本"}

    # ---- 第 1 步：提取候选 JSON 文本 ----
    candidate = _extract_json_candidate(text)

    # ---- 第 2 步：尝试直接解析（最快路径）----
    result = _try_parse(candidate)
    if result is not None:
        return result

    # ---- 第 3 步：清洗后重试 ----
    cleaned = _clean_json_text(candidate)
    result = _try_parse(cleaned)
    if result is not None:
        return result

    # ---- 第 4 步：用原始文本尝试花括号/方括号提取 ----
    extracted = _extract_braces(text)
    if extracted and extracted != candidate:
        result = _try_parse(extracted)
        if result is not None:
            return result
        cleaned2 = _clean_json_text(extracted)
        result = _try_parse(cleaned2)
        if result is not None:
            return result

    # ---- 所有方法失败 ----
    return {"_raw": text[:500], "_error": "JSON 解析失败"}


def _extract_json_candidate(text: str) -> str:
    """从 LLM 输出中提取最可能的 JSON 文本。"""

    # 去 BOM
    text = text.lstrip("\ufeff")

    # 1. 完整的 markdown 代码块 ```json ... ```
    match = re.search(r"```(?:json|JSON)?\s*\n?([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()

    # 2. 不完整的 markdown 块（只有开头 ```json 没有结尾）
    match = re.search(r"```(?:json|JSON)?\s*\n?([\s\S]+)", text)
    if match:
        inner = match.group(1).strip()
        # 尝试找到 JSON 的结尾
        extracted = _extract_braces(inner)
        if extracted:
            return extracted
        return inner

    # 3. 没有代码块，尝试用花括号/方括号提取
    extracted = _extract_braces(text)
    if extracted:
        return extracted

    return text.strip()


def _extract_braces(text: str) -> str | None:
    """
    从文本中提取最外层的 {...} 或 [...]，正确处理嵌套和字符串内的括号。
    """
    for open_char, close_char in [('{', '}'), ('[', ']')]:
        start = text.find(open_char)
        if start == -1:
            continue

        depth = 0
        in_string = False
        escape = False
        end = -1

        for i in range(start, len(text)):
            ch = text[i]

            if escape:
                escape = False
                continue

            if ch == '\\' and in_string:
                escape = True
                continue

            if ch == '"' and not escape:
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end > start:
            return text[start:end + 1]

    return None


def _clean_json_text(text: str) -> str:
    """
    清洗 JSON 文本，修复常见的非标准格式。
    """
    # 去 BOM
    text = text.lstrip("\ufeff")

    # 移除 // 行注释（不在字符串内的）
    text = re.sub(r'(?<!["\w])//[^\n]*', '', text)

    # 移除 /* */ 块注释
    text = re.sub(r'/\*[\s\S]*?\*/', '', text)

    # Python 布尔值 → JSON 布尔值（仅在非字符串上下文）
    # 使用 word boundary 避免替换字符串内容
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)
    text = re.sub(r'\bNone\b', 'null', text)

    # 单引号 → 双引号（简单场景，处理 {'key': 'value'} 模式）
    # 仅当没有双引号时才尝试，避免破坏正常 JSON
    if '"' not in text and "'" in text:
        text = text.replace("'", '"')

    # 尾部逗号：,] 或 ,}（允许中间有空白）
    text = re.sub(r',\s*([\]}])', r'\1', text)

    # 值中的未转义换行符 → 转义换行（在双引号字符串内部）
    text = _escape_control_chars_in_strings(text)

    return text.strip()


def _escape_control_chars_in_strings(text: str) -> str:
    """
    在 JSON 字符串值内部，将未转义的控制字符（换行、制表等）替换为转义序列。
    """
    result = []
    in_string = False
    escape = False
    i = 0

    while i < len(text):
        ch = text[i]

        if escape:
            result.append(ch)
            escape = False
            i += 1
            continue

        if ch == '\\' and in_string:
            result.append(ch)
            escape = True
            i += 1
            continue

        if ch == '"':
            in_string = not in_string
            result.append(ch)
            i += 1
            continue

        if in_string:
            # 替换控制字符
            if ch == '\n':
                result.append('\\n')
            elif ch == '\r':
                result.append('\\r')
            elif ch == '\t':
                result.append('\\t')
            elif ord(ch) < 0x20:
                result.append(f'\\u{ord(ch):04x}')
            else:
                result.append(ch)
        else:
            result.append(ch)

        i += 1

    return ''.join(result)


def _try_parse(text: str) -> dict | list | None:
    """尝试解析 JSON 文本，成功返回结果，失败返回 None。"""
    if not text:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
