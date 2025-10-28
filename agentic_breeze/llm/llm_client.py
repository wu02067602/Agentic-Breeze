from typing import Any, Dict, List, Optional

from .breeze_client import BreezeClient


class LLMConnector:
    """使用 BreezeClient 的 LLM 連線器（支援 Breeze 模型端點）。

    本類別以簡潔設計為原則，透過 BreezeClient 的 `chat_completions_create` 實作
    三種常見互動：
    - 單次對答（僅 user 提示）
    - 帶歷史訊息的對答（messages 陣列）
    - 伴隨工具定義的對答（function calling / tools schema）

    初始化參數：
        api_key (str): 用於 API 認證的金鑰（Bearer Token）。
        api_base_url (str): 兼容 API 的基底位址，預設為 "https://api.openai.com/v1"。
        default_model (str): 預設模型名稱，預設 "gpt-3.5-turbo"。
        timeout (int): 請求逾時秒數，預設 30。

    例外情況：
        ValueError: 初始化或方法參數不合法。
        Exception: BreezeClient 呼叫或服務端錯誤。

    相依與相容性說明：
        - 使用 BreezeClient 呼叫 Breeze 模型。
        - tools 採用 OpenAI function calling schema，結果位於 choices[0].message.tool_calls。

    使用範例：
    - 單次對答
        >>> connector = LLMConnector(api_key="sk-...", api_base_url="https://api.openai.com/v1")
        >>> text = connector.single_query(prompt="今天天氣如何？")
        >>> print(text)
    
    - 帶歷史訊息的對答
        >>> connector = LLMConnector(api_key="sk-...", api_base_url="https://api.openai.com/v1")
        >>> text = connector.chat_with_history(messages=[{"role": "user", "content": "今天天氣如何？"}])
        >>> print(text)
    
    - 伴隨工具定義的對答
        >>> connector = LLMConnector(api_key="sk-...", api_base_url="https://api.openai.com/v1")
        >>> text = connector.tool_assisted_query(messages=[{"role": "user", "content": "今天天氣如何？"}], tools=[{"type": "function", "function": {"name": "get_weather", "description": "Get the weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}})
        >>> print(text)
    """

    def __init__(
        self,
        host_type: str = "ollama",
        timeout: int = 30,
        max_tokens: int = 1000,
        temperature: float = 0.5,
    ) -> None:

        self._client: BreezeClient = BreezeClient(host_type=host_type)
        self._default_model: str = self._client._model
        self._timeout: int = timeout
        self._max_tokens: int = max_tokens
        self._temperature: float = temperature

    def single_query(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """單次對答：使用單一使用者提示詞呼叫模型並回傳文字結果。

        Args:
            prompt (str): 使用者提示內容，不可為空。
            model (Optional[str]): 模型名稱；預設使用初始化的 `default_model`。
            max_tokens (Optional[int]): 限制生成的最大 token 數量；可不填由伺服器預設。
            temperature (Optional[float]): 取樣溫度；可不填由伺服器預設。
            system_prompt (Optional[str]): 系統提示，可用來規範助手行為。

        Returns:
            str: 助手生成的文字內容。若回應不含文字，回傳空字串。

        Examples:
            >>> connector = LLMConnector(api_key="sk-...", api_base_url="https://api.openai.com/v1")
            >>> text = connector.single_query(prompt="今天天氣如何？")
            >>> print(text)

        Raises:
            ValueError: 當 `prompt` 為空或參數不合法。
            requests.RequestException: 當請求或回應發生錯誤。
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt 不可為空，且需為字串。")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens 必須為正整數。")

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.chat_completions_create(
            model=model or self._default_model,
            messages=messages,
            max_tokens=max_tokens or self._max_tokens,
            temperature=temperature or self._temperature,
            timeout=self._timeout,
            stream=False,
        )
        msg = resp.choices[0].message
        return msg.content or ""

    def chat_with_history(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """歷史訊息對答：以既有 `messages` 呼叫模型並回傳文字結果。

        Args:
            messages (List[Dict[str, str]]): 對話歷史，需符合 OpenAI Chat 格式，
                每則訊息需含 `role`（system/user/assistant/tool）與 `content`（部分情形 tool 角色可含其它欄位）。
            model (Optional[str]): 模型名稱；預設使用初始化的 `default_model`。
            max_tokens (Optional[int]): 限制生成的最大 token 數量；可不填由伺服器預設。
            temperature (Optional[float]): 取樣溫度；可不填由伺服器預設。

        Returns:
            str: 助手生成的文字內容。若回應不含文字，回傳空字串。

        Examples:
            >>> connector = LLMConnector(api_key="sk-...", api_base_url="https://api.openai.com/v1")
            >>> text = connector.chat_with_history(messages=[{"role": "user", "content": "今天天氣如何？"}, {"role": "assistant", "content": "今天天氣晴朗"}, {"role": "user", "content": "請問明天呢？"}])
            >>> print(text)

        Raises:
            ValueError: 當 `messages` 為空或結構不合法。
            requests.RequestException: 當請求或回應發生錯誤。
        """
        if not messages or not isinstance(messages, list):
            raise ValueError("messages 不可為空，且需為列表。")

        params: Dict[str, Any] = {
            "model": model or self._default_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": self._timeout,
        }
        # Ensure stream is False for both methods
        params["stream"] = False
        resp = self._client.chat_completions_create(**{k: v for k, v in params.items() if v is not None})
        msg = resp.choices[0].message
        return msg.content or ""

    def tool_assisted_query(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        *,
        model: Optional[str] = None,
        tool_choice: Optional[str] = "auto",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """工具輔助對答：附帶工具定義呼叫模型，回傳文字與工具呼叫資訊。

        本方法會將 `tools`（function calling schema）一併傳入 `/chat/completions`，
        並回傳精簡結構：{"content": str | None, "tool_calls": list | None, "raw": dict}

        Args:
            messages (List[Dict[str, Any]]): 對話歷史（可含 system/user/assistant/tool 角色）。
            tools (List[Dict[str, Any]]): 工具定義，採用 OpenAI 工具格式，例如：
                [{
                    "type": "function",
                    "function": {
                        "name": "search_weather",
                        "description": "查詢天氣",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
                    }
                }]
            model (Optional[str]): 模型名稱；預設使用初始化的 `default_model`。
            tool_choice (Optional[str]): 工具選擇策略（例如 "auto"、"none"、或指定函式）。
            max_tokens (Optional[int]): 限制生成的最大 token 數量。
            temperature (Optional[float]): 取樣溫度。

        Returns:
            Dict[str, Any]: 結果物件，包含：
                - content (str | None): 助手文字內容（若模型選擇改以工具呼叫，則可能為 None）。
                - tool_calls (list | None): 工具呼叫清單（若無則為 None）。
                - raw (dict): 伺服器回應的原始 JSON（供進一步處理）。

        Examples:
            >>> connector = LLMConnector(api_key="sk-...", api_base_url="https://api.openai.com/v1")
            >>> text = connector.tool_assisted_query(messages=[{"role": "user", "content": "今天天氣如何？"}], tools=[{"type": "function", "function": {"name": "get_weather", "description": "Get the weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}})
            >>> print(text)

        Raises:
            ValueError: 當 `messages` 或 `tools` 不合法。
            requests.RequestException: 當請求或回應發生錯誤。
        """
        if not messages or not isinstance(messages, list):
            raise ValueError("messages 不可為空，且需為列表。")
        if not tools or not isinstance(tools, list):
            raise ValueError("tools 不可為空，且需為列表（OpenAI function 工具格式）。")

        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens 必須為正整數。")

        params: Dict[str, Any] = {
            "model": model or self._default_model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": self._timeout,
        }
        # Ensure stream is False for both methods
        params["stream"] = False
        resp = self._client.chat_completions_create(**{k: v for k, v in params.items() if v is not None})

        msg = resp.choices[0].message
        content = msg.content if isinstance(msg.content, str) else None
        tool_calls = getattr(msg, "tool_calls", None) if isinstance(getattr(msg, "tool_calls", None), list) else None

        raw: Any
        if hasattr(resp, "to_dict") and callable(getattr(resp, "to_dict")):
            raw = resp.to_dict()  # type: ignore[attr-defined]
        elif hasattr(resp, "model_dump") and callable(getattr(resp, "model_dump")):
            raw = resp.model_dump()  # pydantic v2
        else:
            raw = {"id": getattr(resp, "id", None), "object": getattr(resp, "object", None)}

        return {"content": content, "tool_calls": tool_calls, "raw": raw}

    def single_query_stream(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ):
        """單次對答串流版本：使用單一使用者提示詞呼叫模型並回傳串流結果。

        Args:
            prompt (str): 使用者提示內容，不可為空。
            model (Optional[str]): 模型名稱；預設使用初始化的 `default_model`。
            max_tokens (Optional[int]): 限制生成的最大 token 數量；可不填由伺服器預設。
            temperature (Optional[float]): 取樣溫度；可不填由伺服器預設。
            system_prompt (Optional[str]): 系統提示，可用來規範助手行為。

        Returns:
            Generator: 串流 chunk 生成器，每個 chunk 包含增量內容。

        Examples:
            >>> connector = LLMConnector(host_type='ollama')
            >>> stream = connector.single_query_stream(prompt="今天天氣如何？")
            >>> for chunk in stream:
            ...     if chunk.choices[0].delta.content:
            ...         print(chunk.choices[0].delta.content, end="")

        Raises:
            ValueError: 當 `prompt` 為空或參數不合法。
            requests.RequestException: 當請求或回應發生錯誤。
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt 不可為空，且需為字串。")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens 必須為正整數。")

        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = self._client.chat_completions_create(
            model=model or self._default_model,
            messages=messages,
            max_tokens=max_tokens or self._max_tokens,
            temperature=temperature or self._temperature,
            timeout=self._timeout,
            stream=True,
        )
        
        return stream

    def chat_with_history_stream(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ):
        """歷史訊息對答串流版本：以既有 `messages` 呼叫模型並回傳串流結果。

        Args:
            messages (List[Dict[str, str]]): 對話歷史，需符合 OpenAI Chat 格式。
            model (Optional[str]): 模型名稱；預設使用初始化的 `default_model`。
            max_tokens (Optional[int]): 限制生成的最大 token 數量；可不填由伺服器預設。
            temperature (Optional[float]): 取樣溫度；可不填由伺服器預設。

        Returns:
            Generator: 串流 chunk 生成器，每個 chunk 包含增量內容。

        Examples:
            >>> connector = LLMConnector(host_type='ollama')
            >>> messages = [{"role": "user", "content": "今天天氣如何？"}]
            >>> stream = connector.chat_with_history_stream(messages=messages)
            >>> for chunk in stream:
            ...     if chunk.choices[0].delta.content:
            ...         print(chunk.choices[0].delta.content, end="")

        Raises:
            ValueError: 當 `messages` 為空或結構不合法。
            requests.RequestException: 當請求或回應發生錯誤。
        """
        if not messages or not isinstance(messages, list):
            raise ValueError("messages 不可為空，且需為列表。")

        params: Dict[str, Any] = {
            "model": model or self._default_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": self._timeout,
            "stream": True,
        }
        
        stream = self._client.chat_completions_create(**{k: v for k, v in params.items() if v is not None})
        return stream
