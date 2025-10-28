
import time
import uuid
from typing import List, Dict, Any, Optional

from openai import OpenAI
from mtkresearch.llm.prompt import MRPromptV3


class StreamingChunk:
    """將串流 chunk 轉換為與 OpenAI 格式相容的包裝類別。
    
    Args:
        chunk_data (Dict[str, Any]): 包含串流資料的字典，應包含 'choices' 鍵
    
    Returns:
        StreamingChunk: 串流 chunk 包裝物件
    
    Examples:
        >>> chunk_data = {"choices": [{"delta": {"content": "Hello"}}]}
        >>> chunk = StreamingChunk(chunk_data)
        >>> choices = chunk.choices
    
    Raises:
        KeyError: 當 chunk_data 缺少必要的鍵值時
    """
    def __init__(self, chunk_data: Dict[str, Any]):
        self._data = chunk_data
        
    @property
    def choices(self):
        """取得 choices 列表。
        
        Returns:
            List[Choice]: Choice 物件列表
        """
        return [Choice(self._data.get('choices', [{}])[0])]


class Choice:
    """串流 chunk 的 Choice 包裝類別。
    
    Args:
        choice_data (Dict[str, Any]): 包含選擇資料的字典，應包含 'delta' 和 'finish_reason' 鍵
    
    Returns:
        Choice: Choice 物件
    
    Examples:
        >>> choice_data = {"delta": {"content": "Hello"}, "finish_reason": "stop"}
        >>> choice = Choice(choice_data)
        >>> delta = choice.delta
    
    Raises:
        KeyError: 當 choice_data 缺少必要的鍵值時
    """
    def __init__(self, choice_data: Dict[str, Any]):
        self._data = choice_data
        
    @property
    def delta(self):
        """取得 delta 物件。
        
        Returns:
            Delta: Delta 物件，包含串流增量資料
        """
        return Delta(self._data.get('delta', {}))
    
    @property
    def finish_reason(self):
        """取得結束原因。
        
        Returns:
            Optional[str]: 結束原因字串，可能為 None
        """
        return self._data.get('finish_reason')


class Delta:
    """串流 chunk 的 Delta 包裝類別，表示增量變化。
    
    Args:
        delta_data (Dict[str, Any]): 包含增量資料的字典，應包含 'content' 或 'tool_calls' 鍵
    
    Returns:
        Delta: Delta 物件
    
    Examples:
        >>> delta_data = {"content": "Hello", "tool_calls": None}
        >>> delta = Delta(delta_data)
        >>> content = delta.content
    
    Raises:
        KeyError: 當 delta_data 缺少必要的鍵值時
    """
    def __init__(self, delta_data: Dict[str, Any]):
        self._data = delta_data
        
    @property
    def content(self):
        """取得內容字串。
        
        Returns:
            Optional[str]: 增量內容字串，可能為 None
        """
        return self._data.get('content')
    
    @property
    def tool_calls(self):
        """取得工具呼叫列表。
        
        Returns:
            Optional[List]: 工具呼叫列表，可能為 None
        """
        return self._data.get('tool_calls')


class Message:
    """聊天完成回應的 Message 包裝類別。
    
    Args:
        message_data (Dict[str, Any]): 包含訊息資料的字典，應包含 'content' 或 'tool_calls' 鍵
    
    Returns:
        Message: Message 物件
    
    Examples:
        >>> message_data = {"content": "Hello, how can I help?", "tool_calls": None}
        >>> message = Message(message_data)
        >>> content = message.content
    
    Raises:
        KeyError: 當 message_data 缺少必要的鍵值時
    """
    def __init__(self, message_data: Dict[str, Any]):
        self._data = message_data
        
    @property
    def content(self):
        """取得訊息內容。
        
        Returns:
            Optional[str]: 訊息內容字串，可能為 None
        """
        return self._data.get('content')
    
    @property
    def tool_calls(self):
        """取得工具呼叫列表。
        
        Returns:
            Optional[List]: 工具呼叫列表，可能為 None
        """
        return self._data.get('tool_calls')


class ResponseChoice:
    """聊天完成回應的 Choice 包裝類別。
    
    Args:
        choice_data (Dict[str, Any]): 包含選擇資料的字典，應包含 'message' 和 'finish_reason' 鍵
    
    Returns:
        ResponseChoice: ResponseChoice 物件
    
    Examples:
        >>> choice_data = {"message": {"content": "Hello"}, "finish_reason": "stop"}
        >>> choice = ResponseChoice(choice_data)
        >>> message = choice.message
    
    Raises:
        KeyError: 當 choice_data 缺少必要的鍵值時
    """
    def __init__(self, choice_data: Dict[str, Any]):
        self._data = choice_data
        
    @property
    def message(self):
        """取得訊息物件。
        
        Returns:
            Message: Message 物件，包含回應內容
        """
        return Message(self._data.get('message', {}))
    
    @property
    def finish_reason(self):
        """取得結束原因。
        
        Returns:
            Optional[str]: 結束原因字串，可能為 None
        """
        return self._data.get('finish_reason')


class ChatCompletionResponse:
    """將字典形式的回應轉換為支援物件存取的包裝類別。
    
    Args:
        response_data (Dict[str, Any]): 包含完整回應資料的字典，應包含 'choices', 'id', 'model' 等鍵
    
    Returns:
        ChatCompletionResponse: ChatCompletionResponse 物件
    
    Examples:
        >>> response_data = {
        ...     "id": "chatcmpl-123",
        ...     "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
        ...     "model": "breeze-2-8b"
        ... }
        >>> response = ChatCompletionResponse(response_data)
        >>> message = response.choices[0].message
    
    Raises:
        KeyError: 當 response_data 缺少必要的鍵值時
    """
    def __init__(self, response_data: Dict[str, Any]):
        self._data = response_data
        
    @property
    def choices(self):
        """取得選擇列表。
        
        Returns:
            List[ResponseChoice]: ResponseChoice 物件列表
        """
        return [ResponseChoice(choice) for choice in self._data.get('choices', [])]
    
    @property
    def id(self):
        """取得回應 ID。
        
        Returns:
            Optional[str]: 回應 ID 字串，可能為 None
        """
        return self._data.get('id')
    
    @property
    def object(self):
        """取得物件類型。
        
        Returns:
            Optional[str]: 物件類型字串，可能為 None
        """
        return self._data.get('object')
    
    @property
    def created(self):
        """取得建立時間戳記。
        
        Returns:
            Optional[int]: Unix 時間戳記，可能為 None
        """
        return self._data.get('created')
    
    @property
    def model(self):
        """取得模型名稱。
        
        Returns:
            Optional[str]: 模型名稱字串，可能為 None
        """
        return self._data.get('model')
    
    @property
    def usage(self):
        """取得使用量資訊。
        
        Returns:
            Optional[Dict[str, int]]: 使用量字典，包含 token 數量資訊，可能為 None
        """
        return self._data.get('usage')
        
    def to_dict(self):
        """將回應轉換為字典。
        
        Returns:
            Dict[str, Any]: 完整的回應資料字典
        """
        return self._data
        
    def model_dump(self):
        """將回應轉換為字典（Pydantic 相容方法）。
        
        Returns:
            Dict[str, Any]: 完整的回應資料字典
        """
        return self._data

class BreezeClient:
    """Breeze 模型的客戶端類別，支援 Ollama 和 vLLM 兩種部署方式。
    
    本類別提供與 Breeze 模型互動的介面，支援聊天完成（chat completion）功能，
    包含串流和非串流模式，以及工具呼叫（function calling）功能。
    
    Args:
        host_type (str): 主機類型，可選 'ollama' 或 'vllm'，預設為 'ollama'
        api_key (Optional[str]): API 金鑰，當 host_type 非標準類型時必須提供
        base_url (Optional[str]): API 基礎 URL，當 host_type 非標準類型時必須提供
    
    Returns:
        BreezeClient: BreezeClient 實例
    
    Examples:
        >>> # 使用 Ollama
        >>> client = BreezeClient(host_type='ollama')
        >>> response = client.chat_completions_create(
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
        
        >>> # 使用 vLLM
        >>> client = BreezeClient(host_type='vllm')
        >>> response = client.chat_completions_create(
        ...     messages=[{"role": "user", "content": "Hello"}]
        ... )
        
        >>> # 使用自訂端點
        >>> client = BreezeClient(
        ...     host_type='custom',
        ...     api_key='your-key',
        ...     base_url='https://your-endpoint.com/v1'
        ... )
    
    Raises:
        AssertionError: 當 host_type 非標準類型但未提供 api_key 或 base_url 時
        ValueError: 當訊息轉換為提示詞時發生錯誤
    """
    def __init__(self, host_type='ollama', api_key=None, base_url=None):
        if host_type == 'ollama':
            base_url = "http://localhost:11434/v1"
            api_key = "ollama"
            self._model = 'ycchen/Breeze2-8B-TextOnly-Q4_K_M-NoTemplate'
        elif host_type == 'vllm':
            base_url = "http://localhost:6667/v1"
            api_key = "token-abc123"
            self._model = 'voidful/Llama-Breeze2-8B-Instruct-text-only'
        else:
            self._model = None
            assert api_key and base_url
        
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._prompt_engine = MRPromptV3()
    
    def chat_completions_create(
        self,
        messages,
        model=None,
        max_tokens=None,
        temperature=0.8,
        top_p=0.5,
        stream=False,
        tool_choice="auto",  # Keep parameter for API compatibility
        tools=None,
        timeout=None,
    ):
        """建立聊天完成請求。
        
        Args:
            messages (List[Dict[str, str]]): 對話訊息列表，每則訊息包含 role 和 content
            model (Optional[str]): 模型名稱，預設使用初始化時設定的模型
            max_tokens (Optional[int]): 生成的最大 token 數量
            temperature (float): 取樣溫度，預設 0.8
            top_p (float): nucleus sampling 參數，預設 0.5
            stream (bool): 是否使用串流模式，預設 False
            tool_choice (str): 工具選擇策略，預設 "auto"
            tools (Optional[List[Dict[str, Any]]]): 工具定義列表，採用 OpenAI 格式
            timeout (Optional[int]): 請求逾時秒數
        
        Returns:
            Union[ChatCompletionResponse, Generator]: 非串流模式回傳 ChatCompletionResponse，串流模式回傳生成器
        
        Examples:
            >>> client = BreezeClient(host_type='ollama')
            >>> response = client.chat_completions_create(
            ...     messages=[{"role": "user", "content": "Hello"}],
            ...     max_tokens=100
            ... )
            >>> print(response.choices[0].message.content)
        
        Raises:
            ValueError: 當訊息轉換為提示詞時發生錯誤
        """
        if model is None:
            model = self._model
        
        # Convert OpenAI tools to MRPromptV3 functions format
        functions = self._convert_openai_tools_to_functions(tools)
        
        # Create MRPromptV3 instance
        prompt_engine = MRPromptV3()
        
        # Convert messages to prompt string
        try:
            final_prompt = prompt_engine.get_prompt(messages, functions=functions)
        except Exception as e:
            raise ValueError(f"Error converting messages to prompt: {str(e)}")
        
        # Estimate prompt tokens
        prompt_tokens = self._estimate_tokens(final_prompt)
        
        # Prepare completion parameters
        completion_params = {
            "model": model,
            "prompt": final_prompt,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "stop": ["<|eot_id|>"]  # Stop at end of turn
        }
        
        if max_tokens:
            completion_params["max_tokens"] = max_tokens
        if timeout:
            completion_params["timeout"] = timeout
            
        if stream:
            return self._handle_streaming_response(completion_params, prompt_engine, model, prompt_tokens)
        else:
            return self._handle_non_streaming_response(completion_params, prompt_engine, model, prompt_tokens)

    def _convert_openai_tools_to_functions(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """將 OpenAI 工具格式轉換為 MRPromptV3 函式格式。
        
        Args:
            tools (Optional[List[Dict[str, Any]]]): OpenAI 格式的工具定義列表
        
        Returns:
            Optional[List[Dict[str, Any]]]: MRPromptV3 格式的函式定義列表，若無工具則回傳 None
        
        Examples:
            >>> tools = [{"type": "function", "function": {"name": "get_weather", "description": "取得天氣"}}]
            >>> functions = client._convert_openai_tools_to_functions(tools)
        
        Raises:
            KeyError: 當工具格式不正確時
        """
        if not tools:
            return None
            
        functions = []
        for tool in tools:
            if tool.get('type') == 'function' and 'function' in tool:
                func_def = tool['function']
                functions.append({
                    'name': func_def['name'],
                    'description': func_def['description'],
                    'parameters': func_def.get('parameters')
                })
        return functions if functions else None

    def _create_openai_response(self, generated_text: str, model: str, prompt_tokens: int, 
                              completion_tokens: int, finish_reason: str = "stop") -> Dict[str, Any]:
        """建立 OpenAI ChatCompletion 格式的回應。
        
        Args:
            generated_text (str): 生成的文字內容
            model (str): 使用的模型名稱
            prompt_tokens (int): 提示詞使用的 token 數量
            completion_tokens (int): 完成回應使用的 token 數量
            finish_reason (str): 完成原因，預設為 "stop"
        
        Returns:
            Dict[str, Any]: OpenAI 格式的回應字典
        
        Examples:
            >>> response = client._create_openai_response(
            ...     "Hello!", "breeze-2-8b", 10, 5
            ... )
        
        Raises:
            ValueError: 當參數值不合法時
        """
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": generated_text,  # This will be replaced with parsed message
                "finish_reason": finish_reason
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }

    def _create_streaming_chunk(self, delta: Optional[Dict[str, Any]], model: str, 
                              finish_reason: Optional[str] = None, index: int = 0) -> Dict[str, Any]:
        """建立 OpenAI ChatCompletionChunk 格式的串流區塊。
        
        Args:
            delta (Optional[Dict[str, Any]]): 增量資料字典
            model (str): 使用的模型名稱
            finish_reason (Optional[str]): 完成原因，預設為 None
            index (int): 選擇索引，預設為 0
        
        Returns:
            Dict[str, Any]: OpenAI 串流區塊格式的字典
        
        Examples:
            >>> chunk = client._create_streaming_chunk(
            ...     {"content": "Hello"}, "breeze-2-8b"
            ... )
        
        Raises:
            ValueError: 當參數值不合法時
        """
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": index,
                "delta": delta or {},
                "finish_reason": finish_reason
            }]
        }

    def _estimate_tokens(self, text: str) -> int:
        """簡單的 token 數量估算（粗略近似值）。
        
        Args:
            text (str): 要估算的文字
        
        Returns:
            int: 估算的 token 數量
        
        Examples:
            >>> tokens = client._estimate_tokens("Hello world")
            >>> print(tokens)  # 2
        
        Raises:
            TypeError: 當 text 不是字串時
        """
        return len(text.split())

    def _handle_non_streaming_response(self, completion_params: Dict[str, Any], 
                                     prompt_engine: MRPromptV3, model: str, prompt_tokens: int):
        """處理非串流回應。
        
        Args:
            completion_params (Dict[str, Any]): 完成請求的參數字典
            prompt_engine (MRPromptV3): 提示詞引擎實例
            model (str): 使用的模型名稱
            prompt_tokens (int): 提示詞使用的 token 數量
        
        Returns:
            ChatCompletionResponse: 聊天完成回應物件
        
        Examples:
            >>> params = {"model": "breeze-2-8b", "prompt": "Hello"}
            >>> response = client._handle_non_streaming_response(params, engine, "breeze-2-8b", 10)
        
        Raises:
            ValueError: 當回應解析失敗時
            requests.RequestException: 當 API 請求失敗時
        """
        # Call completions API
        completion = self._client.completions.create(**completion_params)
        
        # Get generated text
        generated_text = completion.choices[0].text.strip()

        # Estimate completion tokens
        completion_tokens = self._estimate_tokens(generated_text)
        
        # Parse generated text using MRPromptV3
        parsed_message = prompt_engine.parse_generated_str(generated_text)

        # Determine finish reason
        finish_reason = "tool_calls" if "tool_calls" in parsed_message else "stop"
        
        # Create OpenAI format response
        response = self._create_openai_response(generated_text, model, prompt_tokens, completion_tokens, finish_reason)
        response["choices"][0]["message"] = parsed_message
        
        return ChatCompletionResponse(response)

    def _handle_streaming_response(self, completion_params: Dict[str, Any], 
                                 prompt_engine: MRPromptV3, model: str, _prompt_tokens: int):
        """處理串流回應，支援 <|use_tool|> 偵測。
        
        Args:
            completion_params (Dict[str, Any]): 完成請求的參數字典
            prompt_engine (MRPromptV3): 提示詞引擎實例
            model (str): 使用的模型名稱
            _prompt_tokens (int): 提示詞使用的 token 數量（未使用）
        
        Returns:
            Generator: 串流 chunk 生成器
        
        Examples:
            >>> params = {"model": "breeze-2-8b", "prompt": "Hello", "stream": True}
            >>> stream = client._handle_streaming_response(params, engine, "breeze-2-8b", 10)
            >>> for chunk in stream:
            ...     print(chunk.choices[0].delta.content)
        
        Raises:
            ValueError: 當回應解析失敗時
            requests.RequestException: 當 API 請求失敗時
        """
        
        def stream_generator():
            # Call streaming completions API
            stream = self._client.completions.create(**completion_params)
            
            accumulated_text = ""
            tool_call_detected = False
            buffer = ""
            
            # Send initial chunk
            yield StreamingChunk(self._create_streaming_chunk({'role': 'assistant', 'content': ''}, model))
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].text:
                    delta_text = chunk.choices[0].text
                    accumulated_text += delta_text
                    buffer += delta_text
                    
                    # Check for tool call token
                    if "<|use_tool|>" in buffer and not tool_call_detected:
                        tool_call_detected = True
                        # Don't send any more content chunks, wait for complete generation
                        continue
                    
                    # If not in tool call mode, stream normally
                    if not tool_call_detected:
                        yield StreamingChunk(self._create_streaming_chunk({'content': delta_text}, model))
            
            # Process final accumulated text
            parsed_message = prompt_engine.parse_generated_str(accumulated_text)
            
            if tool_call_detected or "tool_calls" in parsed_message:
                # Send tool calls as a single chunk
                if "tool_calls" in parsed_message:
                    yield StreamingChunk(self._create_streaming_chunk({
                        'tool_calls': parsed_message['tool_calls']
                    }, model))
                
                # Send final chunk with finish reason
                yield StreamingChunk(self._create_streaming_chunk(None, model, finish_reason="tool_calls"))
            else:
                # Send final chunk for regular completion
                yield StreamingChunk(self._create_streaming_chunk(None, model, finish_reason="stop"))
        
        return stream_generator()

