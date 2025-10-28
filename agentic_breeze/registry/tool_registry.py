from typing import List, Dict, Any, Callable
import json
from agentic_breeze.agents.orchestrator_core.tools.sqlite_tool import SQLiteSchemaTool
from agentic_breeze.agents.orchestrator_core.tools.weather import CWAWeatherTool
from agentic_breeze.agents.orchestrator_core.tools.wiki_tool import WikiTool
from agentic_breeze.agents.orchestrator_core.tools.api_tool import APIRequestTool

# 錯誤訊息前綴，提供一致且可辨識的錯誤格式。
ERROR_PREFIX = "[ToolError]"


class ToolRegistry:
    """
    工具註冊表（Tool Registry）。

    本類別提供工具的註冊、查詢與執行功能，並可輸出符合 OpenAI
    function calling（tools schema）之工具定義，以供上層模組（例如
    `PlanningManager` 與 `LLMConnector`）進行規劃與推理。

    使用情境：
        - `PlanningManager` 透過 `get_llm_tool_schemas()` 取得可用工具 schema，
          交給 LLM 作為可呼叫之函式定義。
        - `ToolExecutor` 透過 `execute_tool()` 依計畫項目執行對應工具。

    使用範例：
        1) 註冊自訂工具並輸出 tools schema：
            >>> registry = ToolRegistry()
            >>> registry.register_tool(
            ...     name="add",
            ...     description="加總兩個整數並回傳字串結果",
            ...     parameters={
            ...         "type": "object",
            ...         "properties": {
            ...             "a": {"type": "integer"},
            ...             "b": {"type": "integer"}
            ...         },
            ...         "required": ["a", "b"],
            ...         "additionalProperties": False,
            ...     },
            ...     handler=lambda a, b: str(a + b),
            ... )
            >>> tools = registry.get_llm_tool_schemas()
            >>> isinstance(tools, list) and tools[0]["type"] == "function"
            True

        2) 直接執行已註冊工具：
            >>> registry.execute_tool("add", a=1, b=2)
            '3'

    內部資料結構：
        self._registry: Dict[str, Dict[str, Any]]
            - key：工具名稱（str）
            - value：包含下列鍵值
                - "schema": Dict[str, Any]
                    符合 OpenAI tools 規範之函式定義
                - "handler": Callable[..., Any]
                    實際執行該工具的函式

        結構範例（示意）：
            {
                "echo": {
                    "schema": {
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "description": "回傳輸入文字（教學用範例）",
                            "parameters": {
                                "type": "object",
                                "properties": {"text": {"type": "string"}},
                                "required": ["text"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "handler": <callable>
                },
                "add": {
                    "schema": { ... },
                    "handler": <callable>
                }
            }

    內建示例工具：
        （無內建示例工具。請於系統啟動時或模組載入時主動註冊所需工具。）
    """

    def __init__(self) -> None:
        """
        初始化工具註冊表。
        
        建立空的工具註冊表並註冊所有預設工具（SQLite、天氣、維基百科等）。
        
        Args:
            無參數
        
        Returns:
            None
        
        Examples:
            >>> registry = ToolRegistry()
            >>> len(registry.get_llm_tool_schemas()) > 0
            True
        
        Raises:
            RuntimeError: 當預設工具註冊失敗時
        """
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._register_default_tools() # 呼叫註冊預設工具
        

    def _register_default_tools(self) -> None:
        """
        註冊預設工具，包含 SQLite、天氣、維基百科等工具。
        
        此方法會自動註冊系統內建的所有預設工具：
        - SQLite 相關工具（定義表、查詢、獲取表資訊）
        - 中央氣象署天氣預報工具
        - 維基百科搜尋工具
        
        Args:
            無參數
        
        Returns:
            None
        
        Examples:
            >>> registry = ToolRegistry()
            >>> # _register_default_tools 會在 __init__ 中自動調用
            >>> tools = registry.get_llm_tool_schemas()
            >>> any('sqlite' in tool['function']['name'] for tool in tools)
            True
        
        Raises:
            ImportError: 當無法匯入所需的工具類別時
            ValueError: 當工具註冊參數不正確時
        """
        # -------------------- 註冊 SQLiteSchemaTool ---------------------
        sqlite_tool_instance = SQLiteSchemaTool(db_path="sample_users.db")

        # -------------------- 註冊 API ---------------------
        api_request_tool_instance = APIRequestTool() 
        cwa_weather_tool_instance = CWAWeatherTool(api_request_tool=api_request_tool_instance)
        wiki_tool_instance = WikiTool(api_request_tool=api_request_tool_instance)

        # define_table (僅用於遷移現有表，不創建新表)
        column_definition_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "欄位名稱。"},
                "dtype": {"type": "string", "enum": ["TEXT", "INTEGER", "REAL", "BLOB", "NUMERIC"], "description": "SQLite 資料型別。"},
                "nullable": {"type": "boolean", "description": "是否允許 NULL 值。預設為 True。"},
                "is_primary_key": {"type": "boolean", "description": "是否為主鍵。預設為 False。"},
                "is_unique": {"type": "boolean", "description": "是否為唯一約束。預設為 False。"},
                "default": {"type": "string", "description": "欄位的預設值。"},
                "description": {"type": "string", "description": "欄位的描述。"},
            },
            "required": ["name", "dtype"],
            "additionalProperties": False,
        }

        table_definition_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "資料表名稱。"},
                "columns": {
                    "type": "array",
                    "items": column_definition_schema,
                    "description": "ColumnDefinition 物件列表，定義資料表的欄位。"
                },
                "description": {"type": "string", "description": "資料表的描述。"},
            },
            "required": ["name", "columns"],
            "additionalProperties": False,
        }

        self.register_tool(
            name="sqlite_define_table",
            description="根據 dataclass 定義的 schema 遷移現有資料表。此工具不允許創建新表，只會對已存在的表追加欄位。",
            parameters={
                "type": "object",
                "properties": {
                    "table_definition": table_definition_schema
                },
                "required": ["table_definition"],
                "additionalProperties": False,
            },
            handler=sqlite_tool_instance.define_table,
        )

        # query_to_dicts
        self.register_tool(
            name="sqlite_query_to_dicts",
            description="執行 SELECT 語句並回傳結果為字典列表。",
            parameters={
                "type": "object",
                "properties": {
                    "sql_query": {"type": "string", "description": "要執行的 SELECT SQL 查詢字串。"},
                    "params": {
                        "type": "array",
                        "description": "SQL 查詢的參數，可以是列表、元組或物件。",
                        "items": {"type": "object"}
                    }
                },
                "required": ["sql_query"],
                "additionalProperties": False,
            },
            handler=sqlite_tool_instance.query_to_dicts,
        )

        # get_table_info
        self.register_tool(
            name="sqlite_get_table_info",
            description="獲取指定資料表的詳細資訊，包括欄位、型別等。",
            parameters={
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "要獲取資訊的資料表名稱。"}
                },
                "required": ["table_name"],
                "additionalProperties": False,
            },
            handler=sqlite_tool_instance.get_table_info,
        )

        # get_national_forecast
        self.register_tool(
            name="cwa_get_national_forecast",
            description="獲取台灣的氣象資訊。資訊來源於中央氣象署 (CWA) 的全國天氣預報資料，可選擇篩選特定地點。",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "CWA 資料集的 ID，預設為 'F-C0032-001' (36 小時天氣預報)。", "default": "F-C0032-001"},
                    "location_name": {"type": "string", "description": "可選的台灣城市名稱，盡可能簡單表示，如果提供，將只回傳該地點的預報。請使用標準中文行政區劃名稱，例如：臺北市、臺南市、高雄市、新北市、桃園市、台中市、台南市、台東市、花蓮縣、宜蘭縣、金門縣、連江縣、澎湖縣。"}
                },
                "required": [],
                "additionalProperties": False,
            },
            handler=cwa_weather_tool_instance.get_national_forecast,
        )

        # smart_content
        self.register_tool(
            name="wiki_smart_content",
            description="根據關鍵字搜尋 Wikipedia，並回傳最佳匹配頁面的標題、URL 和解說。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要在 Wikipedia 上搜尋的關鍵字或短語。"},
                    "limit": {"type": "integer", "description": "要回傳的搜尋結果數量。預設為 2。", "default": 2, "minimum": 1, "maximum": 10}
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=wiki_tool_instance.smart_content,
        )

    def register_tool(
        self,
        *,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """
        註冊工具。

        於登錄表中新增一個可供 LLM 呼叫與系統執行的工具定義。工具名稱
        應具唯一性；若同名將覆蓋原有設定。

        Args:
            name (str): 工具名稱（唯一），不可為空
            description (str): 工具用途與行為的說明，供 LLM 判斷何時調用
            parameters (Dict[str, Any]): JSON Schema 物件，對應 OpenAI function calling 的
                parameters 欄位，用於描述參數型態與必填欄位
            handler (Callable[..., Any]): 實際執行函式，呼叫簽名為 handler(**kwargs)，
                需與 parameters 定義相容
        
        Returns:
            None

        Examples:
            >>> registry = ToolRegistry()
            >>> registry.register_tool(
            ...     name="add",
            ...     description="加總兩個整數並回傳字串結果",
            ...     parameters={
            ...         "type": "object",
            ...         "properties": {
            ...             "a": {"type": "integer"},
            ...             "b": {"type": "integer"}
            ...         },
            ...         "required": ["a", "b"]
            ...     },
            ...     handler=lambda a, b: str(a + b)
            ... )
            >>> result = registry.execute_tool("add", a=1, b=2)
            >>> result
            '3'

        Raises:
            ValueError: 當 name 為空或 handler 不可呼叫時
        """
        if not name or not isinstance(name, str):
            raise ValueError("tool name 不可為空，且需為字串。")
        if not callable(handler):
            raise ValueError("handler 必須可呼叫。")

        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }

        self._registry[name] = {"schema": schema, "handler": handler}

    def get_llm_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        取得工具 schema 列表。

        回傳符合 OpenAI function calling 規格之工具定義，用以作為 LLM 的
        tools 參數。每一個元素皆為 {"type": "function", "function": {...}}。

        Args:
            無參數

        Returns:
            List[Dict[str, Any]]: 可用工具的 schema 清單，每個元素符合 OpenAI tools 格式
        
        Examples:
            >>> registry = ToolRegistry()
            >>> schemas = registry.get_llm_tool_schemas()
            >>> isinstance(schemas, list)
            True
            >>> all(schema.get("type") == "function" for schema in schemas)
            True

        Raises:
            RuntimeError: 當獲取 schema 時發生錯誤
        """
        return [entry["schema"] for entry in self._registry.values()]

    # 設計說明（參數位置/關鍵字限制）：
    # - '/'：將 tool_name 設為「位置限定參數」，避免與工具本身參數同名而混淆，並提升可讀性（第一個位置就是工具名）。
    # - '*'：其後僅允許「關鍵字限定參數」；框架旗標（如 raise_on_error）需以名稱傳遞。
    #   工具的實際參數一律收在 **kwargs，原樣轉交 handler，明確區分框架層與工具層參數。
    def execute_tool(self, tool_name: str, /, *, raise_on_error: bool = False, **kwargs: Any) -> str:
        """
        執行工具。

        依據工具名稱查找對應處理函式，並以關鍵字參數方式傳入執行。若執行
        成功且回傳值為非字串，將嘗試以 JSON 進行序列化；失敗時回傳具固定
        前綴之錯誤訊息，或於嚴格模式下拋出例外。

        Args:
            tool_name (str): 工具名稱，必須是已註冊的工具
            raise_on_error (bool): 為 True 時遇到錯誤即拋出例外，為 False 時回傳
                具固定前綴之錯誤字串，預設 False
            **kwargs (Any): 傳遞給工具處理函式的參數，應符合工具 schema 定義

        Returns:
            str: 工具回傳內容之字串表示，若為非字串則會轉為 JSON 或 str

        Examples:
            >>> registry = ToolRegistry()
            >>> registry.register_tool(
            ...     name="echo",
            ...     description="回傳輸入文字",
            ...     parameters={"type": "object", "properties": {"text": {"type": "string"}}},
            ...     handler=lambda text: text
            ... )
            >>> registry.execute_tool("echo", text="Hello")
            'Hello'
            >>> result = registry.execute_tool("non_existent_tool")
            >>> "[ToolError]" in result
            True

        Raises:
            KeyError: 當 raise_on_error=True 且工具未註冊時
            TypeError: 當 raise_on_error=True 且參數不符時
            ValueError: 當 raise_on_error=True 且參數驗證失敗時

        Note:
            - 本實作著重於最小可用，未內建 JSON Schema 驗證
            - 保留最後一道通用例外攔截，提供穩定回傳格式與防止流程中斷
            - 必要時可透過 raise_on_error 開啟嚴格模式
        """
        entry = self._registry.get(tool_name)
        if not entry:
            msg = f"{ERROR_PREFIX} 未註冊的工具: {tool_name}"
            if raise_on_error:
                raise KeyError(msg)
            return msg

        handler: Callable[..., Any] = entry["handler"]
        try:
            result = handler(**kwargs)
        except TypeError as e:
            # 常見於參數不符或缺少必填參數
            msg = f"{ERROR_PREFIX} 使用工具 '{tool_name}' 時參數錯誤: {e}"
            if raise_on_error:
                raise
            return msg
        except ValueError as e:
            # 若未來加入參數驗證（JSON Schema/Pydantic）
            msg = f"{ERROR_PREFIX} 使用工具 '{tool_name}' 時參數驗證失敗: {e}"
            if raise_on_error:
                raise
            return msg
        except Exception as e:
            # 最後保險攔截：提供穩定錯誤格式並避免流程中斷
            msg = f"{ERROR_PREFIX} 使用工具 '{tool_name}' 時執行例外: {e}"
            if raise_on_error:
                raise msg
            return msg

        if isinstance(result, str):
            return result

        try:
            if isinstance(result, (dict, list, int, float, bool)) or result is None:
                return json.dumps(result, ensure_ascii=False)
            # 對於非常規型別，改以 str() 以避免洩漏內部資訊
            return str(result)
        except Exception:
            # 序列化出錯時退回 str()，保證回傳文字格式
            return str(result)