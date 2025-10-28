
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class ReasoningStep:
    """
    推理步驟的資料結構。
    
    用於記錄 AI Agent 執行過程中的每個推理步驟，包含步驟類型、使用的工具、參數和結果。
    
    Attributes:
        step_type (str): 推理步驟類型，可為 "planning"、"execution" 或 "synthesis"
        tool_name (Optional[str]): 使用的工具名稱，若未使用工具則為 None
        parameters (Optional[Dict[str, Any]]): 工具執行參數，若無參數則為 None
        result (Optional[str]): 步驟執行結果，若無結果則為 None
        error (Optional[str]): 執行錯誤訊息，若無錯誤則為 None
        raw_tool_result (Optional[Any]): 工具執行的原始結果，可為任意類型
    
    Examples:
        >>> step = ReasoningStep(
        ...     step_type="execution",
        ...     tool_name="weather_tool",
        ...     parameters={"location": "台北市"},
        ...     result="台北市今日天氣晴朗"
        ... )
    """
    step_type: str  # "planning", "execution", "synthesis"
    tool_name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    result: Optional[str] = None
    error: Optional[str] = None
    raw_tool_result: Optional[Any] = None

@dataclass
class ExecutionPlan:
    """
    執行計畫的資料結構。
    
    用於儲存 AI Agent 的執行計畫，包含需要執行的所有計畫項目和計畫描述。
    
    Attributes:
        plan_items (List[Dict[str, Any]]): 執行計畫的項目列表，每個項目為一個字典
        description (str): 執行計畫的描述，預設為空字串
    
    Examples:
        >>> plan = ExecutionPlan(
        ...     plan_items=[{"tool_name": "weather_tool", "arguments": {"location": "台北市"}}],
        ...     description="查詢台北市天氣"
        ... )
    """
    plan_items: List[Dict[str, Any]]
    description: str = ""
