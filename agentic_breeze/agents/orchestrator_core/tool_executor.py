
from typing import List
import concurrent.futures

from ...registry.tool_registry import ToolRegistry
from .modle.execution_plan import ExecutionPlan, PlanItem


class ToolExecutor:
    """
    工具執行器，負責工具的並行執行與結果收集。
    
    這個類別專門處理工具的實際執行邏輯，
    包括並行處理、錯誤處理和結果收集。
    """
    
    def __init__(self, tool_registry: ToolRegistry):
        """
        初始化工具執行器。
        
        Args:
            tool_registry (ToolRegistry): 工具註冊表，用於執行具體工具
        
        Returns:
            None
        
        Examples:
            >>> from agentic_breeze.registry.tool_registry import ToolRegistry
            >>> registry = ToolRegistry()
            >>> executor = ToolExecutor(registry)
            >>> isinstance(executor, ToolExecutor)
            True
        
        Raises:
            TypeError: 當 tool_registry 不是 ToolRegistry 實例時
        """
        if not isinstance(tool_registry, ToolRegistry):
            raise TypeError("tool_registry must be an instance of ToolRegistry")
        
        self.tool_registry = tool_registry
    
    def execute_plan(self, execution_plan: ExecutionPlan) -> List[str]:
        """
        執行計畫中的所有工具調用。
        
        Args:
            execution_plan (ExecutionPlan): 要執行的計畫物件
            
        Returns:
            List[str]: 所有工具執行的結果列表
        
        Examples:
            >>> from agentic_breeze.agents.orchestrator_core.modle.execution_plan import ExecutionPlan, PlanItem
            >>> plan = ExecutionPlan(plan_items=[PlanItem(tool_name="test", arguments={})], description="test")
            >>> executor = ToolExecutor(tool_registry)
            >>> results = executor.execute_plan(plan)
            >>> isinstance(results, list)
            True
            
        Raises:
            RuntimeError: 當執行過程出現無法處理的嚴重錯誤時
        """
        if not execution_plan.plan_items:
            return []
        
        execution_results = []
        
        # 這裡設定為5個工作執行緒進行並行處理
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # 提交所有工具執行任務
            futures = [
                executor.submit(self._execute_single_tool, plan_item, i)
                for i, plan_item in enumerate(execution_plan.plan_items)
            ]
            
            # 收集所有結果
            for future in futures:
                execution_results.append(future.result())
                
        return execution_results
    
    def _execute_single_tool(self, plan_item: PlanItem, index: int) -> str:
        """
        執行單一工具調用並更新推理步驟。
        
        Args:
            plan_item (PlanItem): 單一工具調用的計畫項目
            index (int): 工具在計畫中的索引，用於日誌記錄
            
        Returns:
            str: 工具執行的結果字串，或錯誤訊息
        
        Examples:
            >>> from agentic_breeze.agents.orchestrator_core.modle.execution_plan import PlanItem
            >>> item = PlanItem(tool_name="test_tool", arguments={})
            >>> result = executor._execute_single_tool(item, 0)
            >>> isinstance(result, str)
            True
        
        Raises:
            無特定錯誤（錯誤會轉換為結果字串返回）
        """
        
        tool_name = plan_item.tool_name
        parameters = plan_item.arguments
        
        print(f"執行工具 {index+1}: {tool_name} 參數: {parameters}")
        
        if not tool_name:
            error_msg = "工具名稱缺失"
            return "錯誤: 工具名稱缺失"
        
        # 透過 ToolRegistry 執行工具
        result = self.tool_registry.execute_tool(tool_name, **parameters)
        print(f"工具 {index+1} 執行結果: {str(result)[:100]}...")
        
        return result
