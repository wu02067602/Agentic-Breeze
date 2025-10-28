
from typing import List, Dict, Optional

from ..prompts.prompt_manager import PromptManager
from .orchestrator_core.planning_manager import PlanningManager
from .orchestrator_core.tool_executor import ToolExecutor
from .orchestrator_core.query_rewriter import QueryRewriter
from .orchestrator_core.conversation_manager import ConversationManager
from .orchestrator_core.synthesis_generator import SynthesisGenerator

class Orchestrator:
    """
    總指揮代理。
    
    這個類別負責初始化所有核心組件，並將外部請求委託給適當的管理器處理。
    """
    
    def __init__(self, 
                 prompt_manager: PromptManager, 
                 planning_manager: PlanningManager, 
                 tool_executor: ToolExecutor, 
                 conversation_manager: ConversationManager, 
                 synthesis_generator: SynthesisGenerator,
                 query_rewriter: QueryRewriter
                 ):
        """
        初始化總指揮代理。
        
        Args:
            prompt_manager (PromptManager): 提示管理器
            planning_manager (PlanningManager): 規劃管理器
            tool_executor (ToolExecutor): 工具執行器
            conversation_manager (ConversationManager): 對話管理器
            synthesis_generator (SynthesisGenerator): 綜合生成器
            query_rewriter (QueryRewriter): 問句重寫器
        
        Returns:
            None
        
        Examples:
            >>> orchestrator = Orchestrator(
            ...     prompt_manager=prompt_mgr,
            ...     planning_manager=plan_mgr,
            ...     tool_executor=tool_exec,
            ...     conversation_manager=conv_mgr,
            ...     synthesis_generator=synth_gen,
            ...     query_rewriter=query_rw
            ... )
        
        Raises:
            TypeError: 當任何參數類型不正確時
        """
        # 初始化各個核心組件
        self.prompt_manager = prompt_manager # 提示管理器
        self.planning_manager = planning_manager # 規劃管理器
        self.tool_executor = tool_executor # 工具執行器
        self.conversation_manager = conversation_manager # 對話管理器
        self.synthesis_generator = synthesis_generator # 綜合生成器
        self.query_rewriter = query_rewriter # 問句重寫器

    def aquery(self, complex_question: str) -> str:
        """
        處理複雜查詢，執行完整的推理循環。

        Args:
            complex_question (str): 使用者的複雜查詢問題

        Returns:
            str: AI 生成的回答內容

        Examples:
            >>> orchestrator = Orchestrator(prompt_manager, planning_manager, tool_executor, conversation_manager, synthesis_generator, query_rewriter)
            >>> orchestrator.aquery("今天天氣如何？")
            "今天天氣晴朗，氣溫攝氏25度。"

        Raises:
            ValueError: 當 complex_question 不是字串類型時
            RuntimeError: 當推理過程出現無法處理的錯誤時
        """
        if not isinstance(complex_question, str):
            raise ValueError("complex_question must be a string")
        
        # 重寫問句
        rewritten_question = self.query_rewriter.rewrite_query(history=[], query=complex_question)
        # 計畫執行時需要的工具
        tools = self.planning_manager.plan_question(question=rewritten_question, history=[])
        # 執行工具
        results = self.tool_executor.execute_plan(tools)
        used_tools = [item.tool_name for item in tools.plan_items]
        # 綜合結果
        synthesis = self.synthesis_generator.synthesize_result(complex_question, results, used_tools)
        # 回傳結果
        return synthesis
    
    def aquery_with_history(self, 
                            complex_question: str, 
                            history: Optional[List[Dict[str, str]]] = None
                            ) -> str:
        """
        支援多輪對話歷史的查詢入口。

        Args:
            complex_question (str): 使用者的複雜查詢問題
            history (Optional[List[Dict[str, str]]]): 對話歷史記錄，預設為 None

        Returns:
            str: AI 生成的回答內容

        Examples:
            >>> orchestrator = Orchestrator(prompt_manager, planning_manager, tool_executor, conversation_manager, synthesis_generator, query_rewriter)
            >>> orchestrator.aquery_with_history("今天適合出門嗎？", [{"role": "user", "content": "今天天氣如何？"}, {"role": "assistant", "content": "今天天氣晴朗，氣溫攝氏25度。"}])
            "今天適合出門，氣溫攝氏25度。"

        Raises:
            ValueError: 當 complex_question 不是字串類型時
            RuntimeError: 當推理過程出現無法處理的錯誤時
        """
        if not isinstance(complex_question, str):
            raise ValueError("complex_question must be a string")
        
        # 歷史清理（避免前端傳入雜訊）
        sanitized_history = self.conversation_manager.sanitize_history(history or [], max_items=20)

        # 若為元對話，直接路由處理
        if self.conversation_manager.is_meta_question(complex_question, sanitized_history):
            return self.conversation_manager.handle_meta_conversation(complex_question, sanitized_history)

        # 一般任務流程：重寫問句 → 規劃
        rewritten_question = self.query_rewriter.rewrite_query(sanitized_history, complex_question)
        execution_plan = self.planning_manager.plan_question(rewritten_question, sanitized_history)
        # 執行工具
        results = self.tool_executor.execute_plan(execution_plan)
        used_tools = [item.tool_name for item in execution_plan.plan_items]
        # 綜合結果
        synthesis = self.synthesis_generator.synthesize_result(complex_question, results, used_tools)
        return synthesis
    
    def get_reasoning_history(self) -> List[Dict[str, str]]:
        """
        獲取完整的推理歷史記錄。

        Returns:
            List[Dict[str, str]]: 推理歷史記錄列表，每個項目包含 role 和 content

        Examples:
            >>> orchestrator = Orchestrator(prompt_manager, planning_manager, tool_executor, conversation_manager, synthesis_generator, query_rewriter)
            >>> history = orchestrator.get_reasoning_history()
            >>> isinstance(history, list)
            True
        
        Raises:
            無特定錯誤
        """
        return self.conversation_manager.get_reasoning_history()
    
    def clear_reasoning_history(self) -> None:
        """
        清空推理歷史記錄。
        
        Returns:
            None

        Examples:
            >>> orchestrator = Orchestrator(prompt_manager, planning_manager, tool_executor, conversation_manager, synthesis_generator, query_rewriter)
            >>> orchestrator.clear_reasoning_history()
            >>> len(orchestrator.get_reasoning_history())
            0
        
        Raises:
            無特定錯誤
        """
        self.conversation_manager.clear_reasoning_history()

    def aquery_with_history_stream(self, 
                                   complex_question: str, 
                                   history: Optional[List[Dict[str, str]]] = None):
        """
        支援多輪對話歷史的串流查詢入口。

        Args:
            complex_question (str): 複雜查詢
            history (Optional[List[Dict[str, str]]]): 對話歷史，預設為 None

        Yields:
            str: 串流回應內容片段
        
        Examples:
            >>> orchestrator = Orchestrator(prompt_manager, planning_manager, tool_executor, conversation_manager, synthesis_generator, query_rewriter)
            >>> for chunk in orchestrator.aquery_with_history_stream("今天天氣如何？"):
            ...     print(chunk, end='')
        
        Raises:
            ValueError: 當 complex_question 不是字串類型時
        """
        if not isinstance(complex_question, str):
            raise ValueError("complex_question must be a string")
        
        # 歷史清理（避免前端傳入雜訊）
        sanitized_history = self.conversation_manager.sanitize_history(history or [], max_items=20)

        # 若為元對話，直接路由處理（不支援串流）
        if self.conversation_manager.is_meta_question(complex_question, sanitized_history):
            yield self.conversation_manager.handle_meta_conversation(complex_question, sanitized_history)
            return

        # 一般任務流程：重寫問句 → 規劃
        yield "🔄 正在分析您的問題..."
        rewritten_question = self.query_rewriter.rewrite_query(sanitized_history, complex_question)
        
        yield " 完成\n🔄 正在制定執行計畫..."
        execution_plan = self.planning_manager.plan_question(rewritten_question, sanitized_history)
        
        # 執行工具
        if execution_plan.plan_items:
            yield " 完成\n🔄 正在執行相關工具..."
            results = self.tool_executor.execute_plan(execution_plan)
            used_tools = [item.tool_name for item in execution_plan.plan_items]
            yield " 完成\n\n"
        else:
            results = []
            used_tools = []
            yield " 完成\n\n"
        
        # 綜合結果（串流版本）
        yield from self.synthesis_generator.synthesize_result_stream(complex_question, results, used_tools)
