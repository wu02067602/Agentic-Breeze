from typing import List, Dict

from ...llm.llm_client import LLMConnector
from ...prompts.prompt_manager import PromptManager

class QueryRewriter:
    """
    問句重寫器。用於重寫複雜問句，使其更符合查詢需求。

    Attributes:
        llm_client: LLMConnector，用於向LLM發出請求。    
        prompt_manager: PromptManager，用於管理提示詞。

    Methods:
        rewrite_query(query: str) -> str: 重寫問句。
    """
    
    def __init__(self, llm_client: LLMConnector, prompt_manager: PromptManager):
        """
        初始化問句重寫器。

        Args:
            llm_client (LLMConnector): 用於向 LLM 發出請求的客戶端
            prompt_manager (PromptManager): 用於管理提示詞的管理器
        
        Returns:
            None
        
        Examples:
            >>> from agentic_breeze.llm.llm_client import LLMConnector
            >>> from agentic_breeze.prompts.prompt_manager import PromptManager
            >>> llm = LLMConnector()
            >>> pm = PromptManager()
            >>> rewriter = QueryRewriter(llm, pm)
            >>> isinstance(rewriter, QueryRewriter)
            True
        
        Raises:
            TypeError: 當參數類型不正確時
        """
        self.llm_client = llm_client # LLMConnector，用於向LLM發出請求。
        self.prompt_manager = prompt_manager # PromptManager，用於管理提示詞。

    def rewrite_query(self, history: List[Dict[str, str]], query: str) -> str:
        """
        重寫問句。使問句更符合查詢需求。

        Args:
            history: List[Dict[str, str]], 對話歷史。
            query: str, 複雜問句。

        Returns:
            str, 重寫後的問句。

        Examples:
            >>> query_rewriter = QueryRewriter(llm_client, prompt_manager)
            >>> query_rewriter.rewrite_query([{"role": "user", "content": "今天天氣如何？"}, {"role": "assistant", "content": "今天天氣晴朗，氣溫攝氏25度。"}], "今天適合出門嗎？")
            "今天的天氣與氣溫適合出門嗎？"
        
        Raises:
            無特定錯誤（若重寫失敗則返回原始問句）
        """
        prompt = self.prompt_manager.build_query_rewriter_prompt(history, query)
        rewritten_query = self.llm_client.single_query(prompt)

        if not rewritten_query or rewritten_query.strip() == "":
            return query

        return rewritten_query
        
