
from typing import List, Dict, Optional

from ...llm.llm_client import LLMConnector


class ConversationManager:
    """
    對話管理器：專注於對話前處理與路由決策。
    - 歷史清理（sanitize）
    - 元對話判斷（is_meta_question）
    - 問題釐清（clarify_question_with_history）
    - 元對話處理（handle_meta_conversation）
    """

    def __init__(self, llm_client: LLMConnector):
        """
        初始化對話管理器。

        Args:
            llm_client (LLMConnector): LLM 客戶端，用於問題釐清和元對話的回應生成
        
        Returns:
            None
        
        Examples:
            >>> from agentic_breeze.llm.llm_client import LLMConnector
            >>> llm = LLMConnector()
            >>> manager = ConversationManager(llm)
            >>> isinstance(manager, ConversationManager)
            True
        
        Raises:
            TypeError: 當 llm_client 不是 LLMConnector 實例時
        """
        if not isinstance(llm_client, LLMConnector):
            raise TypeError("llm_client must be an instance of LLMConnector")
        self.llm_client = llm_client
        self._reasoning_history: List[Dict[str, str]] = []

    def sanitize_history(self, history: Optional[List[Dict[str, str]]], max_items: int = 10) -> List[Dict[str, str]]:
        """
        清理前端傳入的對話歷史，僅保留必要欄位並限制長度。
        - 僅保留 role in {user, assistant}
        - 轉字串、去除空白；空內容移除
        - 僅保留最後 max_items 筆
        
        Args:
            history (Optional[List[Dict[str, str]]]): 對話歷史記錄，預設為 None
            max_items (int): 最多保留的對話數量，預設為 10
        
        Returns:
            List[Dict[str, str]]: 清理後的對話歷史
        
        Examples:
            >>> manager = ConversationManager(llm_client)
            >>> history = [{"role": "user", "content": "你好"}]
            >>> cleaned = manager.sanitize_history(history, max_items=5)
            >>> len(cleaned) <= 5
            True
        
        Raises:
            無特定錯誤
        """
        history = history or []
        cleaned: List[Dict[str, str]] = []
        for m in history:
            role = str(m.get("role", "")).lower()
            content = str(m.get("content", "")).strip()
            if not content:
                continue
            role = "user" if role == "user" else "assistant"
            cleaned.append({"role": role, "content": content})
        if len(cleaned) > max_items:
            cleaned = cleaned[-max_items:]
        return cleaned

    def is_meta_question(self, question: str, history: Optional[List[Dict[str, str]]] = None) -> bool:
        """
        判斷是否為元對話（META）。使用輕量關鍵詞判斷，必要時以 LLM 確認。

        Args:
            question: str, 使用者問題。
            history: Optional[List[Dict[str, str]]], 對話歷史。

        Returns:
            bool, 是否為元對話。

        Examples:
            >>> conversation_manager = ConversationManager(llm_client)
            >>> conversation_manager.is_meta_question("你是誰？")
            True
            >>> conversation_manager.is_meta_question("推薦好吃的餐廳")
            False
        
        Raises:
            無特定錯誤
        """
        q = (question or "").strip()
        if not q:
            return False
        
        # 註解：原本的 LLM 判斷邏輯
        prompt = (
            "請判斷以下問題是否屬於元對話（關於助理/對話/說明/摘要/重述）。\n"
            "只輸出一個詞：META 或 TASK。\n"
            f"問題：{q}"
        )
        label = (self.llm_client.single_query(prompt) or "").strip().upper()
        return label.startswith("META")

    def clarify_question_with_history(self, question: str, 
                                     history: List[Dict[str, str]]) -> str:
        """
        基於對話歷史，釐清當前問題，使其更具體、適合工具查詢。僅輸出一個最清楚、最具體的一句話。

        Args:
            question: str, 當前問題。
            history: List[Dict[str, str]], 對話歷史。

        Returns:
            str, 釐清後的問題。

        Examples:
            >>> conversation_manager = ConversationManager(llm_client)
            >>> history = [{"role": "user", "content": "今天天氣如何？"}, {"role": "assistant", "content": "今天天氣晴朗。"}]
            >>> conversation_manager.clarify_question_with_history("那明天呢？", history)
            "明天的天氣如何？"
        
        Raises:
            無特定錯誤
        """
        system_instruction = (
            "請分析使用者的最後問題是否為獨立的新問題，或是延續性問題。"
            "若為獨立新問題，直接輸出該問題；若為延續性問題，可結合必要脈絡重寫。"
            "請將使用者的最後問題重寫成一個最清楚、最具體、適合工具查詢的一句話。只輸出那一句話，不要多餘解釋。"
        )
        clarification_messages: List[Dict[str, str]] = []
        clarification_messages.append({"role": "system", "content": system_instruction})
        clarification_messages.extend(history or [])
        clarification_messages.append({"role": "user", "content": question.strip()})
        clarified_question = (self.llm_client.chat_with_history(clarification_messages) or "").strip()
        return clarified_question or question.strip()
    
    def handle_meta_conversation(self, complex_question: str, history: List[Dict[str, str]]) -> str:
        """
        處理元對話問題，直接透過 LLM 生成回應。

        Args:
            complex_question: str, 複雜問題。
            history: List[Dict[str, str]], 對話歷史。

        Returns:
            str, LLM 生成的回應。

        Examples:
            >>> conversation_manager = ConversationManager(llm_client)
            >>> history = [{"role": "user", "content": "你好"}]
            >>> conversation_manager.handle_meta_conversation("你是誰？", history)
            "我是AI助理，很高興為您服務。"
        
        Raises:
            無特定錯誤
        """
        meta_messages: List[Dict[str, str]] = []
        meta_messages.extend(history or [])
        meta_messages.append({"role": "user", "content": complex_question.strip()})
        final_answer = (self.llm_client.chat_with_history(meta_messages) or "").strip()
        return final_answer

    def get_reasoning_history(self) -> List[Dict[str, str]]:
        """
        獲取推理歷史記錄。
        
        Returns:
            List[Dict[str, str]]: 推理歷史記錄的副本
        
        Examples:
            >>> manager = ConversationManager(llm_client)
            >>> history = manager.get_reasoning_history()
            >>> isinstance(history, list)
            True
        
        Raises:
            無特定錯誤
        """
        return list(self._reasoning_history)

    def clear_reasoning_history(self) -> None:
        """
        清空推理歷史記錄。
        
        Returns:
            None
        
        Examples:
            >>> manager = ConversationManager(llm_client)
            >>> manager.clear_reasoning_history()
            >>> len(manager.get_reasoning_history())
            0
        
        Raises:
            無特定錯誤
        """
        self._reasoning_history.clear()
