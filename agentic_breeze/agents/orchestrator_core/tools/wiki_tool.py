# wiki_tool.py
from typing import Optional, List
from pydantic import BaseModel, Field
from .api_tool import APIRequestTool
import json
import logging

_logger = logging.getLogger(__name__)

class WikiSearchParams(BaseModel):
    """Wikipedia 搜尋工具的參數"""
    query: str = Field(..., description="要在 Wikipedia 上搜尋的關鍵字或短語。")
    limit: int = Field(10, description="要回傳的搜尋結果數量。預設為 10。", ge=1, le=50)

class WikiSmartContentParams(BaseModel):
    """Wikipedia 智慧摘要工具的參數"""
    query: str = Field(..., description="要搜尋並獲取摘要的關鍵字或短語。")
    limit: int = Field(5, description="搜尋時的結果數量限制。預設為 5。", ge=1, le=10)

class WikiTool:
    """
    使用 Wikipedia 官方 MediaWiki API 的查詢工具。
    """

    def __init__(self, api_request_tool: APIRequestTool, lang: str = "zh", user_agent: Optional[str] = None):
        self.api_request_tool = api_request_tool
        self.base_url = f"https://{lang}.wikipedia.org/w/api.php"
        self.headers = {"User-Agent": user_agent} if user_agent else None

    def search(self, query: str, limit: int = 10) -> str:
        """
        在 Wikipedia 上搜尋條目。適用於查找實體名稱或主題。

        參數:
            - query (str): 搜尋關鍵字或短語。
            - limit (int): 回傳結果數量上限（1~50）。

        返回:
            - str: JSON 字串，格式如 {"status":"success","titles": [...], "raw": [...]} 或 {"status":"error","message": "..."}

        使用範例:
            >>> tool = WikiTool(api_request_tool=APIRequestTool())
            >>> res = json.loads(tool.search("台灣"))
            >>> res["status"]
            'success'

        可能觸發的錯誤:
            - 上游 HTTP 錯誤（會以 {"status":"error","message": "..."} 回傳）
            - 回應結構異常（以 error JSON 字串回傳）
        """
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
            "formatversion": 2
        }
        res = self.api_request_tool.execute_request(
            method="GET",
            url=self.base_url,
            query_params=params,
            headers=self.headers
        )
        _logger.debug("wiki_search API 原始回應: %s", json.dumps(res['response_body'], indent=2, ensure_ascii=False))
        if res["error"]:
            return json.dumps({"status": "error", "message": res["error"]})

        try:
            items = res["response_body"]["query"]["search"]
            titles = [it["title"] for it in items]
            return json.dumps({"status": "success", "titles": titles, "raw": items})
        except (KeyError, TypeError, IndexError) as e:
            return json.dumps({"status": "error", "message": f"Unexpected response format: {e}"})

    def get_page_info(self, title: str) -> str:
        """
        獲取指定頁面的詳細資訊，包括頁面 ID、標題與 URL。

        參數:
            - title (str): 頁面標題。

        返回:
            - str: JSON 字串，格式如 {"status":"success","page": {...}} 或 {"status":"error","message": "..."}

        使用範例:
            >>> tool = WikiTool(api_request_tool=APIRequestTool())
            >>> res = json.loads(tool.get_page_info("台灣"))
            >>> res.get("status") in {"success","error"}
            True

        可能觸發的錯誤:
            - 上游 HTTP 錯誤（以 error JSON 字串回傳）
            - 回應結構異常（以 error JSON 字串回傳）
        """
        params = {
            "action": "query",
            "prop": "info",
            "inprop": "url",
            "titles": title,
            "format": "json",
            "formatversion": 2,
            "redirects": 1,
        }
        res = self.api_request_tool.execute_request(
            method="GET",
            url=self.base_url,
            query_params=params,
            headers=self.headers
        )
        if res["error"]:
            return json.dumps({"status": "error", "message": res["error"]})

        try:
            pages = res["response_body"]["query"]["pages"]
            if not pages:
                return json.dumps({"status": "error", "message": f"No page info for: {title}"})
            
            page = None
            if isinstance(pages, dict):
                page_id = next(iter(pages)) # Get the first (and usually only) page ID
                page = pages[page_id]
            elif isinstance(pages, list) and len(pages) > 0:
                page = pages[0]
            
            if page is None:
                return json.dumps({"status": "error", "message": f"Unexpected page structure for: {title}"})

            return json.dumps({"status": "success", "page": page})
        except (KeyError, TypeError, IndexError) as e:
            return json.dumps({"status": "error", "message": f"Unexpected response format for page info: {e}"})

    def get_full_content(self, title: str) -> str:
        """
        獲取指定頁面的完整原始維基文本內容。

        參數:
            - title (str): 頁面標題。

        返回:
            - str: JSON 字串，格式如 {"status":"success","content": "..."} 或 {"status":"error","message": "..."}

        使用範例:
            >>> tool = WikiTool(api_request_tool=APIRequestTool())
            >>> res = json.loads(tool.get_full_content("台灣"))
            >>> res.get("status") in {"success","error"}
            True

        可能觸發的錯誤:
            - 上游 HTTP 錯誤（以 error JSON 字串回傳）
            - 回應結構異常（以 error JSON 字串回傳）
        """
        params = {
            "action": "query",
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "titles": title,
            "format": "json",
            "formatversion": 2,
            "redirects": 1,
        }
        res = self.api_request_tool.execute_request(
            method="GET",
            url=self.base_url,
            query_params=params,
            headers=self.headers
        )
        _logger.debug("wiki_get_full_content API 原始回應: %s", json.dumps(res['response_body'], indent=2, ensure_ascii=False))
        if res["error"]:
            return json.dumps({"status": "error", "message": res["error"]})

        try:
            pages = res["response_body"]["query"]["pages"]
            if not pages:
                return json.dumps({"status": "error", "message": f"No content for: {title}"})

            page_data = None
            if isinstance(pages, dict):
                page_id = next(iter(pages)) # Get the first (and usually only) page ID
                page_data = pages[page_id]
            elif isinstance(pages, list) and len(pages) > 0:
                page_data = pages[0]

            if page_data is None:
                return json.dumps({"status": "error", "message": f"Unexpected page structure for content: {title}"})

            # 提取完整維基文本內容
            revisions = page_data.get("revisions", [{}])
            full_content = revisions[0].get("slots", {}).get("main", {}).get("content")
            if full_content is None:
                return json.dumps({"status": "error", "message": f"Could not find full content for: {title}"})

            return json.dumps({"status": "success", "content": full_content})
        except (KeyError, TypeError, IndexError) as e:
            return json.dumps({"status": "error", "message": f"Unexpected response format for full content: {e}"})

    def smart_content(self, query: str, limit: int = 2) -> str:
        """
        根據關鍵字搜尋 Wikipedia，彙整候選頁面的標題、URL 與原始內容。

        參數:
            - query (str): 搜尋關鍵字。
            - limit (int): 搜尋候選數量。

        返回:
            - str: JSON 字串，格式如 {"status":"success","results": [{"title":"...","url":"...","content":"..."}, ...]}

        使用範例:
            >>> tool = WikiTool(api_request_tool=APIRequestTool())
            >>> res = json.loads(tool.smart_content("台灣", limit=2))
            >>> res.get("status") in {"success","error"}
            True

        可能觸發的錯誤:
            - 上游 HTTP 錯誤（以 error JSON 字串回傳）
            - 回應結構異常（以 error JSON 字串回傳）
        """


        # ... 這裡使用原始的實作邏輯 ...
        s = self.search(query, limit=limit)
        # --- 添加除錯輸出 --- start
        search_results = json.loads(s)
        if search_results.get("status") == "success":
            _logger.debug("wiki_smart_content 搜尋到的標題: %s", search_results.get('titles', []))
        # --- 添加除錯輸出 --- end
        if search_results.get("status") != "success":
            return s
        titles: List[str] = search_results.get("titles", [])
        if not titles:
            return json.dumps({"status": "error", "message": f"No result for: {query}"})

        all_contents = []
        for title in titles:
            info = self.get_page_info(title)
            info_obj = json.loads(info)
            if info_obj.get("status") != "success":
                # 如果某個條目獲取失敗，可以選擇跳過或記錄錯誤，這裡選擇跳過
                continue
            content_text = self.get_full_content(title)
            content_obj = json.loads(content_text)
            if content_obj.get("status") != "success":
                continue
            
            all_contents.append({
                "title": info_obj.get("page", {}).get("title"),
                "url": info_obj.get("page", {}).get("fullurl"),
                "content": content_obj.get("content")
            })
        
        if not all_contents:
            return json.dumps({"status": "error", "message": f"無法為 {query} 獲取任何有效摘要。"})

        return json.dumps({"status": "success", "results": all_contents})
