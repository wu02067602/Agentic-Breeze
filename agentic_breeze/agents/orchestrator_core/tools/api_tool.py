import httpx
from typing import Dict, Any, Optional, Literal
import json
import logging

class APIRequestTool:
    """
    一個可重複使用的 API 請求工具，用於 AI Agent 發送 HTTP 請求。
    支援 GET, POST, PUT, DELETE 方法，並處理查詢參數、標頭和多種請求主體類型。
    """
    def __init__(self, timeout: int = 30, verify: bool = True, follow_redirects: bool = False):
        """
        初始化 APIRequestTool。

        參數:
            - timeout (int): 請求的超時時間（秒）。
            - verify (bool): 是否驗證 TLS 憑證。預設為 True，建議保持以確保安全。
            - follow_redirects (bool): 是否自動跟隨重新導向。

        返回:
            - None

        使用範例:
            >>> tool = APIRequestTool(timeout=10)
            >>> result = tool.execute_request(method="GET", url="https://httpbin.org/get")
            >>> result["status_code"] == 200
            True

        可能觸發的錯誤:
            - httpx.TimeoutException: 連線逾時
            - httpx.RequestError: 網路連線或 DNS 等請求層級錯誤
            - httpx.HTTPStatusError: 4xx/5xx 等非 2xx 狀態碼（在 raise_for_status 之後）
            - ValueError: 發送的 payload 參數互斥檢查未通過
        """
        self.client = httpx.Client(timeout=timeout, verify=verify, follow_redirects=follow_redirects)
        self._closed = False
        self._logger = logging.getLogger(__name__)

    def close(self) -> None:
        """
        關閉底層 HTTP 連線資源。

        參數:
            - None

        返回:
            - None

        使用範例:
            >>> tool = APIRequestTool()
            >>> try:
            ...     tool.execute_request(method="GET", url="https://example.com")
            ... finally:
            ...     tool.close()

        可能觸發的錯誤:
            - 無直接丟出錯誤；若重複關閉將被忽略。
        """
        if not self._closed:
            self.client.close()
            self._closed = True

    def __enter__(self):
        """
        進入 context manager。允許以 with 語法自動管理資源。

        參數:
            - None

        返回:
            - APIRequestTool: 物件本身

        使用範例:
            >>> with APIRequestTool() as tool:
            ...     tool.execute_request(method="GET", url="https://example.com")

        可能觸發的錯誤:
            - 無
        """
        return self

    def __exit__(self, exc_type, exc, tb):
        """
        離開 context manager 時自動釋放資源。

        參數:
            - exc_type: 例外型別
            - exc: 例外實例
            - tb: 追蹤資訊

        返回:
            - None

        使用範例:
            見 __enter__。

        可能觸發的錯誤:
            - 無
        """
        self.close()

    def _send_request(self,
                             method: Literal["GET", "POST", "PUT", "DELETE"],
                             url: str,
                             query_params: Optional[Dict[str, Any]] = None,
                             headers: Optional[Dict[str, str]] = None,
                             json_data: Optional[Dict[str, Any]] = None,
                             form_data: Optional[Dict[str, Any]] = None,
                             raw_text: Optional[str] = None) -> httpx.Response:
        """
        內部方法：根據提供的參數準備並發送 HTTP 請求。
        處理不同的 Body Payload 類型 (JSON, form-data, raw text)。

        參數:
            - method (Literal["GET","POST","PUT","DELETE"]): HTTP 方法。
            - url (str): 目標 URL。
            - query_params (Optional[Dict[str, Any]]): 查詢參數。
            - headers (Optional[Dict[str, str]]): 請求標頭。
            - json_data (Optional[Dict[str, Any]]): JSON 主體（不可與 form_data/raw_text 同時使用）。
            - form_data (Optional[Dict[str, Any]]): 表單主體（不可與 json_data/raw_text 同時使用）。
            - raw_text (Optional[str]): 純文字主體（不可與 json_data/form_data 同時使用）。

        返回:
            - httpx.Response: 原始回應物件。

        使用範例:
            >>> _ = APIRequestTool()._send_request(method="GET", url="https://httpbin.org/get")

        可能觸發的錯誤:
            - ValueError: 當 json_data、form_data、raw_text 同時提供超過一種。
            - httpx.RequestError/TimeoutException: 連線與逾時相關錯誤。
        """
        provided_payloads = [p is not None for p in (json_data, form_data, raw_text)]
        if sum(provided_payloads) > 1:
            raise ValueError("json_data, form_data, raw_text 至多提供一種")
        request_kwargs = {
            "method": method,
            "url": url,
            "params": query_params,
            "headers": headers,
            "json": None,       
            "data": None,       
            "content": None     
        }

        if json_data:
            request_kwargs["json"] = json_data
        elif form_data:
            request_kwargs["data"] = form_data
        elif raw_text:
            request_kwargs["content"] = raw_text

        response = self.client.request(**request_kwargs)
        return response

    def execute_request(self,
                              method: Literal["GET", "POST", "PUT", "DELETE"],
                              url: str,
                              query_params: Optional[Dict[str, Any]] = None,
                              headers: Optional[Dict[str, str]] = None,
                              json_data: Optional[Dict[str, Any]] = None,
                              form_data: Optional[Dict[str, Any]] = None,
                              raw_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Agent 調用的主要方法，用於發送 API 請求並處理回應。

        參數:
            - method (Literal["GET","POST","PUT","DELETE"]): HTTP 請求方法。
            - url (str): 請求的 URL。
            - query_params (Optional[Dict[str, Any]]): 查詢參數字典。
            - headers (Optional[Dict[str, str]]): 請求標頭字典。
            - json_data (Optional[Dict[str, Any]]): JSON 格式的請求主體（與其他 body 互斥）。
            - form_data (Optional[Dict[str, Any]]): Form-data 格式的請求主體（與其他 body 互斥）。
            - raw_text (Optional[str]): 原始文字格式的請求主體（與其他 body 互斥）。

        返回:
            - Dict[str, Any]:
              - status_code (Optional[int]): HTTP 狀態碼；逾時或請求層級錯誤為 None。
              - response_body (Optional[Dict[str, Any] | str]): 回應內容（JSON 或文字）。
              - error (Optional[str]): 錯誤訊息，成功時為 None。

        使用範例:
            >>> tool = APIRequestTool()
            >>> out = tool.execute_request(method="GET", url="https://httpbin.org/status/200")
            >>> out["status_code"]
            200

        可能觸發的錯誤:
            - httpx.TimeoutException/httpx.RequestError/httpx.HTTPStatusError 轉換為回傳字典中的 error。
            - ValueError: 當多種 body 同時提供時（由內部方法丟出並在此處捕捉）。
        """
        try:
            response = self._send_request(method, url, query_params, headers, json_data, form_data, raw_text)
            response.raise_for_status() # 對於 4xx/5xx 狀態碼會拋出 httpx.HTTPStatusError

            # 回應處理
            try:
                response_body = response.json()
            except (json.JSONDecodeError, ValueError): # 如果不是 JSON 格式或有其他 ValueError，則回傳原始文字
                response_body = response.text

            return {
                "status_code": response.status_code,
                "response_body": response_body,
                "error": None
            }

        except ValueError as e:
            return {
                "status_code": None,
                "response_body": None,
                "error": str(e)
            }
        except httpx.TimeoutException as e:
            return {
                "status_code": None,
                "response_body": None,
                "error": f"API request timed out: {e}"
            }
        except httpx.HTTPStatusError as e:
            return {
                "status_code": e.response.status_code,
                "response_body": e.response.text,
                "error": f"API request failed: {e.response.status_code} {e.response.reason_phrase}"
            }
        except httpx.RequestError as e:
            return {
                "status_code": None,
                "response_body": None,
                "error": f"An error occurred while requesting {e.request.url!r}: {e}"
            }
