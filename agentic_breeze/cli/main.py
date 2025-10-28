#!/usr/bin/env python3
"""
Agentic Breeze CLI - Command Line Interface for Agentic Breeze Agent Framework
"""
import argparse
import sys
import os

from agentic_breeze.agents.orchestrator import Orchestrator
from agentic_breeze.llm.llm_client import LLMConnector
from agentic_breeze.prompts.prompt_manager import PromptManager
from agentic_breeze.agents.orchestrator_core.planning_manager import PlanningManager
from agentic_breeze.agents.orchestrator_core.tool_executor import ToolExecutor
from agentic_breeze.agents.orchestrator_core.query_rewriter import QueryRewriter
from agentic_breeze.agents.orchestrator_core.conversation_manager import ConversationManager
from agentic_breeze.agents.orchestrator_core.synthesis_generator import SynthesisGenerator
from agentic_breeze.registry.tool_registry import ToolRegistry


def create_orchestrator():
    """建立並配置 Agentic Breeze orchestrator。
    
    根據環境變數初始化所有必要的組件並組裝成 Orchestrator 實例。
    
    Args:
        無參數，但會讀取環境變數：
        - HOST_TYPE: 主機類型，預設 'ollama'
        - TIMEOUT: 逾時秒數，預設 300
        - MAX_TOKENS: 最大 token 數，預設 1000
        - TEMPERATURE: 溫度參數，預設 0.5
    
    Returns:
        Orchestrator: 配置完成的 Orchestrator 實例
    
    Examples:
        >>> orchestrator = create_orchestrator()
        >>> isinstance(orchestrator, Orchestrator)
        True
    
    Raises:
        ImportError: 當無法導入所需模組時
        ValueError: 當環境變數值無效時
    """
    llm_connector = LLMConnector(
        host_type=os.getenv("HOST_TYPE", "ollama"),
        timeout=int(os.getenv("TIMEOUT", "300")),
        max_tokens=int(os.getenv("MAX_TOKENS", "1000")),
        temperature=float(os.getenv("TEMPERATURE", "0.5"))
    )
    prompt_manager = PromptManager()
    planning_manager = PlanningManager(
        llm_client=llm_connector,
        tool_registry=ToolRegistry(),
        prompt_manager=prompt_manager
    )
    tool_executor = ToolExecutor(
        tool_registry=ToolRegistry(),
    )
    query_rewriter = QueryRewriter(
        llm_client=llm_connector,
        prompt_manager=prompt_manager
    )
    conversation_manager = ConversationManager(
        llm_client=llm_connector
    )
    synthesis_generator = SynthesisGenerator(
        llm_client=llm_connector,
        prompt_manager=prompt_manager
    )

    return Orchestrator(
        prompt_manager=prompt_manager,
        planning_manager=planning_manager,
        tool_executor=tool_executor,
        conversation_manager=conversation_manager,
        synthesis_generator=synthesis_generator,
        query_rewriter=query_rewriter
    )


def run_web_interface():
    """啟動 Gradio 網頁介面。
    
    建立並啟動一個交互式的網頁聊天介面，支援串流回應。
    
    Args:
        無參數
    
    Returns:
        None: 會持續運行直到使用者關閉
    
    Examples:
        >>> # 在終端執行
        >>> run_web_interface()
        === 啟動 Gradio 介面 ===
        Running on local URL: http://127.0.0.1:7860
    
    Raises:
        ImportError: 當 gradio 套件未安裝時
        RuntimeError: 當網頁伺服器啟動失敗時
    """
    try:
        import gradio as gr
        from dotenv import load_dotenv
        load_dotenv()
        
        orchestrator = create_orchestrator()
        
        def chat_interface(user_message, history):
            if not history:
                history = []
            
            # Gradio history format is List[List[str, str]] where inner list is [user_message, bot_message]
            # Our orchestrator expects List[Dict[str, str]]
            formatted_history = []
            for human, assistant in history:
                formatted_history.append({"role": "user", "content": human})
                formatted_history.append({"role": "assistant", "content": assistant})

            # Try streaming first, fallback to non-streaming
            try:
                full_reply = ""
                for chunk in orchestrator.aquery_with_history_stream(user_message, formatted_history):
                    if chunk:
                        full_reply += chunk
                        # Yield partial response for streaming effect
                        yield full_reply
                
                # Final yield with complete response
                yield full_reply
                
            except Exception as e:
                # Fallback to non-streaming
                print(f"Web streaming failed, using non-streaming mode: {e}")
                reply = orchestrator.aquery_with_history(user_message, formatted_history)
                yield reply

        print("=== 啟動 Gradio 介面 ===")
        
        gr.ChatInterface(
            chat_interface,
            title="Agentic Breeze 智慧助理",
            description="與您的 Agentic Breeze 助理對話。輸入 'exit' 或 'quit' 結束對話。",
            examples=[
                ["推薦台灣珍珠奶茶手搖店三間"],
                ["台北夜市知名小吃"],
                ["規劃宜蘭週末旅行"],
            ]
        ).launch()
    except ImportError:
        print("Error: gradio is required for web interface. Install with: pip install gradio")
        sys.exit(1)


def run_chat():
    """執行交互式聊天模式。
    
    在終端中啟動交互式聊天介面，支援多輪對話與串流回應。
    
    Args:
        無參數
    
    Returns:
        None: 會持續運行直到使用者輸入 'exit', 'quit' 或 Ctrl+C
    
    Examples:
        >>> # 在終端執行
        >>> run_chat()
        === Agentic Breeze 智慧助理 CLI ===
        輸入 'exit' 或 'quit' 結束對話
        ----------------------------------------
        
        你: 今天天氣如何？
        Agentic Breeze: 今天天氣晴朗，氣溫25度。
    
    Raises:
        ImportError: 當缺少所需依賴套件時
        KeyboardInterrupt: 當使用者按下 Ctrl+C 時（會被捕捉並正常退出）
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        orchestrator = create_orchestrator()
        
        print("=== Agentic Breeze 智慧助理 CLI ===")
        print("輸入 'exit' 或 'quit' 結束對話")
        print("-" * 40)
        
        history = []
        
        while True:
            try:
                user_input = input("\n你: ").strip()
                
                if user_input.lower() in ['exit', 'quit', '退出']:
                    print("\n再見！")
                    break
                
                if not user_input:
                    continue
                
                # Get streaming response from orchestrator
                print(f"\nAgentic Breeze: ", end="", flush=True)
                full_reply = ""
                
                try:
                    for chunk in orchestrator.aquery_with_history_stream(user_input, history):
                        if chunk:
                            print(chunk, end="", flush=True)
                            full_reply += chunk
                    print()  # New line after streaming complete
                except Exception as e:
                    # Fallback to non-streaming if streaming fails
                    print(f"串流模式失敗，切換至一般模式: {e}")
                    full_reply = orchestrator.aquery_with_history(user_input, history)
                    print(full_reply)
                
                # Update history
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": full_reply})
                
            except KeyboardInterrupt:
                print("\n\n再見！")
                break
            except Exception as e:
                print(f"\n錯誤: {e}")
                
    except ImportError as e:
        print(f"Error: Missing required dependencies: {e}")
        print("Please install required packages with: pip install -r requirements.txt")
        sys.exit(1)


def main():
    """主要的 CLI 入口點。
    
    解析命令列參數並執行對應的功能：
    - web: 啟動網頁介面
    - chat: 啟動終端聊天模式
    - 無參數: 顯示幫助訊息
    
    Args:
        無參數，但會讀取 sys.argv
    
    Returns:
        None
    
    Examples:
        >>> # 在終端執行
        >>> # python -m agentic_breeze.cli.main chat
        >>> # python -m agentic_breeze.cli.main web
        >>> # python -m agentic_breeze.cli.main --version
    
    Raises:
        SystemExit: 當命令列參數錯誤或執行失敗時
    """
    parser = argparse.ArgumentParser(
        description="Agentic Breeze - 智慧助理框架",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="Agentic Breeze 1.0.0"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Web interface command
    web_parser = subparsers.add_parser("web", help="Launch web interface")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive chat")
    
    args = parser.parse_args()
    
    if args.command == "web":
        run_web_interface()
    elif args.command == "chat":
        run_chat()
    else:
        # Default behavior - show help
        parser.print_help()


if __name__ == "__main__":
    main()