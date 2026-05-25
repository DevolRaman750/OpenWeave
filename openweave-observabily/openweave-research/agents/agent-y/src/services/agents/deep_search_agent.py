import json
from typing import List, Dict, Annotated, Optional, Any, Tuple, Union

import sys
from src.utils.agent_gpt4 import AzureGPT4Chat
from src.utils.web_search_agent.tool_web_engine import SerperSearchEngine, WebBrowser
from streamlit_extras.colored_header import colored_header

import streamlit as st
from datetime import datetime
import asyncio
import autogen
import os
from autogen import Agent, AssistantAgent, UserProxyAgent, ConversableAgent
from textwrap import dedent
from src.utils.toolkits import register_toolkits
import time
from src.services.autogen_upgrade.base_agent import ExtendedAssistantAgent, ExtendedUserProxyAgent
from src.services.agents.agent_general_coder import GeneralCoder
from src.utils.tools_util import get_autogen_message_history

import traceback
import random
import tiktoken  # Add this import for calculating token count
from copy import deepcopy

from src.utils.tool_summary import generate_summary

from src.services.agents.agent_tool_library import AgentToolLibrary
from src.services.prompts.deepsearch_prompt import EXECUTOR_SYSTEM_PROMPT, DEEP_SEARCH_SYSTEM_PROMPT, DEEP_SEARCH_CONTEXT_SUMMARY_PROMPT, DEEP_SEARCH_RESULT_REPORT_PROMPT
from src.utils.error_artifacts import write_error_artifact, text_fingerprint

from src.utils.tool_optimizer_dialog import optimize_execution, optimize_dialogue

from configs.oai_config import get_llm_config


class DeepSearchExecutor(ExtendedUserProxyAgent):
    def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    
    async def a_execute_function(
        self, func_call: dict[str, Any], call_id: Optional[str] = None, verbose: bool = False
    ) -> tuple[bool, dict[str, Any]]:
        """Execute a function call and return the result.

        Override this function to modify the way to execute function and tool calls.

        Args:
            func_call: a dictionary extracted from openai message at "function_call" or "tool_calls" with keys "name" and "arguments".
            call_id: a string to identify the tool call.
            verbose (bool): Whether to send messages about the execution details to the
                output stream. When True, both the function call arguments and the execution
                result will be displayed. Defaults to False.


        Returns:
            A tuple of (is_exec_success, result_dict).
            is_exec_success (boolean): whether the execution is successful.
            result_dict: a dictionary with keys "name", "role", and "content". Value of "role" is "function".

        "function_call" deprecated as of [OpenAI API v1.1.0](https://github.com/openai/openai-python/releases/tag/v1.1.0)
        See https://platform.openai.com/docs/api-reference/chat/create#chat-create-function_call
        """
        code_summary_args = await self.a_merge_code_chat_history(func_call)
        if code_summary_args is not None and isinstance(code_summary_args, str):
            func_call["arguments"] = code_summary_args
        
        return await super().a_execute_function(func_call, call_id, verbose)
        
    
    async def a_merge_code_chat_history(self, func_call: dict[str, Any]):

        func_name = func_call.get("name", "")
        func = self._function_map.get(func_name, None)

        arguments = json.loads(func_call.get("arguments", "{}"))
        print(func_name)
            
        if arguments is not None:
            if isinstance(arguments, dict) and func_name == 'create_code_tool':
                chat_history = self._oai_messages
                clean_chat_history = []
                chat_history = chat_history[[key for key in chat_history.keys()][0]]
                
                for message in chat_history:
                    if message.get("tool_responses", None):
                        continue
                    clean_chat_history.append(message)
                clean_chat_history = json.dumps(clean_chat_history, ensure_ascii=False)
                summary_chat_history = generate_summary(clean_chat_history)
                if arguments.get("chat_history", None) is not None:
                    arguments["chat_history"] = summary_chat_history
                else:
                    arguments["chat_history"] = summary_chat_history
                return json.dumps(arguments, ensure_ascii=False)
                
        return None

def get_researcher_system_message():
    return DEEP_SEARCH_SYSTEM_PROMPT.format(current_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")) #+ thinking_prompt


# New Autogen deep search implementation
class AutogenDeepSearchAgent:
    def __init__(self, llm_config=None, code_execution_config=None, return_chat_history=False, save_log=False):
        self.web_browser = WebBrowser()
        self.llm_config = get_llm_config(service_type="deepsearch") if llm_config is None else llm_config
        self.code_execution_config={"work_dir": 'coding', "use_docker": False} if code_execution_config is None else code_execution_config

        # Ensure runtime timeout from launcher/execution config is reflected in LLM calls.
        if isinstance(self.llm_config, dict) and isinstance(self.code_execution_config, dict):
            runtime_timeout = self.code_execution_config.get("timeout")
            if isinstance(runtime_timeout, (int, float)) and runtime_timeout > 0:
                self.llm_config["timeout"] = int(runtime_timeout)

        self.max_turns = int(os.getenv("DEEPSEARCH_MAX_TURNS", self.code_execution_config.get("max_turns", 30)))
        self.deepsearch_timeout = int(os.getenv("DEEPSEARCH_TIMEOUT", self.code_execution_config.get("deepsearch_timeout", 900)))

        # Disable hidden SDK retries unless user explicitly overrides via environment.
        if isinstance(self.llm_config, dict) and "max_retries" not in self.llm_config:
            env_max_retries = os.getenv("OPENAI_CLIENT_MAX_RETRIES")
            if env_max_retries is not None:
                try:
                    self.llm_config["max_retries"] = int(env_max_retries)
                except (TypeError, ValueError):
                    self.llm_config["max_retries"] = 0
            else:
                self.llm_config["max_retries"] = 0
        
        self.return_chat_history = return_chat_history
        self.save_log = save_log
        
        # Add message history summary related parameters
        self.max_tool_messages_before_summary = 2  # How many rounds of tool calls before summarizing
        self.current_tool_call_count = 0
        self.token_limit = 2000  # Set token count limit
        self.encoding = tiktoken.get_encoding("cl100k_base")  # Use OpenAI's encoder
        
        # Create researcher agent - responsible for thinking and analysis
        self.researcher = ExtendedAssistantAgent(
            name="researcher",
            system_message=get_researcher_system_message(),
            llm_config=self.llm_config,
            # is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").replace("**", "").endswith("TERMINATE") or '<TERMINATE>' in x.get("content", ""),
            is_termination_msg=lambda x: (x.get("content", "") and len(x.get("content", "").split("TERMINATE")[-1])<5) or (x.get("content", "") and '<TERMINATE>' in x.get("content", "")),
        )
        
        # Create executor agent - responsible for executing search and browse operations
        self.executor = DeepSearchExecutor(
            name="executor",
            system_message=EXECUTOR_SYSTEM_PROMPT,
            human_input_mode="NEVER",
            llm_config=self.llm_config,
            code_execution_config=self.code_execution_config,
            # is_termination_msg=lambda x: x.get("content", "") and len(x.get("content", "").split("TERMINATE")[-1])<5 or '<TERMINATE>' in x.get("content", ""),
            is_termination_msg=lambda x: (x.get("content", "") and len(x.get("content", "").split("TERMINATE")[-1])<5) or (x.get("content", "") and '<TERMINATE>' in x.get("content", "")),
        )
        
        self.agent_tool_library = AgentToolLibrary(
            llm_config=self.llm_config,
            code_execution_config=self.code_execution_config,
            tool_list={
                "agent_coder": False,
                "deep_search": True,
            },
            chat_history_provider=self._get_researcher_chat_history
        )
        
        # Register tool functions
        self._register_tools()
        
        # Modify agent message handling methods to support dynamic summarization
        self._patch_agent_message_handlers()
        
    def _register_tools(self):
        """Register tool functions to executor agent"""
        register_toolkits(
            [
                self.agent_tool_library.searching,
                self.agent_tool_library.browsing,
                # self.agent_tool_library.create_code_tool,
            ],
            self.researcher,
            self.executor,
        )
    
    def _patch_agent_message_handlers(self):
        """Patch agent message handling methods to support dynamic summarization"""
        # Save original methods
        original_executor_receive = self.executor._process_received_message
        original_researcher_receive = self.researcher._process_received_message
        
        # Add message handling interception for executor
        def executor_receive_with_summary(message, sender, silent):
            # Check if it's a function call from the researcher
            message_history = deepcopy(self.executor.chat_messages[self.researcher])
            if sender == self.researcher and len(message_history)>1:
                if 'tool_responses' in message_history[-1] and 'tool_calls' in message_history[-2]:
                    # Increase tool call count
                    self._summarize_tool_response(message_history, message)
                    self.current_tool_call_count += 1
            
            # Process message normally
            return original_executor_receive(message, sender, silent)
        
        # Add message handling interception for researcher
        def researcher_receive_with_summary(message, sender, silent):
            # Check if it's a tool response from the executor
            if sender == self.executor and self.current_tool_call_count >= self.max_tool_messages_before_summary:
                # Execute message history summarization
                # Reset counter
                self.current_tool_call_count = 0
            
            # Process message normally
            return original_researcher_receive(message, sender, silent)
        
        # Replace original methods
        self.executor._process_received_message = executor_receive_with_summary
        self.researcher._process_received_message = researcher_receive_with_summary
    
    def _summarize_tool_response(self, chat_history, current_message):
        """Summarize message history"""
        # Get current conversation history
        
        tool_calls = chat_history[-2]['tool_calls']
        tool_responses_list = chat_history[-1]['tool_responses']
        
        del self.executor.chat_messages[self.researcher][-1]['content']
        del self.researcher.chat_messages[self.executor][-2]['content']

        if not isinstance(tool_responses_list, list):
            tool_responses_list = [tool_responses_list]
        
        summary_list = []
            
        for tool_responses in tool_responses_list:
        
            if isinstance(tool_responses, list) or isinstance(tool_responses, dict):
                tool_responses = json.dumps(tool_responses)
            elif not isinstance(tool_responses, str):
                tool_responses = str(tool_responses)
            
            if isinstance(tool_calls, list) or isinstance(tool_calls, dict):
                tool_calls = json.dumps(tool_calls)
            elif not isinstance(tool_calls, str):
                tool_calls = str(tool_calls)
            
            # Calculate token count instead of character count
            token_count = len(self.encoding.encode(tool_responses))
            if token_count < self.token_limit:
                continue
            
            # chat_history.append(current_message)
            chat_history = json.dumps(chat_history[:-2], ensure_ascii=False)
            
            # Generate summary
            response_summary = self._generate_summary_for_search_result(chat_history, tool_responses)
            # print(response_summary)
            summary_list.append(response_summary)
        
        try:
            for idx, sumary in enumerate(summary_list):
                self.executor.chat_messages[self.researcher][-1]['tool_responses'][idx]['content'] = sumary
                self.researcher.chat_messages[self.executor][-2]['tool_responses'][idx]['content'] = sumary
        except Exception as e:
            print(e)
            import pdb;pdb.set_trace()
            

    def _generate_summary_for_search_result(self, messages, tool_responses):
        """Generate summary for a set of messages"""
        
        # Use LLM to generate summary
        summary_prompt = DEEP_SEARCH_CONTEXT_SUMMARY_PROMPT.format(tool_responses=tool_responses, messages=messages)
        
        # Use researcher's LLM config to create a temporary client for summary generation
        from autogen.oai import OpenAIWrapper
        client = OpenAIWrapper(**self.llm_config)
        
        # Create message list
        messages_list = [{"role": "user", "content": summary_prompt}]
        
        # Directly use client's create method without passing additional API parameters
        response = client.create(messages=messages_list)
        self._print_llm_usage(response, label="tool-summary")
            
        summary = response.choices[0].message.content
        
        return summary

    def _print_llm_usage(self, response, label: str = "deepsearch") -> None:
        usage = getattr(response, "usage", None)
        choices = getattr(response, "choices", []) or []
        finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
        prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
        completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
        total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None
        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")

        max_tokens = self.llm_config.get("max_tokens") if isinstance(self.llm_config, dict) else None
        print(
            f"[llm-usage] label={label} prompt={prompt_tokens} "
            f"completion={completion_tokens} total={total_tokens} "
            f"max_tokens={max_tokens} finish_reason={finish_reason}"
        )

    def _is_retryable_llm_error(self, error: Exception) -> bool:
        if isinstance(error, TimeoutError):
            return True

        error_type_name = type(error).__name__.lower()
        retryable_error_types = (
            "apiconnectionerror",
            "apitimeouterror",
            "remoteprotocolerror",
            "readtimeout",
            "connecttimeout",
        )
        if any(name in error_type_name for name in retryable_error_types):
            return True

        text = str(error).lower()
        retryable_markers = (
            "timed out",
            "timeout",
            "bad gateway",
            "502",
            "429",
            "rate limit",
            "temporarily unavailable",
            "service unavailable",
            "connection reset",
            "connection error",
            "server disconnected",
            "disconnected without sending a response",
            "remote protocol",
        )
        return any(marker in text for marker in retryable_markers)

    def _set_llm_timeout(self, timeout_seconds: int):
        if not isinstance(self.llm_config, dict):
            return
        self.llm_config["timeout"] = timeout_seconds
        if hasattr(self.researcher, "llm_config") and isinstance(self.researcher.llm_config, dict):
            self.researcher.llm_config["timeout"] = timeout_seconds
        if hasattr(self.executor, "llm_config") and isinstance(self.executor.llm_config, dict):
            self.executor.llm_config["timeout"] = timeout_seconds

    def _compute_retry_delay(self, attempt_index: int) -> float:
        # Exponential backoff with small jitter to avoid synchronized retries.
        base_delay = min(2 ** attempt_index, 10)
        jitter = random.uniform(0.2, 0.8)
        return float(base_delay + jitter)

    async def _run_chat_with_heartbeat(
        self,
        chat_coro,
        attempt: int,
        max_attempts: int,
        attempt_timeout: int,
        request_timeout: int,
    ):
        """Run chat coroutine while printing periodic heartbeat logs during long waits."""
        heartbeat_seconds = int(os.getenv("DEEPSEARCH_HEARTBEAT_SECONDS", "15"))
        heartbeat_seconds = max(5, heartbeat_seconds)

        chat_task = asyncio.create_task(chat_coro)
        started_at = time.monotonic()

        while True:
            done, _ = await asyncio.wait({chat_task}, timeout=heartbeat_seconds)
            if done:
                return await chat_task

            elapsed = int(time.monotonic() - started_at)
            if elapsed >= attempt_timeout:
                chat_task.cancel()
                try:
                    await chat_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
                raise TimeoutError(
                    f"Deep search attempt exceeded wall-clock timeout ({attempt_timeout}s)."
                )

            model = None
            base_url = None
            if isinstance(self.llm_config, dict):
                config_list = self.llm_config.get("config_list", [])
                if isinstance(config_list, list) and config_list and isinstance(config_list[0], dict):
                    model = config_list[0].get("model")
                    base_url = config_list[0].get("base_url")

            print(
                f"[deepsearch] waiting for model reply "
                f"attempt {attempt}/{max_attempts}, elapsed={elapsed}s, "
                f"request_timeout={request_timeout}s, deepsearch_timeout={attempt_timeout}s, "
                f"model={model}, base_url={base_url}"
            )
    
    async def deep_search(self, query: str, summary_prompt: str | None = None) -> str:
        """
        Execute deep search and return results
        
        Args:
            query: User's query question
            
        Returns:
            Search results and answer
        """
        # Reset tool call count
        self.current_tool_call_count = 0
        
        self.original_query = query
        
        result_summary_prompt = summary_prompt or DEEP_SEARCH_RESULT_REPORT_PROMPT

        initial_message = dedent(f"""
        I need you to help me research the following question in depth:
        
        {query}
        """)
        
        self.agent_tool_library.update_chat_history({"original_query": self.original_query})
        self.researcher.update_system_message(get_researcher_system_message())
        
        max_attempts = int(os.getenv("DEEPSEARCH_RETRY_ATTEMPTS", "1"))
        max_attempts = max(1, max_attempts)
        llm_timeout = 240
        if isinstance(self.llm_config, dict):
            timeout_from_cfg = self.llm_config.get("timeout")
            if isinstance(timeout_from_cfg, (int, float)) and timeout_from_cfg > 0:
                llm_timeout = int(timeout_from_cfg)

        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            attempt_timeout = self.deepsearch_timeout
            self._set_llm_timeout(int(llm_timeout))
            print(
                f"[deepsearch] starting attempt {attempt}/{max_attempts} "
                f"with request_timeout={llm_timeout}s, deepsearch_timeout={attempt_timeout}s, "
                f"max_turns={self.max_turns}"
            )
            try:
                chat_result = await self._run_chat_with_heartbeat(
                    self.executor.a_initiate_chat(
                        self.researcher,
                        message=initial_message,
                        max_turns=self.max_turns,
                        summary_method="reflection_with_llm", # Supported strings are "last_msg" and "reflection_with_llm":
                        summary_args={
                            'summary_prompt': result_summary_prompt
                        }
                    ),
                    attempt=attempt,
                    max_attempts=max_attempts,
                    attempt_timeout=int(attempt_timeout),
                    request_timeout=int(llm_timeout),
                )
                break
            except Exception as e:
                last_error = e
                is_retryable = self._is_retryable_llm_error(e)
                if not is_retryable or attempt >= max_attempts:
                    break

                delay_seconds = self._compute_retry_delay(attempt)
                print(
                    f"[deepsearch] transient error on attempt {attempt}/{max_attempts} "
                    f"(timeout={attempt_timeout}s): {e}. Retrying in {delay_seconds:.2f}s"
                )
                await asyncio.sleep(delay_seconds)

        if last_error is not None and 'chat_result' not in locals():
            e = last_error
            llm_first = {}
            if isinstance(self.llm_config, dict):
                config_list = self.llm_config.get("config_list", [])
                if isinstance(config_list, list) and config_list:
                    llm_first = config_list[0] if isinstance(config_list[0], dict) else {}

            error_context = {
                "query": text_fingerprint(query),
                "max_turns": self.max_turns,
                "retry": {
                    "max_attempts": max_attempts,
                    "final_timeout": self.llm_config.get("timeout") if isinstance(self.llm_config, dict) else None,
                    "deepsearch_timeout": self.deepsearch_timeout,
                    "retryable": self._is_retryable_llm_error(e),
                },
                "tool_call_count": self.current_tool_call_count,
                "token_limit": self.token_limit,
                "return_chat_history": self.return_chat_history,
                "llm": {
                    "model": llm_first.get("model"),
                    "base_url": llm_first.get("base_url"),
                    "api_type": llm_first.get("api_type"),
                    "timeout": self.llm_config.get("timeout") if isinstance(self.llm_config, dict) else None,
                    "temperature": self.llm_config.get("temperature") if isinstance(self.llm_config, dict) else None,
                    "top_p": self.llm_config.get("top_p") if isinstance(self.llm_config, dict) else None,
                    "max_tokens": self.llm_config.get("max_tokens") if isinstance(self.llm_config, dict) else None,
                },
            }
            artifact_path = write_error_artifact(
                component="deepsearch",
                operation="deep_search",
                error=e,
                context=error_context,
            )
            setattr(e, "_repomaster_artifact_path", artifact_path)
            if artifact_path:
                print(f"[deepsearch] Error artifact: {artifact_path}")
            raise e

        final_answer = self._extract_final_answer(chat_result)
        if self.return_chat_history:
            return final_answer, get_autogen_message_history(chat_result.chat_history)
        return final_answer
    
    def _extract_final_answer(self, chat_result) -> str:
        """Extract final answer from chat history"""
        # Extract final result

        final_answer = chat_result.summary
        
        if isinstance(final_answer, dict):
            final_answer = final_answer['content']
        
        if final_answer is None:
            final_answer = ""
        final_answer = final_answer.strip().lstrip()
        
        messages = chat_result.chat_history
        final_content = messages[-1].get("content", "")
        if final_content:
            final_content = final_content.strip().lstrip()
        
        if final_answer == "":
            final_answer = final_content
        
        return final_answer

    def get_partial_research_text(self) -> str:
        """Return best-effort text from the current deepsearch conversation."""
        chunks: list[str] = []
        seen_ids: set[int] = set()

        for agent in (self.researcher, self.executor):
            chat_messages = getattr(agent, "chat_messages", {})
            if not isinstance(chat_messages, dict):
                continue

            for messages in chat_messages.values():
                if not isinstance(messages, list):
                    continue
                for message in messages:
                    if not isinstance(message, dict) or id(message) in seen_ids:
                        continue
                    seen_ids.add(id(message))

                    content = message.get("content")
                    if isinstance(content, str) and content.strip():
                        chunks.append(content.strip())

                    for key in ("tool_calls", "tool_responses"):
                        value = message.get(key)
                        if value:
                            try:
                                chunks.append(json.dumps(value, ensure_ascii=False))
                            except Exception:
                                chunks.append(str(value))

        return "\n\n".join(chunks)[-20000:]
    
    
    def web_agent_answer(self, query: Annotated[str, "The initial search query"]) -> str:
        """
        Execute deep search and return results
        
        Args:
            query: User's query question
            
        Returns:
            JSON string of search results
        """
        return asyncio.run(self.deep_search(query))
    
    async def run(self, query: str) -> str:
        """
        Execute deep search and return results (async version)
        """
        self.return_chat_history = True
        final_answer, chat_result = await self.deep_search(query)
        return {
            "final_answer": final_answer,
            "trajectory": chat_result
        }
    
    async def a_web_agent_answer(self, query: Annotated[str, "The initial search query"]) -> str:
        """
        Execute deep search and return results (async version)
        
        Args:
            query: User's query question
            
        Returns:
            JSON string of search results
        """
        try:
            return await self.deep_search(query)
        except Exception as e:
            artifact_path = getattr(e, "_repomaster_artifact_path", "")
            if not artifact_path:
                artifact_path = write_error_artifact(
                    component="deepsearch",
                    operation="a_web_agent_answer",
                    error=e,
                    context={
                        "query": text_fingerprint(query),
                        "return_chat_history": self.return_chat_history,
                    },
                )
                setattr(e, "_repomaster_artifact_path", artifact_path)

            error_msg = f"Error occurred during deep search: {str(e)}\n"
            error_msg += "Detailed error information:\n"
            error_msg += traceback.format_exc()
            if artifact_path:
                error_msg += f"\nError artifact: {artifact_path}"
            print(error_msg)
            return f"Error occurred during search:\n{error_msg}"

    def _get_researcher_chat_history(self) -> dict:
        """
        Get researcher's current chat_history for passing to code_tool
        
        This method will:
        1. Get the latest conversation history between researcher and executor
        2. Filter out tool response messages, keeping only useful conversation content
        3. Limit message length and count to avoid passing too much information
        4. Add context information like current time and original query
        
        Returns:
            dict: Contains processed chat_history and related context information
        """
        try:
            result = {
                "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Add original query information
            if hasattr(self, 'original_query'):
                result["original_query"] = self.original_query
            
            # Get conversation history between researcher and executor
            if hasattr(self.researcher, 'chat_messages') and self.executor in self.researcher.chat_messages:
                chat_messages = self.researcher.chat_messages[self.executor]
                
                chat_messages = json.dumps(chat_messages, ensure_ascii=False)
                # chat_messages = optimize_execution(chat_messages)
                chat_messages = optimize_dialogue(chat_messages)
                
                result["chat_history"] = chat_messages
            
            return result

        except Exception as e:
            print(f"Error getting researcher chat history: {e}")
            import traceback
            traceback.print_exc()
            return {
                "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "error": f"Failed to get chat history: {str(e)}"
            }

if __name__ == "__main__":
    from dotenv import load_dotenv
    
    load_dotenv("/mnt/ceph/huacan/Code/Tasks/envs/.env")
    
    # Use new AutogenDeepSearchAgent
    deep_search_agent = AutogenDeepSearchAgent()
    # Set test query
    # query = "What is the process architecture of autogen's groupchat, and what is its underlying system prompt"
    # query = "Cheapest flight price from Beijing to Yuyao"
    # query = "Query today's subscription code for the new stock Guotaijunan on Shanghai Stock Exchange STAR Market"
    query = "Which companies have the top 30 gross profit margins in the semiconductor industry in 2024, and rank them"
    # query = "Free & free trial accounts can no longer use chat with premium models on Cursor Version 0.45 or less. Please upgrade to Pro or use Cursor Version 0.46 or later. Install Cursor at https://www.cursor.com/downloads or update from within the editor. How to solve this problem, can cursor continue to be used for free?"
    answer = deep_search_agent.web_agent_answer(query)
    print(f"Answer: {answer}")  
