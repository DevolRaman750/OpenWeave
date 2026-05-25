import json
import argparse
import asyncio
import os
import math
import re
import time
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Annotated, Optional, Any
from pathlib import Path
from textwrap import dedent
from dotenv import load_dotenv
import aiohttp
import autogen
from autogen.cache import Cache

from src.services.autogen_upgrade.base_agent import ExtendedUserProxyAgent, ExtendedAssistantAgent, check_code_block
from src.utils.toolkits import register_toolkits

from src.services.agents.deep_search_agent import AutogenDeepSearchAgent
from src.core.acm_intake import load_stage1_acm_intake, ACMValidationError, Stage1IntakeResult
from src.utils.error_artifacts import write_error_artifact, text_fingerprint
from src.core.candidate_filter import filter_candidates
from src.core.category_map import get_topic_signals

from src.core.git_task import TaskManager, AgentRunner


GITHUB_REPO_SEARCH_SUMMARY_PROMPT = """
Return ONLY valid JSON for the user's GitHub repository search task.

Select the best 5 repositories that are most likely to contain the algorithm,
approach, tools, or implementation logic needed to solve the problem described
in input_acm.json and the repomaster query.

Rules:
- Return exactly a JSON array.
- Return at most 5 objects.
- Do not include markdown, prose, comments, headings, citations, or TERMINATE.
- Every object must include repo_name, repo_url, and repo_description.
- repo_url must be a canonical GitHub repository URL: https://github.com/owner/repo
- Rank best first.

Format:
[
  {
    "repo_name": "owner/repo",
    "repo_url": "https://github.com/owner/repo",
    "repo_description": "why this repo is relevant"
  }
]
"""


scheduler_system_message = dedent("""Role: Enhanced Task Scheduler

Primary Responsibility:
The Enhanced Task Scheduler's main duty is to analyze user input, create a structured task plan based on available tools, and then select and call appropriate tools from the given set to fulfill user requirements. The Enhanced Task Scheduler can work in four distinct modes to fulfill user requirements:
1. **Web Search Mode**: Search the internet for real-time information, current events, or general knowledge.
2. **Repository Mode**: Search and use GitHub repositories or local repositories to solve tasks through hierarchical repository analysis and autonomous exploration. This is the primary approach for complex coding tasks that require repository-level solutions.
3. **General Code Assistant Mode**: Provide general programming assistance without specific repositories.

Mode Selection Strategy:
- **Prioritize Web Search**: Before selecting a primary mode, first determine if the user's query can be directly answered or solved using the `web_search` tool. This is ideal for questions requiring real-time data, definitions, or general knowledge.
- **Repository Mode**: Use `run_repository_agent` for tasks involving code repositories. This unified tool automatically detects whether the repository is a GitHub URL or local path. Triggered when:
  * User mentions local file paths, specific local directories, or phrases like "analyze this local repo"
  * User wants to find existing solutions, mentions GitHub, or needs specialized tools/libraries
  * User provides specific repository URLs or paths
- **General Code Assistant Mode**: Used for general programming questions, requests for code examples, debugging help, or when no specific repository is mentioned.

Working Process:
1.  **Task Analysis and Initial Assessment**:
    *   Upon receiving user input, thoroughly analyze the requirements.
    *   **Step 1 - Web Search Assessment**: Determine if the task requires real-time data, latest information, current events, or external knowledge that would benefit from web search. If yes, execute web search first.
    *   **Step 2 - Plan Creation**: Create a structured plan. For tasks potentially solvable by Repository Mode, this plan must prioritize a two-step approach:
        1.  Search for relevant GitHub repositories using available tools.
        2.  If a suitable repository is identified, plan to use the unified repository tool to execute the task using that repository.

2.  **Tool Selection and Execution Based on Mode**:
    *   Execute the plan by selecting one appropriate tool at a time.
    *   **Repository Mode (GitHub/Local)**: Use the `run_repository_agent` tool with the repository URL or local path. The tool automatically detects whether it's a GitHub repository or local repository.
    *   **General Code Assistant Mode**: Use the `run_general_code_assistant` tool for programming guidance and solutions.
    *   **Web Search**: Use the `web_search` tool for any queries that require real-time information, external documentation, or general knowledge. This tool can be used on its own for non-coding questions or as a part of solving a larger coding task.
    *   **Repository Mode (Repository-First Approach)**:
        1.  **Repository Search**: First, use the `github_repo_search` tool to find a list of the most relevant GitHub repositories for the task.
        2.  **Sequential Execution**: Select the most promising repository from the search results and execute the task using the `run_repository_agent` tool.
        3.  **Result Evaluation and Switching**:
            *   After execution, critically evaluate if the result satisfies the user's requirements. Consider: (1) Was code successfully executed? (2) Does the output directly address the task? (3) Does the result contain the requested information?
            *   If the current repository failed to produce a satisfactory result, select the *next best* repository from the search results and try again with `run_repository_agent`.
            *   Continue this process of executing and evaluating until a repository successfully completes the task, or all viable repository options have been exhausted.

3.  **Sequential Execution and Fallback**:
    *   Subsequent tools are selected based on: (1) The outcomes of previously executed tools (2) The current state of the plan (3) The capabilities of the available tools.
    *   Execute the selected tool(s) according to the plan.
    *   If one approach (e.g., Repository mode) doesn't yield a solution, consider trying an alternative mode if appropriate (e.g., switching to General Code Assistant Mode).
    *   Be persistent in finding a solution. If one repository doesn't work, try another.

Important Notes:
1.  Always validate that local paths exist before using the repository mode with local paths.
2.  In general code assistant mode, aim to create practical, executable examples.
3.  If a tool is successfully called, answer based on the tool's results and your overall task plan.
4.  If no available tool can solve the task, generate an answer based on your own knowledge.
5.  When the task is completed successfully, reply ONLY with "TERMINATE".

Turn-taking and De-duplication Policy:
- Before sending any message, compare it with the previous two messages. If your response would substantially repeat previously stated content (facts, sentences, or structure), do not restate it. If the task has already been fully answered, reply with exactly "TERMINATE" instead.
""")

user_proxy_system_message = dedent("""Role: Execution Proxy (User Proxy)

Primary Rules:
- Do not provide user-facing answers.
- Never paraphrase or repeat content already provided by scheduler_agent.
- Summarize tool outputs succinctly for the scheduler_agent only when needed; avoid restating conclusions.
- If your candidate message would substantially repeat the last two messages, send exactly "TERMINATE" instead.
- After scheduler_agent has delivered the final complete answer, respond with exactly "TERMINATE" and stop.
""")


@dataclass
class RepoScore:
    repo: str
    final_score: float
    star_score: float
    readme_score: float
    commit_score: float
    issue_score: float
    topic_match_score: float
    reason: str


WEIGHTS = {
    "star": 0.35,        # was 0.40, slightly reduced
    "readme": 0.28,      # was 0.30, slightly reduced
    "commit": 0.17,      # was 0.20, slightly reduced
    "issue": 0.08,       # was 0.10, slightly reduced
    "topic_match": 0.12, # NEW - pulled weight from all four
}

class RepoMasterAgent:
    """
    RepoMaster agent for searching and utilizing GitHub repositories to solve user tasks.
    
    This agent can search for relevant GitHub repositories based on user tasks, analyze repository content,
    and generate solutions. The main work is accomplished through collaboration between scheduler agent and user agent.
    """
    # def __init__(self, local_repo_path: str, work_dir: str, remote_repo_path=None, llm_config=None, code_execution_config=None, task_type=None, use_venv=False, task_id=None, is_cleanup_venv=True, args={}):
    def __init__(self, llm_config=None, code_execution_config=None):
    
        self.llm_config = llm_config
        self.code_execution_config = code_execution_config
        
        self.repo_searcher = AutogenDeepSearchAgent(
            llm_config=self.llm_config,
            code_execution_config=self.code_execution_config,
        )

        self._acm_stage1 = self._load_local_acm_context()
        self._acm_context = self._acm_stage1.canonical_manifest
        
        self.work_dir = code_execution_config['work_dir']
        self._repo_topics_cache: dict[str, list[str]] = {}
        
        self.initialize_agents()
        self.register_tools()

    def _load_local_acm_context(self) -> Stage1IntakeResult:
        """Load ACM context and execute Stage 1 parsing + normalization."""
        acm_path = Path(__file__).resolve().parents[2] / "input_acm.json"
        if not acm_path.exists():
            raise FileNotFoundError(
                f"input_acm.json not found at {acm_path}. "
                "RepoMaster unified ACM mode requires this file."
            )

        try:
            stage1_result = load_stage1_acm_intake(acm_path)
        except ACMValidationError as exc:
            raise ValueError(f"ACM Stage 1 validation failed: {exc}") from exc

        if stage1_result.warnings:
            print("[acm-stage1] Warnings detected during ACM normalization:")
            for warning in stage1_result.warnings:
                print(f"  - {warning}")

        return stage1_result

    def _compact_acm_value(self, value: Any, depth: int = 0) -> Any:
        """Recursively compact ACM payload while preserving full structure semantics."""
        max_depth = 8
        max_dict_keys = 48
        max_list_items = 24
        max_str_len = 2000

        if depth > max_depth:
            return "[truncated-depth]"

        if isinstance(value, dict):
            compacted: dict[str, Any] = {}
            items = list(value.items())
            for index, (key, item_value) in enumerate(items):
                if index >= max_dict_keys:
                    compacted["__truncated_keys__"] = len(items) - max_dict_keys
                    break
                compacted[str(key)] = self._compact_acm_value(item_value, depth + 1)
            return compacted

        if isinstance(value, list):
            compacted_list: list[Any] = []
            for index, item in enumerate(value):
                if index >= max_list_items:
                    compacted_list.append(f"[truncated_items:{len(value) - max_list_items}]")
                    break
                compacted_list.append(self._compact_acm_value(item, depth + 1))
            return compacted_list

        if isinstance(value, str):
            normalized = value.strip()
            if len(normalized) > max_str_len:
                return normalized[:max_str_len] + " ...[truncated]"
            return normalized

        if isinstance(value, (int, float, bool)) or value is None:
            return value

        return str(value)

    def _compact_acm_context(self, acm: dict) -> dict:
        """Compact full ACM object for context windows while preserving all key sections."""
        if not isinstance(acm, dict):
            return {}
        compacted = self._compact_acm_value(acm)
        return compacted if isinstance(compacted, dict) else {}

    def _resolve_repo_search_query(self, task: str) -> str:
        """Resolve repository search query strictly from ACM repomaster_query."""
        search_intent = self._acm_context.get("search_intent", {}) if isinstance(self._acm_context, dict) else {}
        if isinstance(search_intent, dict):
            repomaster_query = search_intent.get("repomaster_query")
            if isinstance(repomaster_query, str) and repomaster_query.strip():
                return repomaster_query.strip()

        raise ValueError(
            "input_acm.json is missing search_intent.repomaster_query. "
            "RepoMaster ACM mode requires this field and does not accept external query input."
        )

    def _get_stage1_output_or_raise(self) -> Stage1IntakeResult:
        """Return validated Stage 1 output used by scheduler orchestration."""
        stage1 = getattr(self, "_acm_stage1", None)
        if stage1 is None or not isinstance(stage1, Stage1IntakeResult):
            raise ValueError(
                "Stage 1 ACM output is unavailable. "
                "Ensure ACM intake parsing and normalization completed before orchestration."
            )

        if not isinstance(stage1.canonical_manifest, dict) or not stage1.canonical_manifest:
            raise ValueError("Stage 1 canonical_manifest is missing or invalid")

        return stage1

    def _build_github_search_prompt(self, stage1_output: Stage1IntakeResult, repomaster_query: str) -> str:
        """Build a minimal retrieval prompt: query + ACM reference only."""
        acm_reference_path = (Path(__file__).resolve().parents[2] / "input_acm.json").as_posix()
        prompt = f"""
Please search for GitHub repositories related to the task:
    <repomaster_query>
    {repomaster_query}
    </repomaster_query>

    <acm_reference>
    Use local ACM file as reference only: {acm_reference_path}
    Do not include the full ACM JSON body in your answer context.
    </acm_reference>

Please search the GitHub repository for the solution.
Follow these steps:
    1. Use repomaster_query as the primary and mandatory retrieval query.
    2. Do not paste or expand full input_acm.json into context; keep context compact.
    3. If additional constraints are needed, reference input_acm.json by path only.
    4. Carefully read the README file of each repository.
    5. Determine whether the code in the repository can solve this competition task based on the README file.
    6. After reading all the README files, select the top 5 GitHub repositories that are most suitable to solve this task (when selecting, consider the code quality of the repository and whether it is suitable to solve this task).
    7. Return only valid JSON. Do not include markdown, prose, headings, or TERMINATE.
    8. Return at most 5 ranked GitHub repositories in the format https://github.com/owner/repo.
The JSON format should be like this:
[
    {{
        "repo_name": "owner/repo",
        "repo_url": "https://github.com/owner/repo",
        "repo_description": "why this repo can solve the input_acm problem"
    }},
    ...
]
"""
        return prompt

    def _extract_repo_slug(self, value: str) -> Optional[str]:
        if not isinstance(value, str) or not value.strip():
            return None
        normalized = value.strip().replace(".git", "")
        match = re.search(r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", normalized, re.IGNORECASE)
        if not match:
            parts = [part for part in normalized.split("/") if part]
            if len(parts) >= 2 and all(" " not in part for part in parts[-2:]):
                owner, repo = parts[-2], parts[-1]
                return f"{owner}/{repo}"
            return None
        owner = match.group(1)
        repo = match.group(2)
        return f"{owner}/{repo}"

    def _canonical_repo_url(self, slug: str) -> str:
        return f"https://github.com/{slug}"

    def _safe_clip_score(self, value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(numeric, 1.0))

    def _parse_link_header_last_page(self, link_header: str) -> Optional[int]:
        if not isinstance(link_header, str) or not link_header.strip():
            return None
        match = re.search(r"[?&]page=(\d+)>;\s*rel=\"last\"", link_header)
        if not match:
            return None
        try:
            return int(match.group(1))
        except (TypeError, ValueError):
            return None

    def _star_score(self, stars: int, max_stars_in_pool: int) -> float:
        if stars <= 0:
            return 0.0
        log_stars = math.log1p(stars)
        log_max = math.log1p(max_stars_in_pool)
        if log_max <= 0:
            return 0.0
        return min(log_stars / log_max, 1.0)

    def _fork_penalty(self, stars: int, forks: int) -> float:
        if stars <= 0:
            return 0.0
        ratio = forks / stars
        if ratio > 0.5:
            return 0.85
        return 1.0

    def _commit_score(self, commit_data: dict[str, Any]) -> float:
        score = 0.0

        last_commit_date = commit_data.get("last_commit_date")
        if isinstance(last_commit_date, str) and last_commit_date.strip():
            try:
                last = datetime.fromisoformat(last_commit_date.replace("Z", "+00:00"))
                days_ago = (datetime.now(timezone.utc) - last).days
                if days_ago <= 30:
                    recency = 1.0
                elif days_ago <= 90:
                    recency = 0.85
                elif days_ago <= 180:
                    recency = 0.65
                elif days_ago <= 365:
                    recency = 0.40
                elif days_ago <= 730:
                    recency = 0.15
                else:
                    recency = 0.0
                score += 0.6 * recency
            except Exception:
                pass

        weekly_counts = commit_data.get("weekly_counts", [])
        if isinstance(weekly_counts, list) and weekly_counts:
            safe_counts = []
            for item in weekly_counts:
                try:
                    safe_counts.append(int(item))
                except (TypeError, ValueError):
                    continue

            if safe_counts:
                recent_12_weeks = safe_counts[-12:]
                avg_commits = sum(recent_12_weeks) / len(recent_12_weeks)
                velocity = min(avg_commits / 10.0, 1.0)
                score += 0.4 * velocity

        return self._safe_clip_score(score)

    def _issue_score(self, issue_data: dict[str, Any]) -> float:
        open_c = int(issue_data.get("open_count", 0) or 0)
        closed_c = int(issue_data.get("closed_count", 0) or 0)
        total = open_c + closed_c

        if total < 5:
            return 0.5

        resolution_rate = closed_c / total
        overload_penalty = 1.0 if open_c < 200 else 0.80
        return self._safe_clip_score(resolution_rate * overload_penalty)

    def _structural_readme_score(self, readme: str) -> float:
        if not isinstance(readme, str) or not readme.strip():
            return 0.0

        checks = {
            "has_installation": bool(re.search(r"pip install|npm install|setup\\.py", readme, re.IGNORECASE)),
            "has_code_example": bool(re.search(r"```", readme)),
            "has_usage_section": bool(re.search(r"## usage|## quickstart|## getting started", readme, re.IGNORECASE)),
            "has_architecture": bool(re.search(r"## architecture|## how it works|## design", readme, re.IGNORECASE)),
            "has_requirements": bool(re.search(r"requirements|dependencies|prerequisites", readme, re.IGNORECASE)),
            "length_adequate": len(readme) > 500,
            "has_links": bool(re.search(r"https?://", readme)),
        }
        weights = {
            "has_installation": 0.15,
            "has_code_example": 0.25,
            "has_usage_section": 0.20,
            "has_architecture": 0.15,
            "has_requirements": 0.10,
            "length_adequate": 0.10,
            "has_links": 0.05,
        }

        score = 0.0
        for key, passed in checks.items():
            if passed:
                score += weights[key]
        return self._safe_clip_score(score)

    def _topic_match_score(self, candidate: dict[str, Any], failure_mode: str) -> float:
        """Score topic alignment depth for final ranking, clamped to [0.0, 1.0]."""
        signals = get_topic_signals(failure_mode)
        allow_set = {
            str(topic).strip().lower()
            for topic in signals.get("github_topics", [])
            if isinstance(topic, str) and topic.strip()
        }
        block_set = {
            str(topic).strip().lower()
            for topic in signals.get("block_topics", [])
            if isinstance(topic, str) and topic.strip()
        }
        keyword_signals = [
            str(keyword).strip().lower()
            for keyword in signals.get("keyword_signals", [])
            if isinstance(keyword, str) and keyword.strip()
        ]

        metadata = candidate.get("metadata", {}) if isinstance(candidate.get("metadata"), dict) else {}

        raw_topics = (
            metadata.get("topics")
            or candidate.get("fetched_topics")
            or candidate.get("search_topics")
            or candidate.get("topics")
            or []
        )
        repo_topics = {
            str(topic).strip().lower()
            for topic in raw_topics
            if isinstance(topic, str) and topic.strip()
        }

        description = str(
            metadata.get("description")
            or candidate.get("repo_description")
            or candidate.get("description")
            or ""
        ).lower()
        repo_name = str(
            candidate.get("full_name")
            or candidate.get("repo_name")
            or candidate.get("slug")
            or ""
        ).lower()
        combined = f"{repo_name} {description}".strip()

        score = 0.0

        if repo_topics.intersection(block_set):
            score -= 0.50

        matched_topics = repo_topics.intersection(allow_set)
        score += min(len(matched_topics) * 0.20, 1.0)

        keyword_hits = [kw for kw in keyword_signals if kw in combined]
        score += min(len(keyword_hits) * 0.10, 0.50)

        if not repo_topics:
            score -= 0.10

        return self._safe_clip_score(score)

    def _build_problem_spec(self, stage1_output: Stage1IntakeResult) -> dict[str, str]:
        manifest = stage1_output.canonical_manifest if isinstance(stage1_output.canonical_manifest, dict) else {}
        problem_domain = manifest.get("problem_domain", {}) if isinstance(manifest, dict) else {}
        codebase_context = manifest.get("codebase_context", {}) if isinstance(manifest, dict) else {}

        error_name = "UnknownError"
        failure_mode = "Unknown failure mode"
        error_description = "No description available"

        if isinstance(problem_domain, dict):
            error_name = str(problem_domain.get("error_type") or error_name)
            failure_mode = str(problem_domain.get("symptom") or failure_mode)
            failing_node = str(problem_domain.get("failing_node") or "")
            if failing_node and failing_node not in failure_mode:
                failure_mode = f"{failure_mode}; failing_node={failing_node}"

        if isinstance(codebase_context, dict):
            architecture_pattern = str(codebase_context.get("architecture_pattern") or "")
            filepath = str(codebase_context.get("filepath") or "")
            primary_slice = str(codebase_context.get("primary_slice") or "")
            parts = [part for part in [architecture_pattern, filepath, primary_slice[:600]] if part]
            if parts:
                error_description = " | ".join(parts)

        return {
            "error_name": error_name,
            "failure_mode": failure_mode,
            "error_description": error_description,
        }

    def _resolve_failure_mode_key(self, stage1_output: Stage1IntakeResult) -> str:
        """Resolve the best category key for topic/keyword candidate filtering."""
        manifest = stage1_output.canonical_manifest if isinstance(stage1_output.canonical_manifest, dict) else {}
        problem_domain = manifest.get("problem_domain", {}) if isinstance(manifest, dict) else {}
        codebase_context = manifest.get("codebase_context", {}) if isinstance(manifest, dict) else {}

        symptom = str(problem_domain.get("symptom") or "").strip().lower()
        failing_node = str(problem_domain.get("failing_node") or "").strip().lower()
        error_type = str(problem_domain.get("error_type") or "").strip().lower()
        architecture_pattern = str(codebase_context.get("architecture_pattern") or "").strip().lower()
        combined = " ".join([symptom, failing_node, error_type, architecture_pattern]).strip()

        direct_signals = get_topic_signals(symptom)
        if direct_signals.get("github_topics") or direct_signals.get("keyword_signals") or direct_signals.get("block_topics"):
            return symptom

        if any(token in combined for token in ["memory", "recall", "vector", "rag", "embedding"]):
            return "memory"
        if any(token in combined for token in ["tool", "function call", "tool calling", "call_tool", "api integration"]):
            return "tool_call"
        if any(token in combined for token in ["correction", "self-correct", "retry", "feedback", "refine", "repair"]):
            return "correction_loop"
        if any(token in combined for token in ["context window", "token", "summar", "long context", "truncat", "context"]):
            return "context_management"
        if any(token in combined for token in ["multi-agent", "multi agent", "orchestration", "swarm", "coordination"]):
            return "multi_agent"
        if any(token in combined for token in ["reasoning", "planning", "tree", "chain of thought", "mcts", "search"]):
            return "algorithm"

        return symptom or "unknown"

    def _extract_json_object_or_list(self, text: str) -> Any:
        if not isinstance(text, str) or not text.strip():
            return None

        content = text.strip()

        code_block_matches = re.findall(r"```(?:json)?\\s*(.*?)```", content, flags=re.IGNORECASE | re.DOTALL)
        for candidate in code_block_matches:
            candidate = candidate.strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        for idx, ch in enumerate(content):
            if ch not in "[{":
                continue
            try:
                parsed, _ = decoder.raw_decode(content[idx:])
                return parsed
            except json.JSONDecodeError:
                continue

        return None

    def _extract_candidates_from_deepsearch(self, deepsearch_result: str) -> list[dict[str, Any]]:
        parsed = self._extract_json_object_or_list(deepsearch_result)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]

        if isinstance(parsed, dict):
            for key in ["repositories", "repos", "items", "results", "data"]:
                value = parsed.get(key)
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]

        return []

    def _extract_repo_candidates_from_text(self, text: str) -> list[dict[str, Any]]:
        if not isinstance(text, str):
            return []

        matches = re.findall(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", text)
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        for url in matches:
            slug = self._extract_repo_slug(url)
            if not slug or slug in seen:
                continue
            seen.add(slug)
            candidates.append(
                {
                    "repo_name": slug,
                    "repo_url": self._canonical_repo_url(slug),
                    "repo_description": "Candidate extracted from deepsearch repository research.",
                }
            )
            if len(candidates) >= 5:
                break

        return candidates

    async def _finalize_repo_search_json(self, deepsearch_result: str, original_query: str) -> str:
        extracted = self._extract_repo_candidates_from_text(deepsearch_result)
        if extracted:
            return json.dumps(extracted, ensure_ascii=False, indent=2)

        from autogen.oai import OpenAIWrapper

        prompt = f"""
The previous GitHub repository search did not return valid JSON.

Original search task:
{original_query}

Collected research text:
{deepsearch_result[:12000]}

Return only top 5 repo URLs JSON.
Use exactly this JSON array format and no other text:
[
  {{
    "repo_name": "owner/repo",
    "repo_url": "https://github.com/owner/repo",
    "repo_description": "short reason this repo can help solve the input_acm problem"
  }}
]
""".strip()

        client = OpenAIWrapper(**self.llm_config)
        response = await asyncio.to_thread(client.create, messages=[{"role": "user", "content": prompt}])
        self._print_llm_usage(response, label="repo-json-finalizer")
        content = ""
        if hasattr(response, "choices") and response.choices:
            content = response.choices[0].message.content or ""
        return content.strip()

    def _print_llm_usage(self, response, label: str = "scheduler") -> None:
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

    def _build_fast_github_queries(self, repomaster_query: str, stage1_output: Stage1IntakeResult) -> list[str]:
        manifest = stage1_output.canonical_manifest if isinstance(stage1_output.canonical_manifest, dict) else {}
        problem_domain = manifest.get("problem_domain", {}) if isinstance(manifest, dict) else {}
        codebase_context = manifest.get("codebase_context", {}) if isinstance(manifest, dict) else {}

        symptom = str(problem_domain.get("symptom") or "")
        failing_node = str(problem_domain.get("failing_node") or "")
        architecture = str(codebase_context.get("architecture_pattern") or "")

        raw_queries = [
            repomaster_query,
            f"{repomaster_query} in:name,description,readme",
            f"{symptom} {failing_node} {architecture}".strip(),
            f"{symptom} {failing_node} {architecture} in:name,description,readme".strip(),
            "langgraph tool calling in:name,description,readme",
            "langgraph tools agent in:name,description,readme",
            "langchain tools agent timeout in:name,description,readme",
            "langgraph tool calling timeout",
            "langgraph TimeoutWarning call_tool_node",
            "call_tool_node langgraph",
            "langchain langgraph agent tools timeout",
        ]

        queries: list[str] = []
        for query in raw_queries:
            cleaned = " ".join(str(query).split())
            if cleaned and cleaned not in queries:
                queries.append(cleaned)
        return queries

    async def _search_github_repo_candidates(
        self,
        repomaster_query: str,
        stage1_output: Stage1IntakeResult,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        github_pat = os.getenv("GITHUB_PAT", "").strip()
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if github_pat:
            headers["Authorization"] = f"Bearer {github_pat}"

        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()
        timeout = aiohttp.ClientTimeout(total=20)

        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            for query in self._build_fast_github_queries(repomaster_query, stage1_output):
                if len(candidates) >= limit:
                    break
                try:
                    async with session.get(
                        "https://api.github.com/search/repositories",
                        params={
                            "q": query,
                            "sort": "stars",
                            "order": "desc",
                            "per_page": min(10, limit),
                        },
                    ) as response:
                        if response.status != 200:
                            continue
                        payload = await response.json()
                except Exception:
                    continue

                for item in payload.get("items", []):
                    if not isinstance(item, dict):
                        continue
                    slug = self._extract_repo_slug(str(item.get("html_url") or item.get("full_name") or ""))
                    if not slug or slug in seen:
                        continue
                    seen.add(slug)
                    candidates.append(
                        {
                            "repo_name": slug,
                            "repo_url": self._canonical_repo_url(slug),
                            "repo_description": str(item.get("description") or ""),
                            "topics": item.get("topics") if isinstance(item.get("topics"), list) else [],
                        }
                    )
                    if len(candidates) >= limit:
                        break

        return candidates

    def _normalize_topics(self, topics: Any) -> list[str]:
        if not isinstance(topics, list):
            return []
        normalized: list[str] = []
        for topic in topics:
            if not isinstance(topic, str):
                continue
            cleaned = topic.strip().lower()
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
        return normalized

    def _fetch_topics_from_search_result(self, repo_item: dict[str, Any]) -> list[str]:
        if not isinstance(repo_item, dict):
            return []
        return self._normalize_topics(repo_item.get("topics", []))

    async def _fetch_repo_topics(
        self,
        session: aiohttp.ClientSession,
        owner: str,
        repo: str,
    ) -> list[str]:
        slug = f"{owner}/{repo}".lower()
        cached = self._repo_topics_cache.get(slug)
        if cached is not None:
            return list(cached)

        url = f"https://api.github.com/repos/{owner}/{repo}/topics"
        headers = {
            "Accept": "application/vnd.github.mercy-preview+json",
        }

        topics: list[str] = []
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    payload = await response.json()
                    if isinstance(payload, dict):
                        topics = self._normalize_topics(payload.get("names", []))
        except Exception:
            topics = []

        self._repo_topics_cache[slug] = list(topics)
        return topics

    async def _enrich_candidate_with_topics(
        self,
        session: aiohttp.ClientSession,
        candidate: dict[str, Any],
        owner: str,
        repo: str,
        use_search_payload: bool = True,
    ) -> dict[str, Any]:
        topics: list[str] = []

        if use_search_payload:
            topics = self._fetch_topics_from_search_result(candidate)

        if not topics:
            topics = await self._fetch_repo_topics(session, owner, repo)

        candidate["fetched_topics"] = topics
        return candidate

    async def _fetch_repo_metadata_and_readme(
        self,
        session: aiohttp.ClientSession,
        owner: str,
        repo: str,
        candidate: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        api_base = "https://api.github.com"
        repo_api_url = f"{api_base}/repos/{owner}/{repo}"
        readme_api_url = f"{repo_api_url}/readme"

        metadata: dict[str, Any] = {
            "stars": 0,
            "forks": 0,
            "open_issues": 0,
            "description": "",
            "readme": "",
            "topics": [],
            "commit_data": {
                "commits": 0,
                "last_commit_date": None,
                "weekly_counts": [],
            },
            "issue_data": {
                "open_count": 0,
                "closed_count": 0,
            },
        }

        try:
            async with session.get(repo_api_url) as response:
                if response.status == 200:
                    payload = await response.json()
                    if isinstance(payload, dict):
                        metadata["stars"] = int(payload.get("stargazers_count") or 0)
                        metadata["forks"] = int(payload.get("forks_count") or 0)
                        open_issues_count = int(payload.get("open_issues_count") or 0)
                        metadata["open_issues"] = open_issues_count
                        metadata["description"] = str(payload.get("description") or "")
                        metadata["issue_data"]["open_count"] = open_issues_count
                        metadata["topics"] = self._normalize_topics(payload.get("topics", []))
        except Exception:
            pass

        if isinstance(candidate, dict):
            enriched_candidate = await self._enrich_candidate_with_topics(
                session=session,
                candidate=candidate,
                owner=owner,
                repo=repo,
                use_search_payload=True,
            )
            payload_topics = self._normalize_topics(enriched_candidate.get("fetched_topics", []))
            if payload_topics:
                metadata["topics"] = payload_topics
        elif not metadata["topics"]:
            metadata["topics"] = await self._fetch_repo_topics(session, owner, repo)

        commits_api_url = f"{repo_api_url}/commits"
        commits_params = {
            "per_page": 1,
        }
        try:
            async with session.get(commits_api_url, params=commits_params) as response:
                if response.status == 200:
                    payload = await response.json()
                    if isinstance(payload, list) and payload:
                        first = payload[0] if isinstance(payload[0], dict) else {}
                        commit_obj = first.get("commit", {}) if isinstance(first, dict) else {}
                        committer_obj = commit_obj.get("committer", {}) if isinstance(commit_obj, dict) else {}
                        author_obj = commit_obj.get("author", {}) if isinstance(commit_obj, dict) else {}
                        metadata["commit_data"]["last_commit_date"] = (
                            committer_obj.get("date")
                            or author_obj.get("date")
                        )

                    link_header = response.headers.get("Link")
                    last_page = self._parse_link_header_last_page(link_header)
                    if isinstance(last_page, int) and last_page > 0:
                        metadata["commit_data"]["commits"] = last_page
                    elif isinstance(payload, list):
                        metadata["commit_data"]["commits"] = len(payload)
        except Exception:
            pass

        commit_activity_url = f"{repo_api_url}/stats/commit_activity"
        try:
            async with session.get(commit_activity_url) as response:
                if response.status == 200:
                    activity = await response.json()
                    if isinstance(activity, list):
                        weekly_counts = []
                        for week in activity:
                            if not isinstance(week, dict):
                                continue
                            try:
                                weekly_counts.append(int(week.get("total", 0) or 0))
                            except (TypeError, ValueError):
                                continue
                        metadata["commit_data"]["weekly_counts"] = weekly_counts
                elif response.status == 202:
                    metadata["commit_data"]["weekly_counts"] = []
        except Exception:
            pass

        closed_issues_search_url = "https://api.github.com/search/issues"
        closed_issues_params = {
            "q": f"repo:{owner}/{repo} type:issue state:closed",
            "per_page": 1,
        }
        try:
            async with session.get(closed_issues_search_url, params=closed_issues_params) as response:
                if response.status == 200:
                    payload = await response.json()
                    if isinstance(payload, dict):
                        metadata["issue_data"]["closed_count"] = int(payload.get("total_count") or 0)
        except Exception:
            pass

        readme_headers = {
            "Accept": "application/vnd.github.raw+json",
        }
        try:
            async with session.get(readme_api_url, headers=readme_headers) as response:
                if response.status == 200:
                    metadata["readme"] = await response.text()
        except Exception:
            pass

        return metadata

    def _call_semantic_readme_llm(self, readme: str, problem_spec: dict[str, str]) -> tuple[float, str]:
        prompt = f"""
You are evaluating a GitHub repository README for relevance to a specific agent failure.

PROBLEM:
- Error: {problem_spec['error_name']}
- Failure mode: {problem_spec['failure_mode']}
- Description: {problem_spec['error_description']}

README (first 2000 chars):
{readme[:2000]}

Score this README's relevance from 0.0 to 1.0 based on:
- Does it describe patterns that address the failure mode? (most important)
- Does it show code for error handling / retry / memory / correction?
- Is the solution approach clearly explained?

Return ONLY a JSON: {{"score": 0.0, "reason": "one sentence"}}
""".strip()

        from autogen.oai import OpenAIWrapper

        client = OpenAIWrapper(**self.llm_config)
        response = client.create(messages=[{"role": "user", "content": prompt}])
        self._print_llm_usage(response, label="readme-semantic-score")
        content = ""
        if hasattr(response, "choices") and response.choices:
            content = response.choices[0].message.content or ""

        parsed = self._extract_json_object_or_list(content)
        if isinstance(parsed, dict):
            score = self._safe_clip_score(parsed.get("score", 0.0))
            reason = str(parsed.get("reason") or "semantic relevance scored")
            return score, reason

        return 0.0, "semantic score unavailable"

    def _semantic_readme_score(self, readme: str, problem_spec: dict[str, str]) -> tuple[float, str]:
        if not isinstance(readme, str) or not readme.strip():
            return 0.0, "README missing"

        if os.getenv("ENABLE_SEMANTIC_REPO_SCORING", "false").strip().lower() not in {"1", "true", "yes"}:
            text = readme[:8000].lower()
            terms = [
                str(problem_spec.get("error_name") or "").lower(),
                str(problem_spec.get("failure_mode") or "").lower(),
                "timeout",
                "tool",
                "agent",
                "langgraph",
                "langchain",
                "retry",
                "graph",
            ]
            hits = sum(1 for term in terms if term and term in text)
            score = self._safe_clip_score(hits / 6.0)
            return score, "fast README keyword relevance"

        try:
            return self._call_semantic_readme_llm(readme, problem_spec)
        except Exception as exc:
            return 0.0, f"semantic scoring failed: {type(exc).__name__}"

    def _score_repo_pool(
        self,
        repos: list[dict[str, Any]],
        problem_spec: dict[str, str],
        github_token: str,
    ) -> list[RepoScore]:
        del github_token  # Metadata is already fetched upstream in this pipeline.

        if not repos:
            return []

        max_stars = max(
            int((r.get("metadata", {}) if isinstance(r, dict) else {}).get("stars", 0) or 0)
            for r in repos
        ) or 1

        scored: list[RepoScore] = []

        for repo in repos:
            if not isinstance(repo, dict):
                continue

            metadata = repo.get("metadata", {}) if isinstance(repo.get("metadata"), dict) else {}
            commit_data = metadata.get("commit_data", {}) if isinstance(metadata.get("commit_data"), dict) else {}
            issue_data = metadata.get("issue_data", {}) if isinstance(metadata.get("issue_data"), dict) else {}

            stars = int(metadata.get("stars", 0) or 0)
            forks = int(metadata.get("forks", 0) or 0)
            readme = str(metadata.get("readme") or "")

            s_star = self._star_score(stars, max_stars)
            s_star *= self._fork_penalty(stars, forks)
            s_star = self._safe_clip_score(s_star)

            structural = self._structural_readme_score(readme)
            semantic_score, semantic_reason = self._semantic_readme_score(readme, problem_spec)
            s_readme = self._safe_clip_score(0.40 * structural + 0.60 * semantic_score)

            s_commit = self._commit_score(commit_data)
            s_issue = self._issue_score(issue_data)
            s_topic = self._topic_match_score(repo, str(problem_spec.get("failure_mode") or ""))

            final = (
                WEIGHTS["star"] * s_star
                + WEIGHTS["readme"] * s_readme
                + WEIGHTS["commit"] * s_commit
                + WEIGHTS["issue"] * s_issue
                + WEIGHTS["topic_match"] * s_topic
            )

            scored.append(
                RepoScore(
                    repo=str(repo.get("slug") or repo.get("full_name") or repo.get("repo_name") or "unknown/unknown"),
                    final_score=round(final, 4),
                    star_score=round(s_star, 4),
                    readme_score=round(s_readme, 4),
                    commit_score=round(s_commit, 4),
                    issue_score=round(s_issue, 4),
                    topic_match_score=round(s_topic, 4),
                    reason=semantic_reason,
                )
            )

        return sorted(scored, key=lambda x: x.final_score, reverse=True)

    def _format_topic_collection(self, values: list[str], max_items: int = 8) -> str:
        normalized = [
            str(value).strip().lower()
            for value in values
            if isinstance(value, str) and str(value).strip()
        ]
        if not normalized:
            return "{}"

        preview = normalized[:max_items]
        suffix = ", ..." if len(normalized) > max_items else ""
        return "{" + ", ".join(preview) + suffix + "}"

    def _format_candidate_topics_for_audit(self, candidate: dict[str, Any], max_items: int = 4) -> str:
        raw_topics = (
            candidate.get("fetched_topics")
            or candidate.get("search_topics")
            or candidate.get("topics")
            or []
        )
        topics = self._normalize_topics(raw_topics)
        if not topics:
            return "[]"

        preview = topics[:max_items]
        suffix = ", ..." if len(topics) > max_items else ""
        return "[" + ", ".join(preview) + suffix + "]"

    def _format_ranked_repo_results(
        self,
        scored_results: list[dict[str, Any]],
        failure_mode_key: str,
        topic_signals: dict[str, list[str]],
        passed_candidates: list[dict[str, Any]],
        rejected_candidates: list[dict[str, Any]],
    ) -> str:
        lines = ["--- Scoring ---"]

        if not scored_results:
            lines.append("No repositories were scored.")
            return "\n".join(lines)

        for idx, item in enumerate(scored_results[:5], start=1):
            repo_score = item.get("repo_score", {}) if isinstance(item.get("repo_score"), dict) else {}
            signals = item.get("scoring_signals", {}) if isinstance(item.get("scoring_signals"), dict) else {}

            repo_name = str(item.get("repo") or item.get("repo_name") or "unknown/unknown")
            final_score = float(repo_score.get("final_score", 0.0) or 0.0)
            star_score = float(signals.get("star_score", 0.0) or 0.0)
            readme_score = float(signals.get("readme_score", 0.0) or 0.0)
            commit_score = float(signals.get("commit_score", 0.0) or 0.0)
            issue_score = float(signals.get("issue_score", 0.0) or 0.0)
            topic_score = float(signals.get("topic_match_score", 0.0) or 0.0)

            lines.append(f"Rank {idx}: {repo_name:<30} | {final_score:.3f}")
            lines.append(
                f"        star:{star_score:.2f} readme:{readme_score:.2f} "
                f"commit:{commit_score:.2f} issue:{issue_score:.2f} topic:{topic_score:.2f}"
            )

            filter_reason = str(item.get("filter_reason") or "")
            if "lenient" in filter_reason.lower() and topic_score <= 0.20:
                lines.append("        ^ Passed filter but topic score is low - ranked accordingly")

            if idx != min(len(scored_results), 5):
                lines.append("")

        return "\n".join(lines)

    async def _score_repository_candidates(
        self,
        deepsearch_result: str,
        stage1_output: Stage1IntakeResult,
    ) -> Optional[str]:
        candidates = self._extract_candidates_from_deepsearch(deepsearch_result)
        if not candidates:
            return None

        normalized_candidates: list[dict[str, Any]] = []
        seen_slugs: set[str] = set()

        for candidate in candidates:
            possible_url = (
                candidate.get("repo_url")
                or candidate.get("url")
                or candidate.get("link")
                or candidate.get("html_url")
                or candidate.get("repository")
            )
            possible_name = candidate.get("repo_name") or candidate.get("name") or candidate.get("full_name")
            slug = self._extract_repo_slug(str(possible_url or possible_name or ""))
            if not slug or slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            normalized_candidates.append(
                {
                    "slug": slug,
                    "repo_name": str(candidate.get("repo_name") or candidate.get("name") or slug),
                    "repo_url": self._canonical_repo_url(slug),
                    "repo_description": str(candidate.get("repo_description") or candidate.get("description") or candidate.get("snippet") or ""),
                    "search_topics": self._fetch_topics_from_search_result(candidate),
                }
            )

        if not normalized_candidates:
            return None

        failure_mode_key = self._resolve_failure_mode_key(stage1_output)
        passed_candidates, rejected_candidates = filter_candidates(
            candidates=normalized_candidates,
            failure_mode=failure_mode_key,
            strict=True,
            max_pass=20,
        )

        if len(passed_candidates) < 5:
            existing_slugs = {
                str(candidate.get("slug") or "")
                for candidate in passed_candidates
                if isinstance(candidate, dict)
            }
            for candidate in rejected_candidates:
                if len(passed_candidates) >= 20:
                    break
                slug = str(candidate.get("slug") or "")
                if not slug or slug in existing_slugs:
                    continue
                candidate["filter_reason"] = str(
                    candidate.get("filter_reason")
                    or "ranked by lenient fallback because strict topic filter produced fewer than 5 repos"
                )
                passed_candidates.append(candidate)
                existing_slugs.add(slug)

        if not passed_candidates:
            for candidate in normalized_candidates:
                candidate["filter_reason"] = "ranked by fallback scoring because strict topic filter found no pass"
            passed_candidates = normalized_candidates[:20]
            rejected_candidates = []

        normalized_candidates = passed_candidates

        github_pat = os.getenv("GITHUB_PAT", "").strip()
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if github_pat:
            headers["Authorization"] = f"Bearer {github_pat}"

        timeout = aiohttp.ClientTimeout(total=25)
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            tasks = []
            for item in normalized_candidates:
                owner, repo = item["slug"].split("/", 1)
                tasks.append(self._fetch_repo_metadata_and_readme(session, owner, repo, candidate=item))
            fetched = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, payload in enumerate(fetched):
            if isinstance(payload, Exception) or not isinstance(payload, dict):
                normalized_candidates[idx]["metadata"] = {
                    "stars": 0,
                    "forks": 0,
                    "open_issues": 0,
                    "description": normalized_candidates[idx]["repo_description"],
                    "readme": "",
                    "topics": normalized_candidates[idx].get("search_topics", []),
                    "commit_data": {
                        "commits": 0,
                        "last_commit_date": None,
                        "weekly_counts": [],
                    },
                    "issue_data": {
                        "open_count": 0,
                        "closed_count": 0,
                    },
                }
            else:
                normalized_candidates[idx]["metadata"] = payload
        problem_spec = self._build_problem_spec(stage1_output)
        problem_spec["failure_mode"] = failure_mode_key

        full_scores = self._score_repo_pool(
            normalized_candidates,
            problem_spec,
            github_pat,
        )

        by_slug = {item["slug"]: item for item in normalized_candidates}
        scored_results: list[dict[str, Any]] = []
        for score in full_scores:
            item = by_slug.get(score.repo, {})
            metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
            commit_data = metadata.get("commit_data", {}) if isinstance(metadata.get("commit_data"), dict) else {}
            issue_data = metadata.get("issue_data", {}) if isinstance(metadata.get("issue_data"), dict) else {}
            repo_topics = self._normalize_topics(metadata.get("topics", []))

            topic_signals = get_topic_signals(failure_mode_key)
            expected_topics = topic_signals.get("github_topics", []) if isinstance(topic_signals, dict) else []
            blocked_topics = topic_signals.get("block_topics", []) if isinstance(topic_signals, dict) else []

            expected_match = sorted(set(t for t in repo_topics if t in expected_topics))
            blocked_match = sorted(set(t for t in repo_topics if t in blocked_topics))

            scored_results.append(
                {
                    "repo_name": item.get("repo_name", score.repo),
                    "repo": score.repo,
                    "repo_url": item.get("repo_url", self._canonical_repo_url(score.repo)),
                    "repo_description": str(metadata.get("description") or item.get("repo_description") or ""),
                    "rank_compare": len(scored_results) + 1,
                    "filter_reason": str(item.get("filter_reason") or ""),
                    "repo_score": asdict(score),
                    "scoring_signals": {
                        "star_score": score.star_score,
                        "readme_score": score.readme_score,
                        "commit_score": score.commit_score,
                        "issue_score": score.issue_score,
                        "topic_match_score": score.topic_match_score,
                        "reason": score.reason,
                        "filter_gate": {
                            "failure_mode": failure_mode_key,
                            "passed_candidates": len(normalized_candidates),
                            "rejected_candidates": len(rejected_candidates),
                        },
                        "topic_components": {
                            "fetched_topics": repo_topics,
                            "expected_topic_hits": expected_match,
                            "blocked_topic_hits": blocked_match,
                        },
                        "commit_components": {
                            "commits": int(commit_data.get("commits", 0) or 0),
                            "last_commit_date": commit_data.get("last_commit_date"),
                            "weekly_counts": commit_data.get("weekly_counts", []),
                        },
                        "issue_components": {
                            "open_count": int(issue_data.get("open_count", 0) or 0),
                            "closed_count": int(issue_data.get("closed_count", 0) or 0),
                        },
                    },
                }
            )

        return self._format_ranked_repo_results(
            scored_results=scored_results[:5],
            failure_mode_key=failure_mode_key,
            topic_signals=topic_signals,
            passed_candidates=normalized_candidates,
            rejected_candidates=rejected_candidates,
        )
       
    def initialize_agents(self, **kwargs):    
        """
        Initialize scheduler agent and user agent.
        """
        self.scheduler = ExtendedAssistantAgent(
            name="scheduler_agent",
            system_message=scheduler_system_message,
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").endswith("TERMINATE"),
            llm_config=self.llm_config,
        )

        self.user_proxy = ExtendedUserProxyAgent(
            name="user_proxy",
            system_message=user_proxy_system_message,
            llm_config=self.llm_config,
            is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").endswith("TERMINATE"),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config=self.code_execution_config,
        )
    
    async def web_search(self, query: Annotated[str, "Query for general web search to get real-time information or answer non-code-related questions"]) -> str:
        """
        Perform general web search to find real-time information or solve general problems that don't require code.
        
        This method allows the agent to search the internet for the latest information, such as current events and recent data.
        It is suitable for scenarios that require information beyond the model's knowledge scope or need the latest information.
        
        Features:
        - Search the internet and generate answers based on results
        - Provide real-time information about current events and latest data
        - Return formatted search result information
        - Access information beyond the model's knowledge cutoff date
        """
        return await self.repo_searcher.deep_search(query)
    
    async def github_repo_search(self, task: Annotated[str, "Description of tasks that need to be solved through GitHub repositories, used to search for the most relevant code libraries"]) -> str:
        """
        Search for relevant code repositories on GitHub based on task description.
        
        This method is designed to find the most suitable GitHub repositories based on user tasks. It analyzes
        the README files of repositories to determine their relevance and returns a list of repositories
        most suitable for solving the task.

        Returns:
            A JSON string containing a list of the most matching repository information.
        """
        stage1_output = self._get_stage1_output_or_raise()
        repomaster_query = self._resolve_repo_search_query(task)
        query = self._build_github_search_prompt(stage1_output, repomaster_query)
        overall_budget = max(120, int(os.getenv("REPO_RANKING_TIMEOUT", "900")))
        discovery_budget = max(60, int(os.getenv("REPO_DISCOVERY_TIMEOUT", str(min(600, overall_budget - 180)))))

        print("[scheduler] Deepsearch repo discovery started.")
        original_deepsearch_timeout = getattr(self.repo_searcher, "deepsearch_timeout", discovery_budget)
        self.repo_searcher.deepsearch_timeout = min(int(original_deepsearch_timeout), discovery_budget)
        try:
            deepsearch_text = await self.repo_searcher.deep_search(
                query,
                summary_prompt=GITHUB_REPO_SEARCH_SUMMARY_PROMPT,
            )
        except TimeoutError:
            print("[scheduler] Deepsearch discovery budget reached; ranking candidates found so far.")
            deepsearch_text = self.repo_searcher.get_partial_research_text()
        finally:
            self.repo_searcher.deepsearch_timeout = original_deepsearch_timeout

        candidates_for_scoring = self._extract_candidates_from_deepsearch(deepsearch_text)
        if not candidates_for_scoring:
            candidates_for_scoring = self._extract_repo_candidates_from_text(deepsearch_text)
        deepsearch_result = json.dumps(candidates_for_scoring, ensure_ascii=False, indent=2)

        if len(candidates_for_scoring) < 5:
            print(
                f"[scheduler] Proceeding with {len(candidates_for_scoring)} candidates. "
                "Deepsearch did not collect 5 repo candidates within the discovery budget."
            )

        try:
            scored = await self._score_repository_candidates(deepsearch_result, stage1_output)
            if scored:
                return scored
        except Exception as e:
            artifact_path = write_error_artifact(
                component="scheduler",
                operation="github_repo_scoring",
                error=e,
                context={
                    "query_fingerprint": text_fingerprint(repomaster_query),
                    "deepsearch_result_fingerprint": text_fingerprint(deepsearch_result),
                },
            )
            if artifact_path:
                print(f"[scheduler] Scoring artifact: {artifact_path}")

        return deepsearch_result
    
    def run_repository_agent(
        self, 
        task_description: Annotated[str, "Task description that the user needs to solve, maintain the completeness of the task description without omitting any information"],
        repository: Annotated[str, "Repository path or URL. Can be a GitHub repository URL (format: https://github.com/repo_name/repo_name) or local repository absolute path (e.g.: /path/to/my/project)"],
        input_data: Annotated[Optional[str], "JSON string representing local input data. Must be provided when the user task explicitly mentions or implies the need to use local files as input. Format: '[{\"path\": \"local input data path\", \"description\": \"input data description\"}]'. If the task does not require local input data, an empty list '[]' can be passed."] = None,
        repo_type: Annotated[Optional[str], "Repository type, optional values: 'github' or 'local'. If not specified, it will be automatically detected"] = None
    ):
        """
        Unified interface for executing user tasks based on specified repository (GitHub or local).
        
        This method automatically detects or handles GitHub repositories or local repositories based on specified type,
        then calls the task manager and agent runner to complete the task execution process based on the provided
        task description and input data. The entire process includes:
        1. Automatically detect repository type or use specified type
        2. Validate repository path or URL validity
        3. Validate and process input data
        4. Initialize task environment (create working directory, clone or copy repository, etc.)
        5. Run code agent to analyze and execute tasks
        
        Args:
            task_description: Detailed description of the task to be completed
            repository: Repository path or URL, supports GitHub URL or local absolute path
            input_data: Optional JSON string representing input data files
            repo_type: Optional repository type, automatically detected if not specified
            
        Returns:
            Result of agent executing the task, usually containing task completion status and output content description
        """
        # Automatically detect repository type
        if repo_type is None:
            if repository.startswith(('http://', 'https://')) and 'github.com' in repository:
                repo_type = 'github'
            elif os.path.exists(repository):
                repo_type = 'local'
            else:
                # Try to determine if it's a GitHub URL format
                if repository.startswith(('http://', 'https://')) or repository.count('/') >= 1:
                    repo_type = 'github'
                else:
                    raise ValueError(f"Unable to determine repository type. Please provide a valid GitHub URL or local path: {repository}")
        
        # Validate repository
        if repo_type == 'local':
            if not os.path.exists(repository):
                raise ValueError(f"Local repository path does not exist: {repository}")
        elif repo_type == 'github':
            # Basic GitHub URL format validation
            if not (repository.startswith(('http://', 'https://')) or 
                   ('github.com' in repository or repository.count('/') >= 1)):
                raise ValueError(f"Invalid GitHub repository URL format: {repository}")
        else:
            raise ValueError(f"Unsupported repository type: {repo_type}. Supported types: 'github', 'local'")
        
        # Validate and process input data
        if input_data:
            try:
                input_data = json.loads(input_data)
            except:
                raise ValueError("input_data format error, please check input data format")
            
            assert isinstance(input_data, list), "input_data must be of list type"
            for data in input_data:
                assert isinstance(data, dict), "Elements in input_data must be of dict type"
                assert 'path' in data, "Each data item must contain 'path' field"
                assert 'description' in data, "Each data item must contain 'description' field"
        else:
            input_data = []

        # Build configuration based on repository type
        if repo_type == 'github':
            repo_config = {
                "type": "github",
                "url": repository,
            }
        else:  # local
            repo_config = {
                "type": "local", 
                "path": repository,
            }

        args = argparse.Namespace(
            config_data={
                "repo": repo_config,
                "task_description": task_description,
                "input_data": input_data,
                "root_path": self.work_dir,
            },
            root_path='coding',
        )
        
        task_info = TaskManager.initialize_tasks(args)
        result = AgentRunner.run_agent(task_info, retry_times=1, work_dir=self.work_dir)        

        return result

    def run_general_code_assistant(
        self,
        task_description: Annotated[str, "Programming task or question that needs general coding assistance"],
        work_directory: Annotated[Optional[str], "Specific working directory for code execution. If not provided, uses default work directory"] = None
    ):
        """
        Provide general programming assistance without requiring a specific repository.
        
        This method creates a clean workspace and uses the code exploration agent to help with:
        - General programming questions and guidance
        - Writing and executing code snippets
        - Debugging and troubleshooting
        - Creating examples and demonstrations
        - Algorithm implementations
        - Code explanations and tutorials
        
        Args:
            task_description: Detailed description of the programming task or question
            work_directory: Optional specific working directory for code execution
            
        Returns:
            Result containing programming guidance, code examples, and execution results
        """
        import asyncio
        from src.core.agent_code_explore import CodeExplorer
        
        # Determine working directory
        work_dir = work_directory or self.work_dir
        
        # Create CodeExplorer instance for general programming assistance
        explorer = CodeExplorer(
            local_repo_path=None,
            work_dir=work_dir,
            task_type="general",
            use_venv=True,
            is_cleanup_venv=False,
        )
        
        # Enhance the task description for general programming assistance
        enhanced_task = f"""
You are a general programming assistant. Please help with the following task:

{task_description}

As a programming assistant, you can:
- Write and execute code to solve problems
- Provide programming guidance and explanations  
- Create practical examples and demonstrations
- Debug and troubleshoot issues
- Implement algorithms and data structures
- Explain programming concepts
- Create utility scripts and tools

Working directory: {work_dir}

Please provide comprehensive help including code examples, explanations, and practical solutions.
"""
        
        result = asyncio.run(explorer.a_code_analysis(enhanced_task, max_turns=20))
        return result

    def register_tools(self):
        """
        Register the enhanced toolkit required by the agent.
        """
        register_toolkits(
            [
                self.web_search,
                self.run_repository_agent,           # Unified repository processing mode
                self.run_general_code_assistant,     # General code assistant mode
                self.github_repo_search,
            ],
            self.scheduler,
            self.user_proxy,
        )

    def solve_task_with_repo(self, task: Annotated[str, "Detailed task description that user needs to solve"]) -> str:
        """
        Enhanced RepoMaster that can work with GitHub repositories, local repositories, or provide general programming assistance.
        
        This method is the main entry point of Enhanced RepoMaster, which automatically determines the best approach:
        
        **Three Working Modes:**
        1. **Web Search Mode**: Search the internet for real-time information, current events, or general knowledge
        2. **Repository Mode**: Search and use GitHub repositories or local repositories for specialized tasks with hierarchical analysis
        3. **General Code Assistant Mode**: Provide programming assistance without specific repositories
        
        **Auto-Mode Detection:**
        - Detects real-time information needs → Web Search Mode
        - Detects repository paths/URLs (GitHub or local) → Repository Mode
        - Detects general programming questions → General Code Assistant Mode  
        - Default behavior → Repository Mode
        
        **Process:**
        1. Analyze task requirements and detect appropriate mode
        2. Execute using the most suitable approach with advanced repository analysis
        3. Generate comprehensive solutions through hierarchical understanding
        4. Provide execution results and guidance with context optimization
        
        Args:
            task: Detailed task description that user needs to solve
            
        Returns:
            Complete solution report including analysis methods, execution results, and recommendations
        """
        try:
            stage1_output = self._get_stage1_output_or_raise()

            if not isinstance(stage1_output.canonical_manifest, dict) or not stage1_output.canonical_manifest:
                raise ValueError(
                    "input_acm.json is required for RepoMaster unified execution. "
                    "External task text input is disabled."
                )

            self._acm_context = stage1_output.canonical_manifest
            return asyncio.run(self.github_repo_search(task))
        except Exception as e:
            search_intent = self._acm_context.get("search_intent", {}) if isinstance(self._acm_context, dict) else {}
            stage1_obj = getattr(self, "_acm_stage1", None)
            stage1_query_pack_count = 0
            if isinstance(stage1_obj, Stage1IntakeResult) and isinstance(stage1_obj.query_pack, dict):
                stage1_query_pack_count = len(stage1_obj.query_pack.get("retrieval_queries", []))

            artifact_path = write_error_artifact(
                component="scheduler",
                operation="solve_task_with_repo",
                error=e,
                context={
                    "task": text_fingerprint(task),
                    "has_acm_context": bool(self._acm_context),
                    "acm_top_level_keys": list(self._acm_context.keys()) if isinstance(self._acm_context, dict) else [],
                    "has_search_intent": isinstance(search_intent, dict),
                    "has_repomaster_query": isinstance(search_intent, dict) and bool(search_intent.get("repomaster_query")),
                    "has_stage1_output": isinstance(stage1_obj, Stage1IntakeResult),
                    "stage1_query_pack_count": stage1_query_pack_count,
                    "work_dir": self.work_dir,
                },
            )
            setattr(e, "_repomaster_artifact_path", artifact_path)
            if artifact_path:
                print(f"[scheduler] Error artifact: {artifact_path}")
            raise

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

def load_env():
    from configs.oai_config import get_llm_config
    from dotenv import load_dotenv
    import uuid
    
    llm_config = get_llm_config()
    load_dotenv("configs/.env")
    work_dir = os.path.join(os.getcwd(), "coding", str(uuid.uuid4()))
    code_execution_config = {"work_dir": work_dir, "use_docker": False}
    
    return llm_config, code_execution_config

def main():
    
    llm_config, code_execution_config = load_env()
    repo_master = RepoMasterAgent(
        llm_config=llm_config,
        code_execution_config=code_execution_config,
    )
    import asyncio
    result = repo_master.solve_task_with_repo("What is the stock price of APPLE?")
    print(result)

def test_run_repo_agent():
    llm_config, code_execution_config = load_env()
    
    repo_master = RepoMasterAgent(
        llm_config=llm_config,
        code_execution_config=code_execution_config,
    )

    arguments = {'task_description': 'Extract all text content from the first page of a PDF file and save it to a txt file. The input PDF file path is: GitTaskBench/queries/PDFPlumber_01/input/PDFPlumber_01_input.pdf', 'github_url': 'https://github.com/spatie/pdf-to-text'}    
    
    result = repo_master.run_repository_agent(
        task_description=arguments['task_description'],
        repository=arguments['github_url'],
        input_data=None
    )
    print(result)

def test_run_all():
    llm_config, code_execution_config = load_env()
    
    repo_master = RepoMasterAgent(
        llm_config=llm_config,
        code_execution_config=code_execution_config,
    )
    task = "Help me convert '/data/huacan/Code/workspace/RepoMaster/data/DeepResearcher.pdf' to markdown and save"
    result = repo_master.solve_task_with_repo(task)
    print(result)

if __name__ == "__main__":
    test_run_all()
