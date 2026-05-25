from typing import Any

# Maps normalized failure_mode values to topic and keyword signals used in repo filtering.
FAILURE_MODE_TOPIC_MAP: dict[str, dict[str, list[str]]] = {
    "memory": {
        "github_topics": [
            "memory",
            "episodic-memory",
            "agent-memory",
            "long-term-memory",
            "working-memory",
            "vector-store",
            "retrieval-augmented-generation",
            "rag",
            "knowledge-base",
            "semantic-memory",
            "context-management",
        ],
        "keyword_signals": [
            "episodic",
            "semantic memory",
            "memory store",
            "remember",
            "recall",
            "persistent context",
            "vector db",
            "embedding store",
            "knowledge graph",
        ],
        "block_topics": [
            "game",
            "unity",
            "hardware",
            "embedded",
        ],
    },
    "tool_call": {
        "github_topics": [
            "tool-use",
            "function-calling",
            "tool-calling",
            "agent-tools",
            "llm-tools",
            "openai-functions",
            "api-integration",
            "mcp",
            "model-context-protocol",
            "plugin",
            "action-agent",
            "react-agent",
        ],
        "keyword_signals": [
            "tool use",
            "function call",
            "tool calling",
            "api call",
            "retry",
            "backoff",
            "timeout handling",
            "tool execution",
            "action space",
            "tool registry",
        ],
        "block_topics": [
            "frontend",
            "css",
            "android",
            "ios",
        ],
    },
    "correction_loop": {
        "github_topics": [
            "self-correction",
            "self-repair",
            "reflexion",
            "self-refinement",
            "iterative-refinement",
            "feedback-loop",
            "error-recovery",
            "self-debugging",
            "auto-correction",
            "retry-logic",
            "error-handling",
            "self-improvement",
            "reflection-agent",
        ],
        "keyword_signals": [
            "reflexion",
            "self refine",
            "self repair",
            "correction loop",
            "iterative fix",
            "error recovery",
            "retry strategy",
            "self debug",
            "feedback loop",
            "verbal reinforcement",
            "self critique",
        ],
        "block_topics": [
            "vision",
            "image-processing",
            "audio",
            "robotics",
        ],
    },
    "algorithm": {
        "github_topics": [
            "tree-of-thought",
            "chain-of-thought",
            "reasoning",
            "planning",
            "task-decomposition",
            "mcts",
            "beam-search",
            "monte-carlo",
            "search-algorithm",
            "problem-solving",
            "llm-reasoning",
        ],
        "keyword_signals": [
            "tree of thought",
            "chain of thought",
            "reasoning chain",
            "task planning",
            "decomposition",
            "search strategy",
            "sampling strategy",
            "beam search",
            "MCTS",
        ],
        "block_topics": [
            "sorting",
            "competitive-programming",
            "leetcode",
        ],
    },
    "context_management": {
        "github_topics": [
            "context-window",
            "token-management",
            "prompt-compression",
            "long-context",
            "context-pruning",
            "summarization",
            "sliding-window",
            "context-distillation",
        ],
        "keyword_signals": [
            "context window",
            "token limit",
            "prompt compression",
            "context overflow",
            "truncation",
            "sliding window",
            "summarize context",
            "context pruning",
        ],
        "block_topics": [],
    },
    "multi_agent": {
        "github_topics": [
            "multi-agent",
            "agent-coordination",
            "agent-communication",
            "autogen",
            "crewai",
            "agent-orchestration",
            "collaborative-agents",
            "swarm",
        ],
        "keyword_signals": [
            "multi agent",
            "agent coordination",
            "agent network",
            "orchestration",
            "swarm",
            "collaborative agents",
            "agent communication",
            "agent hierarchy",
        ],
        "block_topics": [
            "game-ai",
            "reinforcement-learning",
        ],
    },
}

_DEFAULT_TOPIC_SIGNALS: dict[str, list[str]] = {
    "github_topics": [],
    "keyword_signals": [],
    "block_topics": [],
}


def get_topic_signals(failure_mode: str) -> dict[str, list[str]]:
    """Return topic and keyword signals for a normalized failure_mode.

    Unknown modes fall back to empty signal lists.
    """
    if not isinstance(failure_mode, str):
        return {
            "github_topics": [],
            "keyword_signals": [],
            "block_topics": [],
        }

    key = failure_mode.strip().lower()
    signals = FAILURE_MODE_TOPIC_MAP.get(key)
    if not isinstance(signals, dict):
        return {
            "github_topics": [],
            "keyword_signals": [],
            "block_topics": [],
        }

    # Return a defensive copy so callers cannot mutate global configuration.
    return {
        "github_topics": list(signals.get("github_topics", [])),
        "keyword_signals": list(signals.get("keyword_signals", [])),
        "block_topics": list(signals.get("block_topics", [])),
    }


__all__ = ["FAILURE_MODE_TOPIC_MAP", "get_topic_signals"]
