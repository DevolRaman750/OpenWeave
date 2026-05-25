from typing import Any

from src.core.category_map import get_topic_signals


class TopicFilter:
    """Hard gate for repository candidates before expensive scoring work."""

    def __init__(self, failure_mode: str, strict: bool = False):
        signals = get_topic_signals(failure_mode)
        self.allow_topics = {
            str(topic).strip().lower()
            for topic in signals.get("github_topics", [])
            if isinstance(topic, str) and topic.strip()
        }
        self.block_topics = {
            str(topic).strip().lower()
            for topic in signals.get("block_topics", [])
            if isinstance(topic, str) and topic.strip()
        }
        self.kw_signals = [
            str(keyword).strip().lower()
            for keyword in signals.get("keyword_signals", [])
            if isinstance(keyword, str) and keyword.strip()
        ]
        self.strict = strict

    def passes(self, candidate: dict[str, Any]) -> tuple[bool, str]:
        """Return (passes, reason) for the three-stage topic/keyword gate."""
        if not isinstance(candidate, dict):
            return False, "invalid candidate payload"

        raw_topics = (
            candidate.get("fetched_topics")
            or candidate.get("search_topics")
            or candidate.get("topics")
            or []
        )
        repo_topics = {
            str(topic).strip().lower()
            for topic in raw_topics
            if isinstance(topic, str) and topic.strip()
        }

        repo_name = str(
            candidate.get("full_name")
            or candidate.get("repo_name")
            or candidate.get("slug")
            or ""
        ).lower()
        description = str(
            candidate.get("description")
            or candidate.get("repo_description")
            or ""
        ).lower()

        blocked = repo_topics.intersection(self.block_topics)
        if blocked:
            return False, f"blocked by topic: {sorted(blocked)}"

        matched = repo_topics.intersection(self.allow_topics)
        if matched:
            return True, f"matched topic: {sorted(matched)}"

        combined_text = f"{repo_name} {description}".strip()
        keyword_hits = [kw for kw in self.kw_signals if kw in combined_text]
        if keyword_hits:
            return True, f"keyword match: {keyword_hits[:2]}"

        if self.strict:
            return False, "no topic or keyword match (strict mode)"

        return True, "unmatched - passed by default (lenient)"


def filter_candidates(
    candidates: list[dict[str, Any]],
    failure_mode: str,
    strict: bool = False,
    max_pass: int = 20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split candidates into passed and rejected lists with strict fallback."""
    topic_filter = TopicFilter(failure_mode=failure_mode, strict=strict)

    passed: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for candidate in candidates:
        current = candidate if isinstance(candidate, dict) else {}
        ok, reason = topic_filter.passes(current)
        current["filter_reason"] = reason
        if ok:
            passed.append(current)
        else:
            rejected.append(current)

    if not passed and strict:
        return filter_candidates(
            candidates=candidates,
            failure_mode=failure_mode,
            strict=False,
            max_pass=max_pass,
        )

    safe_max_pass = max(int(max_pass or 0), 1)
    return passed[:safe_max_pass], rejected


__all__ = ["TopicFilter", "filter_candidates"]