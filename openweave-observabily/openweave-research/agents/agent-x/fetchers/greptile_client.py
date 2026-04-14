from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class GreptileClientError(RuntimeError):
    """Raised when Greptile semantic context retrieval fails."""


class GreptileClient:
    """Fallback semantic fetcher for cross-file repository context."""

    def __init__(
        self,
        api_key: str,
        github_token: str,
        client: httpx.AsyncClient | None = None,
        base_url: str = "https://api.greptile.com/v2",
        timeout: float = 45.0,
        remote: str = "github",
    ) -> None:
        self._api_key = api_key
        self._github_token = github_token
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._remote = remote

    async def fetch_semantic_context(
        self,
        repo: str,
        commit_id: str,
        function_name: str,
    ) -> dict[str, Any]:
        """Query Greptile for the most relevant cross-file context for a function."""

        payload = {
            "messages": [
                {
                    "id": f"{repo}:{commit_id}:{function_name}",
                    "role": "user",
                    "content": (
                        "Find the cross-file dependencies, upstream definitions, and "
                        f"relevant architecture context for the function "
                        f"`{function_name}` in repository `{repo}` at ref `{commit_id}`. "
                        "Prioritize the files and code references that explain how data "
                        "flows into or out of this function."
                    ),
                }
            ],
            "repositories": [
                {
                    "remote": self._remote,
                    "branch": commit_id,
                    "repository": repo,
                }
            ],
            "stream": False,
            "genius": True,
        }

        async with self._get_client() as client:
            try:
                response = await client.post(
                    f"{self._base_url}/query",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "X-GitHub-Token": self._github_token,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return {
                    "message": data.get("message", ""),
                    "dependencies": data.get("sources", []),
                }
            except httpx.HTTPStatusError as exc:
                logger.exception(
                    "Greptile returned %s while querying %s at %s for %s",
                    exc.response.status_code,
                    repo,
                    commit_id,
                    function_name,
                )
                raise GreptileClientError(
                    f"Greptile query failed with status {exc.response.status_code} "
                    f"for {repo}@{commit_id}:{function_name}"
                ) from exc
            except httpx.RequestError as exc:
                logger.exception(
                    "Greptile request failed while querying %s at %s for %s",
                    repo,
                    commit_id,
                    function_name,
                )
                raise GreptileClientError(
                    f"Greptile request failed for {repo}@{commit_id}:{function_name}"
                ) from exc
            except (TypeError, ValueError) as exc:
                logger.exception(
                    "Greptile payload parsing failed for %s at %s for %s",
                    repo,
                    commit_id,
                    function_name,
                )
                raise GreptileClientError(
                    f"Greptile returned an invalid payload for "
                    f"{repo}@{commit_id}:{function_name}"
                ) from exc

    def _get_client(self) -> httpx.AsyncClient | _AsyncClientContext:
        if self._client is not None:
            return _AsyncClientContext(self._client)
        return httpx.AsyncClient(timeout=self._timeout)


class _AsyncClientContext:
    """Wrap an injected client so the fetcher can use a single async-with path."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def __aenter__(self) -> httpx.AsyncClient:
        return self._client

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False
