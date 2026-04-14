from __future__ import annotations

import base64
import binascii
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class GitHubPorterError(RuntimeError):
    """Raised when GitHub content retrieval fails."""


class GitHubPorter:
    """Fetches repository files from the GitHub Contents API."""

    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        base_url: str = "https://api.github.com",
        timeout: float = 30.0,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def fetch_file_content(
        self,
        repo: str,
        commit_id: str,
        filepath: str,
        pat: str,
    ) -> str:
        """Return the raw UTF-8 source for a repo file at a specific ref."""

        async with self._get_client() as client:
            try:
                response = await client.get(
                    f"{self._base_url}/repos/{repo}/contents/{filepath}",
                    headers={
                        "Accept": "application/vnd.github+json",
                        "Authorization": f"Bearer {pat}",
                        "X-GitHub-Api-Version": "2022-11-28",
                    },
                    params={"ref": commit_id},
                )
                response.raise_for_status()
                payload = response.json()
                return self._decode_content(payload, repo=repo, filepath=filepath)
            except httpx.HTTPStatusError as exc:
                logger.exception(
                    "GitHub content API returned %s for %s at %s",
                    exc.response.status_code,
                    repo,
                    filepath,
                )
                raise GitHubPorterError(
                    f"GitHub content fetch failed with status "
                    f"{exc.response.status_code} for {repo}:{filepath}@{commit_id}"
                ) from exc
            except httpx.RequestError as exc:
                logger.exception(
                    "GitHub content API request failed for %s at %s",
                    repo,
                    filepath,
                )
                raise GitHubPorterError(
                    f"GitHub content request failed for {repo}:{filepath}@{commit_id}"
                ) from exc
            except (TypeError, ValueError, binascii.Error, UnicodeDecodeError) as exc:
                logger.exception(
                    "GitHub content decoding failed for %s at %s",
                    repo,
                    filepath,
                )
                raise GitHubPorterError(
                    f"GitHub content payload was invalid for {repo}:{filepath}@{commit_id}"
                ) from exc

    async def fetch_file(
        self,
        repo: str,
        commit_id: str,
        filepath: str,
        pat: str,
    ) -> str:
        """Compatibility alias for the main orchestrator flow."""

        return await self.fetch_file_content(
            repo=repo,
            commit_id=commit_id,
            filepath=filepath,
            pat=pat,
        )

    def _decode_content(self, payload: dict[str, Any], repo: str, filepath: str) -> str:
        if payload.get("type") != "file":
            raise GitHubPorterError(
                f"Expected a file payload for {repo}:{filepath}, got {payload.get('type')!r}"
            )

        if payload.get("encoding") != "base64":
            raise GitHubPorterError(
                f"Expected base64 content for {repo}:{filepath}, got "
                f"{payload.get('encoding')!r}"
            )

        encoded_content = payload.get("content")
        if not isinstance(encoded_content, str) or not encoded_content.strip():
            raise GitHubPorterError(f"Missing content for {repo}:{filepath}")

        normalized_content = encoded_content.replace("\n", "")
        decoded_bytes = base64.b64decode(normalized_content, validate=True)
        return decoded_bytes.decode("utf-8")

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
