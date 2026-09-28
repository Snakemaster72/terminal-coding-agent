from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from tools.base import Tool, ToolInvocation, ToolKind, ToolResult


class WebFetchParams(BaseModel):
    url: str = Field(
        ...,
        description="URL to fetch content from. Must be a valid URL starting with http:// or https://",
    )
    timeout: int = Field(
        30, ge=5, le=120, description="Timeout in seconds for the request (default: 30)"
    )


class WebFetchTool(Tool):
    name = "web_fetch"
    description = "Fetch content from the web. Returns the full text of the page."
    kind = ToolKind.NETWORK
    schema = WebFetchParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:

        params = WebFetchParams(**invocation.params)

        parsed = urlparse(params.url)  # Validate URL
        if not parsed.scheme or not parsed.scheme in ["http", "https"]:
            return ToolResult.error_result(
                f"Invalid URL '{params.url}'. Must start with http:// or https://"
            )

        # Fetch content from the web
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(params.timeout),
                follow_redirects=True,
            ) as client:
                response = await client.get(params.url)  # coroutine
                response.raise_for_status()  # if any http error occurs, raise an exception
                text = response.text
        except httpx.HTTPStatusError as e:
            return ToolResult.error_result(
                f"HTTP error while fetching '{params.url}': {e.response.status_code} {e.response.reason_phrase}"
            )
        except Exception as e:
            return ToolResult.error_result(f"Error fetching '{params.url}': {e!s}")

        if len(text) > 10000:
            text = text[:10000] + "\n\n[Content truncated due to length]"

        return ToolResult.success_result(
            text,
            metadata={
                "status_code": response.status_code,
                "content_length": len(response.content),
            },
        )
