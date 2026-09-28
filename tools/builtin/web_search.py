from ddgs import DDGS
from pydantic import BaseModel, Field

from tools.base import Tool, ToolInvocation, ToolKind, ToolResult


class WebSearchParams(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(
        10, ge=1, le=20, description="Number of search results to return"
    )


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web for information. Returns search results with titles and snippets. Supports ** for recursive search."
    kind = ToolKind.NETWORK
    schema = WebSearchParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:

        params = WebSearchParams(**invocation.params)

        # duckduck go search

        try:
            results = DDGS().text(
                params.query,
                region="us",
                safesearch="off",
                timelimit="y",
                page=1,
                max_results=params.max_results,
            )
        except Exception as e:
            return ToolResult.error_result(f"Error performing web search: {e!s}")

        if not results:
            return ToolResult.success_result(
                f"No results found for query '{params.query}'.",
                metadata={
                    "query": params.query,
                    "results": 0,
                },
            )

        output_lines = [f'Search results for query: "{params.query}"']

        for i, result in enumerate(results, start=1):
            output_lines.append(f"{i}. {result['title']}")
            output_lines.append(f"   URL: {result['href']}")
            if result.get("body"):
                output_lines.append(f"   Snippet: {result['body']}")

            output_lines.append("")

        return ToolResult.success_result(
            "\n".join(output_lines),
            metadata={
                "query": params.query,
                "results": len(results),
            },
        )
