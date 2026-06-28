"""Claude LLM Client with mock fallback for development/testing."""
import logging
from config.settings import settings

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Anthropic Claude API client with graceful degradation."""

    def __init__(self):
        self.client = None
        self.model = "claude-3-sonnet-20240229"
        self._init_client()

    def _init_client(self):
        try:
            import anthropic
            key = settings.claude_api_key
            if key and not key.startswith("sk-ant-your"):
                self.client = anthropic.Anthropic(api_key=key)
                logger.info("Claude client initialized successfully")
            else:
                logger.warning("No valid Claude API key – running in mock mode")
        except Exception as e:
            logger.warning(f"Claude client init failed: {e} – running in mock mode")

    def generate_response(
        self,
        messages: list[dict],
        system_prompt: str = "",
        max_tokens: int = 2048,
    ) -> str:
        """Generate a response, falling back to mock if no real client."""
        if self.client:
            return self._call_api(messages, system_prompt, max_tokens)
        return self._mock_response(messages)

    def _call_api(self, messages: list[dict], system_prompt: str, max_tokens: int) -> str:
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=messages,
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise

    def _mock_response(self, messages: list[dict]) -> str:
        """Return a structured mock response for testing."""
        last = messages[-1].get("content", "") if messages else ""
        return (
            f"[MOCK RESPONSE] Based on the provided context, here is a grounded answer "
            f"to your query: '{last[:80]}...' — Please configure a valid CLAUDE_API_KEY "
            f"in your .env file to enable real LLM responses."
        )

    @property
    def is_mock(self) -> bool:
        return self.client is None


# Singleton instance
claude_client = ClaudeClient()
