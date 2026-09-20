"""ElevenLabs client. Port of elevenPost() from the video project's lib.mjs."""
import httpx

from ..env import require_env

BASE = "https://api.elevenlabs.io/v1"


def eleven_post(path: str, body: dict, client: httpx.Client, params: dict | None = None) -> httpx.Response:
    return client.post(f"{BASE}{path}", json=body, params=params or {},
                       headers={"xi-api-key": require_env("ELEVENLABS_API_KEY")})
