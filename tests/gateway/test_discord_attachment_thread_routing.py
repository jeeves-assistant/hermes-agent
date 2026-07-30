"""Regression tests for Discord attachment ``metadata.thread_id`` routing.

Covers the attachment paths used by local image, video, document, remote
image, animation, and multi-image sends.  Thread metadata must select the
thread rather than the parent channel, matching Discord text delivery.
"""

import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig


def _ensure_discord_mock() -> None:
    """Install a minimal discord module only when discord.py is unavailable."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_adapter_module  # noqa: E402
from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402

pytestmark = pytest.mark.asyncio


@pytest.fixture
def adapter():
    """Create an adapter whose parent and thread channels are independently observable."""
    config = PlatformConfig(enabled=True, token="fake-token")
    instance = DiscordAdapter(config)

    parent = MagicMock(name="parent_channel")
    parent.send = AsyncMock(
        return_value=SimpleNamespace(id=101, attachments=[object()])
    )
    thread = MagicMock(name="thread_channel")
    thread.send = AsyncMock(
        return_value=SimpleNamespace(id=201, attachments=[object()])
    )

    client = MagicMock()
    client.user = SimpleNamespace(id=999)
    client.get_channel = MagicMock(
        side_effect=lambda channel_id: {100: parent, 200: thread}.get(channel_id)
    )
    client.fetch_channel = AsyncMock(return_value=None)
    instance._client = client
    instance._is_forum_parent = MagicMock(return_value=False)
    return instance, parent, thread


class TestMediaMethodsMetadataForwarding:
    async def test_send_image_file_forwards_metadata(self, adapter):
        instance, _, _ = adapter
        instance._send_file_attachment = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="123")
        )

        await instance.send_image_file(
            chat_id="100",
            image_path="/tmp/img.png",
            caption="look",
            metadata={"thread_id": "200"},
        )

        instance._send_file_attachment.assert_awaited_once_with(
            "100", "/tmp/img.png", "look", metadata={"thread_id": "200"}
        )

    async def test_send_video_forwards_metadata(self, adapter):
        instance, _, _ = adapter
        instance._send_file_attachment = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="123")
        )

        await instance.send_video(
            chat_id="100",
            video_path="/tmp/vid.mp4",
            caption="watch",
            metadata={"thread_id": "200"},
        )

        instance._send_file_attachment.assert_awaited_once_with(
            "100", "/tmp/vid.mp4", "watch", metadata={"thread_id": "200"}
        )

    async def test_send_document_forwards_metadata(self, adapter):
        instance, _, _ = adapter
        instance._send_file_attachment = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="123")
        )

        await instance.send_document(
            chat_id="100",
            file_path="/tmp/doc.pdf",
            caption="read this",
            file_name="doc.pdf",
            metadata={"thread_id": "200"},
        )

        instance._send_file_attachment.assert_awaited_once_with(
            "100",
            "/tmp/doc.pdf",
            "read this",
            file_name="doc.pdf",
            metadata={"thread_id": "200"},
        )

    async def test_send_image_file_no_metadata_forwards_none(self, adapter):
        instance, _, _ = adapter
        instance._send_file_attachment = AsyncMock(
            return_value=SimpleNamespace(success=True, message_id="123")
        )

        await instance.send_image_file(chat_id="100", image_path="/tmp/img.png")

        instance._send_file_attachment.assert_awaited_once_with(
            "100", "/tmp/img.png", None, metadata=None
        )


class TestAttachmentDestinationRouting:
    async def test_local_file_routes_to_thread(self, adapter, monkeypatch, tmp_path):
        instance, parent, thread = adapter
        attachment = tmp_path / "attachment.txt"
        attachment.write_text("thread-bound")
        monkeypatch.setattr(discord_adapter_module.discord, "File", MagicMock())

        result = await instance._send_file_attachment(
            "100", str(attachment), metadata={"thread_id": "200"}
        )

        assert result.success is True
        instance._client.get_channel.assert_called_once_with(200)
        thread.send.assert_awaited_once()
        parent.send.assert_not_awaited()

    async def test_local_file_without_metadata_routes_to_parent(
        self, adapter, monkeypatch, tmp_path
    ):
        instance, parent, thread = adapter
        attachment = tmp_path / "attachment.txt"
        attachment.write_text("parent-bound")
        monkeypatch.setattr(discord_adapter_module.discord, "File", MagicMock())

        result = await instance._send_file_attachment("100", str(attachment))

        assert result.success is True
        instance._client.get_channel.assert_called_once_with(100)
        parent.send.assert_awaited_once()
        thread.send.assert_not_awaited()

    @pytest.mark.parametrize(
        ("method_name", "url"),
        [
            ("send_image", "https://cdn.example.test/image.png"),
            ("send_animation", "https://cdn.example.test/animation.gif"),
        ],
    )
    async def test_remote_media_routes_to_thread(
        self, adapter, monkeypatch, method_name, url
    ):
        instance, parent, thread = adapter

        class FakeSession:
            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        fake_aiohttp = types.SimpleNamespace(
            ClientSession=lambda **kwargs: FakeSession(),
            ClientTimeout=lambda **kwargs: kwargs,
        )
        monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
        monkeypatch.setattr(discord_adapter_module, "is_safe_url", lambda _url: True)
        monkeypatch.setattr(
            discord_adapter_module,
            "_read_url_image_with_redirect_guard",
            AsyncMock(return_value=(200, b"image-bytes", {"content-type": "image/png"})),
        )
        monkeypatch.setattr(discord_adapter_module.discord, "File", MagicMock())

        result = await getattr(instance, method_name)(
            "100", url, metadata={"thread_id": "200"}
        )

        assert result.success is True
        instance._client.get_channel.assert_called_once_with(200)
        thread.send.assert_awaited_once()
        parent.send.assert_not_awaited()

    async def test_multi_image_routes_to_thread(
        self, adapter, monkeypatch, tmp_path
    ):
        instance, parent, thread = adapter
        image = tmp_path / "image.png"
        image.write_bytes(b"not-a-real-png")
        monkeypatch.setattr(discord_adapter_module.discord, "File", MagicMock())

        await instance.send_multiple_images(
            "100",
            [(image.as_uri(), "caption")],
            metadata={"thread_id": "200"},
        )

        instance._client.get_channel.assert_called_once_with(200)
        thread.send.assert_awaited_once()
        parent.send.assert_not_awaited()
