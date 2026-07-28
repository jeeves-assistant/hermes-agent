"""Regression coverage for profile-isolated multiplex channel directories."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import Platform
from gateway.channel_directory import (
    build_channel_directories,
    format_directory_for_display,
    load_directory,
    resolve_channel_name,
)
from tools.send_message_tool import send_message_tool


@pytest.fixture
def beta_session():
    """Stamp the profile through ContextVars, even after another test engaged them."""
    from gateway.session_context import _SESSION_PROFILE

    token = _SESSION_PROFILE.set("beta")
    try:
        yield
    finally:
        _SESSION_PROFILE.reset(token)


class _RecordingAdapter:
    def __init__(self):
        self.sent = []
        self.reactions = []

    async def send(self, *, chat_id, content, metadata=None):
        self.sent.append((chat_id, content, metadata))
        return SimpleNamespace(success=True, message_id="sent-1", error=None)

    async def add_reaction(self, chat_id, emoji, message_id=None):
        self.reactions.append((chat_id, emoji, message_id))
        return {"success": True}


def _runner(default_adapter, beta_adapter, *, active_profile="default"):
    adapters = {Platform("photon"): default_adapter}
    profile_adapters = {"beta": {Platform("photon"): beta_adapter}}

    def resolve(platform, profile=None):
        if profile == active_profile:
            return adapters.get(platform)
        if profile in profile_adapters:
            return profile_adapters[profile].get(platform)
        return None

    return SimpleNamespace(
        adapters=adapters,
        _profile_adapters=profile_adapters,
        _authorization_adapter=resolve,
        _active_profile_name=lambda: active_profile,
        config=SimpleNamespace(multiplex_profiles=True),
    )


def _config():
    pconfig = SimpleNamespace(enabled=True, token="test-only", extra={})
    return SimpleNamespace(
        platforms={Platform("photon"): pconfig},
        get_home_channel=lambda _platform: None,
    )


def test_profile_directories_build_and_read_without_cross_profile_metadata(tmp_path, monkeypatch):
    """Same-name targets must remain owned by the profile that discovered them."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    default_adapter = object()
    beta_adapter = object()

    def fake_sessions(platform_name, profile_home=None):
        assert platform_name == "telegram"
        channel_id = "111" if profile_home == tmp_path else "222"
        return [{"id": channel_id, "name": "general", "type": "group"}]

    with patch("gateway.channel_directory._build_from_sessions", side_effect=fake_sessions):
        asyncio.run(
            build_channel_directories(
                {Platform.TELEGRAM: default_adapter},
                profile_adapters={"beta": {Platform.TELEGRAM: beta_adapter}},
                active_profile="default",
                multiplex=True,
            )
        )

    default_dir = load_directory(profile_name="default")
    beta_dir = load_directory(profile_name="beta")
    assert default_dir["profile"] == "default"
    assert beta_dir["profile"] == "beta"
    assert default_dir["platforms"]["telegram"][0]["id"] == "111"
    assert beta_dir["platforms"]["telegram"][0]["id"] == "222"
    assert resolve_channel_name("telegram", "general", profile_name="default") == "111"
    assert resolve_channel_name("telegram", "general", profile_name="beta") == "222"

    default_listing = format_directory_for_display(profile_name="default")
    beta_listing = format_directory_for_display(profile_name="beta")
    assert "111" not in beta_listing
    assert '"111"' not in json.dumps(beta_dir)
    assert "222" not in default_listing
    assert '"222"' not in json.dumps(default_dir)


def test_profile_load_fails_closed_on_legacy_unowned_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "channel_directory.json").write_text(json.dumps({
        "updated_at": "old",
        "platforms": {"photon": [{"id": "default-secret", "name": "general", "type": "group"}]},
    }))

    assert load_directory(profile_name="default")["platforms"] == {}
    assert resolve_channel_name("photon", "general", profile_name="default") is None
    assert "default-secret" not in format_directory_for_display(profile_name="default")


@pytest.mark.parametrize("active_profile", ["", "../default"])
def test_multiplex_send_message_list_fails_closed_without_trustworthy_owner(
    monkeypatch, active_profile
):
    from gateway.session_context import _SESSION_PROFILE

    runner = _runner(
        _RecordingAdapter(), _RecordingAdapter(), active_profile=active_profile
    )
    token = _SESSION_PROFILE.set("")
    try:
        with patch("gateway.run._gateway_runner_ref", lambda: runner):
            result = json.loads(send_message_tool({"action": "list"}))
    finally:
        _SESSION_PROFILE.reset(token)

    assert "error" in result
    assert "no trustworthy profile owner" in result["error"]


def test_invalid_profile_stamp_fails_closed():
    from gateway.session_context import _SESSION_PROFILE

    token = _SESSION_PROFILE.set("../default")
    try:
        result = json.loads(send_message_tool({"action": "list"}))
    finally:
        _SESSION_PROFILE.reset(token)

    assert "error" in result
    assert "Invalid multiplex profile ownership stamp" in result["error"]


@pytest.mark.parametrize("profile_name", ["../default", "/tmp/outside", "bad/name"])
def test_directory_api_rejects_out_of_root_profile_names(profile_name):
    with pytest.raises(ValueError):
        load_directory(profile_name)


def test_valid_but_unserved_profile_stamp_fails_closed():
    from gateway.session_context import _SESSION_PROFILE

    runner = _runner(_RecordingAdapter(), _RecordingAdapter())
    token = _SESSION_PROFILE.set("alpha")
    try:
        with patch("gateway.run._gateway_runner_ref", lambda: runner):
            result = json.loads(send_message_tool({"action": "list"}))
    finally:
        _SESSION_PROFILE.reset(token)

    assert "error" in result
    assert "not served by this multiplex gateway" in result["error"]


@pytest.mark.parametrize("active_profile", ["default", "alpha"])
def test_unstamped_primary_session_uses_active_profile_directory(
    tmp_path, monkeypatch, active_profile
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    default_adapter = _RecordingAdapter()
    beta_adapter = _RecordingAdapter()
    runner = _runner(
        default_adapter, beta_adapter, active_profile=active_profile
    )
    owner_home = tmp_path if active_profile == "default" else tmp_path / "profiles" / active_profile
    owner_home.mkdir(parents=True, exist_ok=True)
    (owner_home / "channel_directory.json").write_text(json.dumps({
        "profile": active_profile,
        "updated_at": "now",
        "platforms": {"photon": [{
            "id": "+19990000003", "name": "primary", "type": "group"
        }]},
    }))

    from gateway.session_context import _SESSION_PROFILE
    token = _SESSION_PROFILE.set("")
    try:
        with patch("gateway.run._gateway_runner_ref", lambda: runner), \
             patch("gateway.config.load_gateway_config", return_value=_config()), \
             patch("tools.interrupt.is_interrupted", return_value=False), \
             patch("gateway.mirror.mirror_to_session", return_value=False):
            listed = json.loads(send_message_tool({"action": "list"}))["targets"]
            result = json.loads(send_message_tool({
                "action": "send", "target": "photon:primary", "message": "hello"
            }))
    finally:
        _SESSION_PROFILE.reset(token)

    assert "primary" in listed
    assert result.get("success") is True, result
    assert default_adapter.sent == [("+19990000003", "hello", None)]
    assert beta_adapter.sent == []


def test_send_message_list_resolve_and_adapter_are_profile_isolated(
    tmp_path, monkeypatch, beta_session
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    default_adapter = _RecordingAdapter()
    beta_adapter = _RecordingAdapter()
    runner = _runner(default_adapter, beta_adapter)

    beta_home = tmp_path / "profiles" / "beta"
    beta_home.mkdir(parents=True)
    (tmp_path / "channel_directory.json").write_text(json.dumps({
        "profile": "default",
        "updated_at": "now",
        "platforms": {"photon": [
            {"id": "+19990000001", "name": "general", "type": "group"},
            {"id": "+19990000011", "name": "default-private", "type": "group"},
        ]},
    }))
    (beta_home / "channel_directory.json").write_text(json.dumps({
        "profile": "beta",
        "updated_at": "now",
        "platforms": {"photon": [
            {"id": "+19990000002", "name": "general", "type": "group"},
            {"id": "+19990000022", "name": "beta-private", "type": "group"},
        ]},
    }))

    with patch("gateway.run._gateway_runner_ref", lambda: runner), \
         patch("gateway.config.load_gateway_config", return_value=_config()), \
         patch("tools.interrupt.is_interrupted", return_value=False), \
         patch("gateway.mirror.mirror_to_session", return_value=False):
        listed = json.loads(send_message_tool({"action": "list"}))["targets"]
        result = json.loads(send_message_tool({
            "action": "send", "target": "photon:general", "message": "hello"
        }))

    assert "general" in listed
    assert "beta-private" in listed
    assert "default-private" not in listed
    assert result.get("success") is True, result
    assert beta_adapter.sent == [("+19990000002", "hello", None)]
    assert default_adapter.sent == []


def test_reaction_resolution_and_adapter_are_profile_isolated(
    tmp_path, monkeypatch, beta_session
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    default_adapter = _RecordingAdapter()
    beta_adapter = _RecordingAdapter()
    runner = _runner(default_adapter, beta_adapter)
    beta_home = tmp_path / "profiles" / "beta"
    beta_home.mkdir(parents=True)
    (beta_home / "channel_directory.json").write_text(json.dumps({
        "profile": "beta",
        "updated_at": "now",
        "platforms": {"photon": [{"id": "+19990000002", "name": "general", "type": "group"}]},
    }))

    with patch("gateway.run._gateway_runner_ref", lambda: runner):
        result = json.loads(send_message_tool({
            "action": "react",
            "target": "photon:general",
            "emoji": "👍",
            "message_id": "msg-1",
        }))

    assert result.get("success") is True, result
    assert beta_adapter.reactions == [("+19990000002", "👍", "msg-1")]
    assert default_adapter.reactions == []


def test_opaque_photon_reaction_uses_stamped_profile_adapter(beta_session):
    default_adapter = _RecordingAdapter()
    beta_adapter = _RecordingAdapter()
    runner = _runner(default_adapter, beta_adapter)

    with patch("gateway.run._gateway_runner_ref", lambda: runner):
        result = json.loads(send_message_tool({
            "action": "react",
            "target": "photon:any;-;+19990000002",
            "emoji": "👍",
            "message_id": "msg-opaque",
        }))

    assert result.get("success") is True, result
    assert beta_adapter.reactions == [
        ("any;-;+19990000002", "👍", "msg-opaque")
    ]
    assert default_adapter.reactions == []


@pytest.mark.parametrize(
    "target",
    ["name;-;x y", "foo;-;bar", "a;-;a;extra", "any;-;+123", "any;+;"],
)
def test_malformed_photon_guid_is_not_treated_as_explicit(target):
    from tools.send_message_tool import _parse_target_ref

    _chat_id, _thread_id, is_explicit = _parse_target_ref("photon", target)
    assert is_explicit is False


def test_cron_name_resolution_uses_job_profile_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    beta_home = tmp_path / "profiles" / "beta"
    beta_home.mkdir(parents=True)
    (tmp_path / "channel_directory.json").write_text(json.dumps({
        "profile": "default",
        "updated_at": "now",
        "platforms": {"photon": [{
            "id": "+19990000031", "name": "general", "type": "group"
        }]},
    }))
    (beta_home / "channel_directory.json").write_text(json.dumps({
        "profile": "beta",
        "updated_at": "now",
        "platforms": {"photon": [{
            "id": "+19990000032", "name": "general", "type": "group"
        }]},
    }))

    from cron.scheduler import _resolve_single_delivery_target

    default_target = _resolve_single_delivery_target(
        {}, "photon:general", profile_name="default"
    )
    beta_target = _resolve_single_delivery_target(
        {}, "photon:general", profile_name="beta"
    )

    assert default_target is not None
    assert beta_target is not None
    assert default_target["chat_id"] == "+19990000031"
    assert beta_target["chat_id"] == "+19990000032"


def test_single_profile_cron_preserves_unowned_legacy_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    directory_path = tmp_path / "channel_directory.json"
    monkeypatch.setattr(
        "gateway.channel_directory.DIRECTORY_PATH", directory_path
    )
    directory_path.write_text(json.dumps({
        "updated_at": "now",
        "platforms": {"photon": [{
            "id": "+19990000041", "name": "general", "type": "group"
        }]},
    }))
    from cron.scheduler import _resolve_delivery_target

    target = _resolve_delivery_target({"deliver": "photon:general"})

    assert target is not None
    assert target["chat_id"] == "+19990000041"


def test_multiplex_cron_uses_trusted_profile_and_matching_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    beta_home = tmp_path / "profiles" / "beta"
    beta_home.mkdir(parents=True)
    (beta_home / "channel_directory.json").write_text(json.dumps({
        "profile": "beta",
        "updated_at": "now",
        "platforms": {"photon": [{
            "id": "+19990000042", "name": "general", "type": "group"
        }]},
    }))
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from cron.scheduler import _resolve_delivery_target
    from gateway.session_context import reset_cron_profile, set_cron_profile
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    prior = is_multiplex_active()
    set_multiplex_active(True)
    profile_token = set_cron_profile("beta")
    home_token = set_hermes_home_override(str(beta_home))
    try:
        target = _resolve_delivery_target({"deliver": "photon:general"})
    finally:
        reset_hermes_home_override(home_token)
        reset_cron_profile(profile_token)
        set_multiplex_active(prior)

    assert target is not None
    assert target["chat_id"] == "+19990000042"


def test_multiplex_cron_rejects_profile_home_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from agent.secret_scope import is_multiplex_active, set_multiplex_active
    from cron.scheduler import _resolve_delivery_target
    from gateway.session_context import reset_cron_profile, set_cron_profile

    prior = is_multiplex_active()
    set_multiplex_active(True)
    token = set_cron_profile("beta")
    try:
        with pytest.raises(RuntimeError, match="does not match scoped Hermes home"):
            _resolve_delivery_target({"deliver": "photon:general"})
    finally:
        reset_cron_profile(token)
        set_multiplex_active(prior)


def test_housekeeping_refreshes_secondary_only_multiplex(monkeypatch):
    from gateway import run as gateway_run

    class FiveTicks:
        def __init__(self):
            self.waits = 0

        def is_set(self):
            return self.waits >= 5

        def wait(self, timeout=None):
            self.waits += 1

    scheduled = []

    def fake_schedule(coro, loop, **kwargs):
        scheduled.append(coro)
        coro.close()
        return SimpleNamespace(result=lambda timeout=None: None)

    monkeypatch.setattr(gateway_run, "safe_schedule_threadsafe", fake_schedule)
    gateway_run._start_gateway_housekeeping(
        FiveTicks(),
        adapters={},
        loop=object(),
        interval=0,
        profile_adapters={"beta": {Platform("photon"): _RecordingAdapter()}},
        active_profile="default",
        multiplex=True,
    )

    assert len(scheduled) == 1


def test_multiplex_cron_provider_binds_profile_owner(tmp_path, monkeypatch):
    from cron.scheduler_provider import InProcessCronScheduler
    from gateway.session_context import get_session_env

    beta_home = tmp_path / "profiles" / "beta"
    beta_home.mkdir(parents=True)
    observed = []
    monkeypatch.setattr(
        "cron.scheduler.tick",
        lambda **_kwargs: observed.append(
            get_session_env("HERMES_CRON_PROFILE", "")
        ),
    )

    class _OneTick:
        stopped = False

        def is_set(self):
            return self.stopped

        def wait(self, _interval):
            self.stopped = True

    InProcessCronScheduler()._start_multiplex(
        _OneTick(), profile_homes=[("beta", beta_home)], interval=0
    )

    assert observed == ["beta"]
    assert get_session_env("HERMES_CRON_PROFILE", "") == ""