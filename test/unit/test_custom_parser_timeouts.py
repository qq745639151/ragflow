import httpx
import pytest

from rag.app import custom_parser


class _TimeoutClient:
    def __init__(self, exc, **kwargs):
        self.exc = exc
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, *args, **kwargs):
        raise self.exc


def test_xidian_connect_timeout_message_uses_human_readable_default(monkeypatch):
    recorded = {}

    def _client_factory(**kwargs):
        recorded.update(kwargs)
        return _TimeoutClient(httpx.ConnectTimeout("boom"), **kwargs)

    monkeypatch.setattr(custom_parser.httpx, "Client", _client_factory)

    with pytest.raises(RuntimeError, match="connect timeout after 30s"):
        custom_parser._fetch_xidian_payload(  # noqa: SLF001
            "sample.pdf",
            b"pdf-bytes",
            parser_config={},
        )

    assert recorded["timeout"] is not None
    assert recorded["trust_env"] is False
    assert recorded["timeout"].connect == 30.0
    assert recorded["timeout"].write == 300.0
    assert recorded["timeout"].read == 3600.0


def test_xidian_read_timeout_message_reports_disabled_when_timeout_value_is_blank(monkeypatch):
    recorded = {}

    def _client_factory(**kwargs):
        recorded.update(kwargs)
        return _TimeoutClient(httpx.ReadTimeout("boom"), **kwargs)

    monkeypatch.setattr(custom_parser.httpx, "Client", _client_factory)

    with pytest.raises(RuntimeError, match="read timeout after disabled"):
        custom_parser._fetch_xidian_payload(  # noqa: SLF001
            "sample.pdf",
            b"pdf-bytes",
            parser_config={
                "xidian_enable_timeout": "true",
                "xidian_timeout": "",
                "xidian_connect_timeout": "",
                "xidian_write_timeout": "",
            },
        )

    assert recorded["timeout"] is not None


def test_xidian_timeouts_can_be_overridden(monkeypatch):
    recorded = {}

    def _client_factory(**kwargs):
        recorded.update(kwargs)
        return _TimeoutClient(httpx.ConnectTimeout("boom"), **kwargs)

    monkeypatch.setattr(custom_parser.httpx, "Client", _client_factory)

    with pytest.raises(RuntimeError, match="connect timeout after 45s"):
        custom_parser._fetch_xidian_payload(  # noqa: SLF001
            "sample.pdf",
            b"pdf-bytes",
            parser_config={
                "xidian_connect_timeout": "45",
                "xidian_write_timeout": "600",
                "xidian_timeout": "7200",
            },
        )

    assert recorded["timeout"] is not None
    assert recorded["timeout"].connect == 45.0
    assert recorded["timeout"].write == 600.0
    assert recorded["timeout"].read == 7200.0
