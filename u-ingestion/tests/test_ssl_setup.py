"""Tests for SSL setup."""

from types import SimpleNamespace

from ingestion.utils import ssl_setup


def test_setup_ssl_with_rbc_security(monkeypatch):
    """Enable corporate certificates when the helper is available."""
    module = SimpleNamespace(called=False)

    def enable_certs():
        module.called = True

    module.enable_certs = enable_certs
    monkeypatch.setattr(
        ssl_setup.importlib,
        "import_module",
        lambda name: module,
    )

    ssl_setup.setup_ssl()

    assert module.called is True


def test_setup_ssl_without_rbc_security(monkeypatch):
    """Fall back to system certificates when the helper is missing."""
    monkeypatch.setattr(
        ssl_setup.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError("missing")),
    )

    ssl_setup.setup_ssl()
