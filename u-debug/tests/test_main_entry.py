"""Tests for the __main__ module and app.main entrypoint."""

from unittest.mock import patch, MagicMock

from debug.app import main


@patch("debug.app.create_app")
@patch("debug.app.load_config")
def test_main_starts_server(mock_config, mock_create):
    """main() loads config, creates app, and starts the server."""
    mock_app = MagicMock()
    mock_create.return_value = mock_app

    main()
    mock_config.assert_called_once()
    mock_create.assert_called_once()
    mock_app.run.assert_called_once_with(
        host="0.0.0.0", port=5001, debug=False, threaded=True
    )
