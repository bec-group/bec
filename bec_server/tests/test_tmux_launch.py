from string import Template
from unittest import mock

from bec_server.tmux_launch import tmux_start, tmux_stop


def test_tmux_start():
    with mock.patch("bec_server.tmux_launch.libtmux") as mock_libtmux_server:
        tmux_start(
            "/path/to/bec",
            "/path/to/config",
            {
                "scan_server": {
                    "path": Template("$base_path/scan_server"),
                    "command": "bec-scan-server",
                },
                "scan_bundler": {
                    "path": Template("$base_path/scan_bundler"),
                    "command": "bec-scan-bundler",
                },
            },
        )
        mock_libtmux_server.Server().new_session.assert_called_once_with(
            "bec", window_name="BEC server. Use `ctrl+b d` to detach.", kill_session=True
        )
        assert (
            mock_libtmux_server.Server().new_session().attached_window.select_layout.call_count == 2
        )

        assert mock_libtmux_server.Server().new_session().set_option.call_count == 1
        assert mock_libtmux_server.Server().new_session().set_option.call_args[0][0] == "mouse"
        assert mock_libtmux_server.Server().new_session().set_option.call_args[0][1] == "on"


def test_tmux_stop_without_sessions():
    with mock.patch("bec_server.tmux_launch.libtmux") as mock_libtmux_server:
        mock_libtmux_server.Server().sessions.filter.return_value = []
        tmux_stop()
        mock_libtmux_server.Server().kill_server.assert_not_called()


def test_tmux_stop_with_sessions():
    session = mock.MagicMock()
    with mock.patch("bec_server.tmux_launch.libtmux") as mock_libtmux_server:
        mock_libtmux_server.Server().sessions.filter.return_value = [session]
        tmux_stop()
        session.kill_session.assert_called_once()
