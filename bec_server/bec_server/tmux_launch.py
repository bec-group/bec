import os
import sys
import libtmux


def activate_venv(pane, service_name, service_path):
    """
    Activate the python environment for a service.
    """

    # check if the current file was install with pip install -e (editable mode)
    # if so, the venv is the service directory and it's called <service_name>_venv
    # otherwise, we simply take the currently running venv

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    if "site-packages" in __file__:
        venv_base_path = os.path.dirname(
            os.path.dirname(os.path.dirname(__file__.split("site-packages", maxsplit=1)[0]))
        )
        pane.send_keys(f"source {venv_base_path}/bin/activate")
        return
    if os.path.exists(f"{service_path}/{service_name}_venv"):
        pane.send_keys(f"source {service_path}/{service_name}_venv/bin/activate")
        return
    if os.path.exists(f"{base_dir}/bec_venv"):
        pane.send_keys(f"source {base_dir}/bec_venv/bin/activate")
        return
    # python_dir = sys.executable
    # base_dir = os.path.dirname(python_dir)
    # pane.send_keys(f"source {base_dir}/activate")

    # raise RuntimeError(f"Could not find a virtual environment for {service_name}. ")


def tmux_start(bec_path: str, config_path: str, services: dict):
    """
    Launch the BEC server in a tmux session. All services are launched in separate panes.

    Args:
        bec_path (str): Path to the BEC source code
        config (str): Path to the config file
        services (dict): Dictionary of services to launch. Keys are the service names, values are path and command templates.

    """
    tmux_server = libtmux.Server()
    session = tmux_server.new_session(
        "bec", window_name="BEC server. Use `ctrl+b d` to detach.", kill_session=True
    )

    # create panes and run commands
    panes = []
    for ii, service_info in enumerate(services.items()):
        service, service_config = service_info

        if ii == 0:
            pane = session.attached_window.attached_pane
        else:
            pane = session.attached_window.split_window(vertical=False)
        panes.append(pane)

        activate_venv(
            pane,
            service_name=service,
            service_path=service_config["path"].substitute(base_path=bec_path),
        )

        if config_path:
            pane.send_keys(f"{service_config['command']} --config {config_path}")
        else:
            pane.send_keys(f"{service_config['command']}")
        session.attached_window.select_layout("tiled")

    session.mouse_all_flag = True
    session.set_option("mouse", "on")


def tmux_stop():
    """
    Stop the BEC server.
    """
    tmux_server = libtmux.Server()
    avail_sessions = tmux_server.sessions.filter(session_name="bec")
    if len(avail_sessions) != 0:
        avail_sessions[0].kill_session()
