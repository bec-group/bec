import os
from unittest import mock

import yaml

import bec_client_lib.core
from bec_client_lib.core import BECMessage
from bec_client_lib.core.tests.utils import ConnectorMock, create_session_from_config
from device_server.devices.devicemanager import DeviceConfigError, DeviceManagerDS

dir_path = os.path.dirname(bec_client_lib.core.__file__)


def test_request_response():
    connector = ConnectorMock("")
    device_manager = DeviceManagerDS(connector)

    def get_config_from_mock():
        with open(f"{dir_path}/tests/test_config.yaml", "r") as session_file:
            device_manager._session = create_session_from_config(yaml.safe_load(session_file))
        device_manager._load_session()

    def mocked_failed_connection(obj):
        if obj.name == "samx":
            raise ConnectionError

    config_reply = BECMessage.RequestResponseMessage(accepted=True, message="")
    with mock.patch.object(device_manager, "connect_device", wraps=mocked_failed_connection):
        with mock.patch.object(device_manager, "_get_config", get_config_from_mock):
            with mock.patch.object(
                device_manager.config_helper,
                "wait_for_config_reply",
                return_value=config_reply,
            ):
                device_manager.initialize("")
                with mock.patch.object(
                    device_manager.config_update_handler, "send_config_request_reply"
                ) as request_reply:
                    device_manager.config_update_handler.parse_config_request(
                        msg=BECMessage.DeviceConfigMessage(
                            action="update", config={"something": "something"}
                        )
                    )
                    request_reply.assert_called_once()
