import os

import bec_lib.core
import yaml
from bec_lib.core import DeviceManagerBase as DeviceManager
from bec_lib.core import MessageEndpoints, ServiceConfig
from bec_lib.core.tests.utils import ConnectorMock, create_session_from_config

from scan_server.scan_assembler import ScanAssembler
from scan_server.scan_queue import (
    InstructionQueueItem,
    RequestBlock,
    RequestBlockQueue,
    ScanQueue,
)
from scan_server.scan_server import ScanServer
from scan_server.scan_worker import InstructionQueueStatus
from scan_server.scans import RequestBase

# pylint: disable=missing-function-docstring
# pylint: disable=protected-access

# dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.dirname(bec_lib.core.__file__)


def load_ScanServerMock():
    connector = ConnectorMock("")
    device_manager = DeviceManager(connector, "")
    device_manager.producer = connector.producer()
    with open(f"{dir_path}/tests/test_config.yaml", "r") as f:
        device_manager._session = create_session_from_config(yaml.safe_load(f))
    device_manager._load_session()
    return ScanServerMock(device_manager, connector)


class WorkerMock:
    def __init__(self) -> None:
        self.scan_id = None
        self.scan_motors = []
        self.current_scanID = None
        self.current_scan_info = None
        self.status = InstructionQueueStatus.IDLE
        self.current_instruction_queue_item = None


class ScanServerMock(ScanServer):
    def __init__(self, device_manager, connector) -> None:
        self.device_manager = device_manager
        super().__init__(
            ServiceConfig(redis={"host": "dummy", "port": 6379}), connector_cls=ConnectorMock
        )
        self.scan_worker = WorkerMock()

    def _start_metrics_emitter(self):
        pass

    def _start_update_service_info(self):
        pass

    def _start_device_manager(self):
        pass

    def shutdown(self):
        pass

    @property
    def scan_number(self) -> int:
        """get the current scan number"""
        return 2

    @scan_number.setter
    def scan_number(self, val: int):
        pass

    @property
    def dataset_number(self) -> int:
        """get the current dataset number"""
        return 3

    @dataset_number.setter
    def dataset_number(self, val: int):
        pass
