import os

import yaml
from bec_utils.connector import ConnectorBase
from koss.devicemanager import DeviceManagerKOSS
from koss.koss import KOSS
from koss.scan_assembler import ScanAssembler
from koss.scan_worker import InstructionQueueStatus

dir_path = os.path.dirname(os.path.realpath(__file__))


def load_KossMock():
    connector = ConnectorMock("")
    device_manager = DeviceManagerKOSS(connector, "")
    with open(f"{dir_path}/test_session.yaml", "r") as f:
        device_manager._session = yaml.safe_load(f)
    device_manager._load_session()
    return KossMock(device_manager, connector)


class ConsumerMock:
    def start(self):
        pass


class ProducerMock:
    message_sent = {}

    def set(self, topic, msg):
        pass

    def send(self, queue, msg):
        self.message_sent = {"queue": queue, "msg": msg}


class ConnectorMock(ConnectorBase):
    def consumer(self, *args, **kwargs) -> ConsumerMock:
        return ConsumerMock()

    def producer(self, *args, **kwargs):
        return ProducerMock()


class WorkerMock:
    def __init__(self) -> None:
        self.scan_id = None
        self.scan_motors = []
        self.current_scanID = None
        self.current_scan_info = None
        self.status = InstructionQueueStatus.IDLE


class KossMock(KOSS):
    def __init__(self, dm, connector) -> None:
        self.dm = dm
        super().__init__(bootstrap_server="dummy", connector_cls=ConnectorMock, scibec_url="dummy")
        self.scan_worker = WorkerMock()

    def _start_devicemanager(self):
        pass

    def _start_scan_server(self):
        pass

    def shutdown(self):
        pass
