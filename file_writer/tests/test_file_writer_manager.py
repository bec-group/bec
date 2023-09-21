import os
from unittest import mock

import bec_lib.core
import numpy as np
import pytest
import yaml
from bec_lib.core import BECMessage, DeviceManagerBase, MessageEndpoints, ServiceConfig
from bec_lib.core.bec_errors import ServiceConfigError
from bec_lib.core.redis_connector import MessageObject
from bec_lib.core.tests.utils import ConnectorMock, create_session_from_config

from file_writer import FileWriterManager
from file_writer.file_writer import FileWriter
from file_writer.file_writer_manager import ScanStorage

# pylint: disable=missing-function-docstring
# pylint: disable=protected-access

dir_path = os.path.dirname(bec_lib.core.__file__)


def load_FileWriter():
    connector = ConnectorMock("")
    device_manager = DeviceManagerBase(connector, "")
    device_manager.producer = connector.producer()
    with open(f"{dir_path}/tests/test_config.yaml", "r") as session_file:
        device_manager._session = create_session_from_config(yaml.safe_load(session_file))
    device_manager._load_session()
    return FileWriterManagerMock(device_manager, connector)


class FileWriterManagerMock(FileWriterManager):
    def __init__(self, device_manager, connector) -> None:
        self.device_manager = device_manager
        config = ServiceConfig(
            redis={"host": "dummy", "port": 6379},
            config={"file_writer": {"plugin": "default_NeXus_format", "base_path": "./"}},
        )
        super().__init__(config=config, connector_cls=ConnectorMock)

    def _start_device_manager(self):
        pass

    def shutdown(self):
        pass


def test_scan_segment_callback():
    file_manager = load_FileWriter()
    msg = BECMessage.ScanMessage(
        point_id=1, scanID="scanID", data={"data": "data"}, metadata={"scan_number": 1}
    )
    msg_bundle = BECMessage.BundleMessage()
    msg_bundle.append(msg.dumps())
    msg_raw = MessageObject(value=msg_bundle.dumps(), topic="scan_segment")

    file_manager._scan_segment_callback(msg_raw, parent=file_manager)
    assert file_manager.scan_storage["scanID"].scan_segments[1] == {"data": "data"}


def test_scan_status_callback():
    file_manager = load_FileWriter()
    msg = BECMessage.ScanStatusMessage(
        scanID="scanID",
        status="closed",
        info={
            "scan_number": 1,
            "DIID": "DIID",
            "stream": "stream",
            "num_points": 1,
            "enforce_sync": True,
        },
    )
    msg_raw = MessageObject(value=msg.dumps(), topic="scan_status")

    file_manager._scan_status_callback(msg_raw, parent=file_manager)
    assert file_manager.scan_storage["scanID"].scan_finished is True


class MockWriter(FileWriter):
    def __init__(self, file_writer_manager):
        super().__init__(file_writer_manager)
        self.write_called = False

    def write(self, file_path: str, data):
        self.write_called = True


def test_write_file():
    file_manager = load_FileWriter()
    file_manager.scan_storage["scanID"] = ScanStorage(10, "scanID")
    with mock.patch.object(
        file_manager.writer_mixin, "compile_full_filename"
    ) as mock_create_file_path:
        mock_create_file_path.return_value = "path"
        # replace NexusFileWriter with MockWriter
        file_manager.file_writer = MockWriter(file_manager)
        file_manager.write_file("scanID")
        assert file_manager.file_writer.write_called is True


def test_create_file_path():
    file_manager = load_FileWriter()
    file_manager.file_writer_config["base_path"] = "./"
    file_path = file_manager.writer_mixin.compile_full_filename(10, "master.h5", create_dir=False)
    assert file_path == os.path.abspath("./data/S00000-00999/S00010/S00010_master.h5")


def test_write_file_raises_alarm_on_error():
    file_manager = load_FileWriter()
    file_manager.scan_storage["scanID"] = ScanStorage(10, "scanID")
    with mock.patch.object(
        file_manager.writer_mixin, "compile_full_filename"
    ) as mock_compile_filename:
        with mock.patch.object(file_manager, "connector") as mock_connector:
            mock_compile_filename.return_value = "path"
            # replace NexusFileWriter with MockWriter
            file_manager.file_writer = MockWriter(file_manager)
            file_manager.file_writer.write = mock.Mock(side_effect=Exception("error"))
            file_manager.write_file("scanID")
            mock_connector.raise_alarm.assert_called_once()


def test_update_baseline_reading():
    file_manager = load_FileWriter()
    file_manager.scan_storage["scanID"] = ScanStorage(10, "scanID")
    with mock.patch.object(file_manager, "producer") as mock_producer:
        mock_producer.get.return_value = BECMessage.ScanBaselineMessage(
            scanID="scanID", data={"data": "data"}
        ).dumps()
        file_manager.update_baseline_reading("scanID")
        assert file_manager.scan_storage["scanID"].baseline == {"data": "data"}
        mock_producer.get.assert_called_once_with(MessageEndpoints.public_scan_baseline("scanID"))


def test_scan_storage_append():
    storage = ScanStorage(10, "scanID")
    storage.append(1, {"data": "data"})
    assert storage.scan_segments[1] == {"data": "data"}
    assert storage.scan_finished is False


def test_scan_storage_ready_to_write():
    storage = ScanStorage(10, "scanID")
    storage.num_points = 1
    storage.scan_finished = True
    storage.append(1, {"data": "data"})
    assert storage.ready_to_write() is True


def test_update_file_references():
    file_manager = load_FileWriter()
    with mock.patch.object(file_manager, "producer") as mock_producer:
        file_manager.update_file_references("scanID")
        mock_producer.keys.assert_not_called()


def test_update_file_references_gets_keys():
    file_manager = load_FileWriter()
    file_manager.scan_storage["scanID"] = ScanStorage(10, "scanID")
    with mock.patch.object(file_manager, "producer") as mock_producer:
        file_manager.update_file_references("scanID")
        mock_producer.keys.assert_called_once_with(MessageEndpoints.public_file("scanID", "*"))


def test_update_async_data():
    file_manager = load_FileWriter()
    file_manager.scan_storage["scanID"] = ScanStorage(10, "scanID")
    with mock.patch.object(file_manager, "producer") as mock_producer:
        with mock.patch.object(file_manager, "_process_async_data") as mock_process:
            key = f"{MessageEndpoints.device_async_readback('scanID', 'dev1')}:stream"
            mock_producer.keys.return_value = [
                key.encode(),
            ]
            data = [
                (b"0-0", b'{"data": "data"}'),
            ]
            mock_producer.xrange.return_value = data
            file_manager.update_async_data("scanID")
            mock_producer.xrange.assert_called_once_with(key, min="-", max="+")
            mock_process.assert_called_once_with(data, "scanID", "dev1")


def test_process_async_data_single_entry():
    file_manager = load_FileWriter()
    file_manager.scan_storage["scanID"] = ScanStorage(10, "scanID")
    data = [
        (b"0-0", {b"data": BECMessage.DeviceMessage(signals={"data": np.zeros((10, 10))}).dumps()}),
    ]
    file_manager._process_async_data(data, "scanID", "dev1")
    assert np.isclose(
        file_manager.scan_storage["scanID"].async_data["dev1"]["data"], np.zeros((10, 10))
    ).all()


def test_process_async_data_extend():
    file_manager = load_FileWriter()
    file_manager.scan_storage["scanID"] = ScanStorage(10, "scanID")
    data = [
        (
            b"0-0",
            {
                b"data": BECMessage.DeviceMessage(
                    signals={"data": np.zeros((10, 10))}, metadata={"async_update": "extend"}
                ).dumps()
            },
        )
        for ii in range(10)
    ]
    file_manager._process_async_data(data, "scanID", "dev1")
    assert file_manager.scan_storage["scanID"].async_data["dev1"]["data"].shape == (100, 10)


def test_process_async_data_append():
    file_manager = load_FileWriter()
    file_manager.scan_storage["scanID"] = ScanStorage(10, "scanID")
    data = [
        (
            b"0-0",
            {
                b"data": BECMessage.DeviceMessage(
                    signals={"data": np.zeros((10, 10))}, metadata={"async_update": "append"}
                ).dumps()
            },
        )
        for ii in range(10)
    ]
    file_manager._process_async_data(data, "scanID", "dev1")
    assert len(file_manager.scan_storage["scanID"].async_data["dev1"]["data"]) == 10


def test_process_async_data_replace():
    file_manager = load_FileWriter()
    file_manager.scan_storage["scanID"] = ScanStorage(10, "scanID")
    data = [
        (
            b"0-0",
            {
                b"data": BECMessage.DeviceMessage(
                    signals={"data": np.zeros((10, 10))}, metadata={"async_update": "replace"}
                ).dumps()
            },
        )
        for ii in range(10)
    ]
    file_manager._process_async_data(data, "scanID", "dev1")
    assert file_manager.scan_storage["scanID"].async_data["dev1"]["data"].shape == (10, 10)


def test_update_scan_storage_with_status_ignores_none():
    file_manager = load_FileWriter()
    file_manager.update_scan_storage_with_status(
        BECMessage.ScanStatusMessage(
            scanID=None,
            status="closed",
            info={},
        )
    )
    assert file_manager.scan_storage == {}


def test_ready_to_write():
    file_manager = load_FileWriter()
    file_manager.scan_storage["scanID"] = ScanStorage(10, "scanID")
    file_manager.scan_storage["scanID"].scan_finished = True
    file_manager.scan_storage["scanID"].num_points = 1
    file_manager.scan_storage["scanID"].scan_segments = {"0": {"data": np.zeros((10, 10))}}
    assert file_manager.scan_storage["scanID"].ready_to_write() is True
    file_manager.scan_storage["scanID1"] = ScanStorage(101, "scanID1")
    file_manager.scan_storage["scanID1"].scan_finished = True
    file_manager.scan_storage["scanID1"].num_points = 2
    file_manager.scan_storage["scanID1"].scan_segments = {"0": {"data": np.zeros((10, 10))}}
    assert file_manager.scan_storage["scanID1"].ready_to_write() is False


def test_ready_to_write_forced():
    file_manager = load_FileWriter()
    file_manager.scan_storage["scanID"] = ScanStorage(10, "scanID")
    file_manager.scan_storage["scanID"].scan_finished = False
    file_manager.scan_storage["scanID"].forced_finish = True
    assert file_manager.scan_storage["scanID"].ready_to_write() is True
