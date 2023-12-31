import uuid
from unittest import mock

import pytest
from bec_lib.core import BECMessage, MessageEndpoints
from utils import load_ScanServerMock

from scan_server.errors import DeviceMessageError, ScanAbortion
from scan_server.scan_assembler import ScanAssembler
from scan_server.scan_queue import (
    InstructionQueueItem,
    InstructionQueueStatus,
    QueueManager,
    RequestBlock,
    RequestBlockQueue,
    ScanQueue,
)
from scan_server.scan_worker import DeviceMsg, ScanWorker


def get_scan_worker() -> ScanWorker:
    k = load_ScanServerMock()
    return ScanWorker(parent=k)


class RequestBlockQueueMock(RequestBlockQueue):
    request_blocks = []
    _scanID = []

    @property
    def scanID(self):
        return self._scanID

    def append(self, msg):
        pass


class InstructionQueueMock(InstructionQueueItem):
    def __init__(self, parent: ScanQueue, assembler: ScanAssembler, worker: ScanWorker) -> None:
        super().__init__(parent, assembler, worker)
        self.queue = RequestBlockQueueMock(self, assembler)
        # self.queue.active_rb = []
        self.idx = 1

    def append_scan_request(self, msg):
        self.scan_msgs.append(msg)
        self.queue.append(msg)

    def __next__(self):
        if (
            self.status
            in [
                InstructionQueueStatus.RUNNING,
                InstructionQueueStatus.DEFERRED_PAUSE,
                InstructionQueueStatus.PENDING,
            ]
            and self.idx < 5
        ):
            self.idx += 1
            return "instr_status"

        else:
            raise StopIteration

        # while self.status == InstructionQueueStatus.PAUSED:
        #     return "instr_paused"

        # return "instr"


@pytest.mark.parametrize(
    "instruction,devices",
    [
        (
            BECMessage.DeviceInstructionMessage(
                device="samy",
                action="wait",
                parameter={
                    "type": "move",
                    "group": "scan_motor",
                    "wait_group": "scan_motor",
                },
                metadata={"readout_priority": "monitored", "DIID": 3},
            ),
            ["samy"],
        ),
        (
            BECMessage.DeviceInstructionMessage(
                device=["samx", "samy"],
                action="wait",
                parameter={
                    "type": "move",
                    "group": "scan_motor",
                    "wait_group": "scan_motor",
                },
                metadata={"readout_priority": "monitored", "DIID": 3},
            ),
            ["samx", "samy"],
        ),
        (
            BECMessage.DeviceInstructionMessage(
                device="",
                action="wait",
                parameter={
                    "type": "move",
                    "group": "scan_motor",
                    "wait_group": "scan_motor",
                },
                metadata={"readout_priority": "monitored", "DIID": 3},
            ),
            ["samx", "samy"],
        ),
        (
            BECMessage.DeviceInstructionMessage(
                device="",
                action="wait",
                parameter={
                    "type": "move",
                    "group": "primary",
                    "wait_group": "scan_motor",
                },
                metadata={"readout_priority": "monitored", "DIID": 3},
            ),
            ["samx", "samy"],
        ),
        (
            BECMessage.DeviceInstructionMessage(
                device="",
                action="wait",
                parameter={
                    "type": "move",
                    "group": "nogroup",
                    "wait_group": "scan_motor",
                },
                metadata={"readout_priority": "monitored", "DIID": 3},
            ),
            ["samx", "samy"],
        ),
    ],
)
def test_get_devices_from_instruction(instruction, devices):
    worker = get_scan_worker()
    worker.scan_motors = devices
    worker.readout_priority.update({"monitored": devices})

    returned_devices = worker._get_devices_from_instruction(instruction)

    if not instruction.content.get("device"):
        group = instruction.content["parameter"].get("group")
        if group == "primary":
            assert returned_devices == worker.device_manager.devices.monitored_devices(
                worker.scan_motors
            )
        elif group == "scan_motor":
            assert returned_devices == devices
        else:
            assert returned_devices == []
    else:
        assert returned_devices == [worker.device_manager.devices[dev] for dev in devices]


@pytest.mark.parametrize(
    "instructions",
    [
        (
            BECMessage.DeviceInstructionMessage(
                device="samx",
                action="set",
                parameter={"value": 10, "wait_group": "scan_motor"},
                metadata={"readout_priority": "monitored", "DIID": 3},
            )
        ),
        BECMessage.DeviceInstructionMessage(
            device="samx",
            action="set",
            parameter={"value": 10, "wait_group": "scan_motor"},
            metadata={"readout_priority": "monitored", "DIID": None},
        ),
    ],
)
def test_add_wait_group(instructions):
    worker = get_scan_worker()
    if instructions.metadata["DIID"]:
        worker._add_wait_group(instructions)
        assert worker._groups == {"scan_motor": {"samx": 3}}

        worker._groups["scan_motor"] = {"samy": 2}
        worker._add_wait_group(instructions)
        assert worker._groups == {"scan_motor": {"samy": 2, "samx": 3}}

    else:
        with pytest.raises(DeviceMessageError) as exc_info:
            worker._add_wait_group(instructions)
        assert exc_info.value.args[0] == "Device message metadata does not contain a DIID entry."


def test_add_wait_group_to_existing_wait_group():
    instr1 = BECMessage.DeviceInstructionMessage(
        device="samx",
        action="set",
        parameter={"value": 10, "wait_group": "scan_motor"},
        metadata={"readout_priority": "monitored", "DIID": 3},
    )
    instr2 = BECMessage.DeviceInstructionMessage(
        device="samx",
        action="set",
        parameter={"value": 10, "wait_group": "scan_motor"},
        metadata={"readout_priority": "monitored", "DIID": 4},
    )
    worker = get_scan_worker()
    worker._add_wait_group(instr1)
    worker._add_wait_group(instr2)
    assert worker._groups == {"scan_motor": {"samx": 4}}


@pytest.mark.parametrize(
    "instructions,wait_type",
    [
        (
            BECMessage.DeviceInstructionMessage(
                device="samy",
                action="wait",
                parameter={
                    "type": "move",
                    "group": "scan_motor",
                    "wait_group": "scan_motor",
                },
                metadata={"readout_priority": "monitored", "DIID": 3},
            ),
            "move",
        ),
        (
            BECMessage.DeviceInstructionMessage(
                device="samy",
                action="wait",
                parameter={
                    "type": "read",
                    "group": "scan_motor",
                    "wait_group": "scan_motor",
                },
                metadata={"readout_priority": "monitored", "DIID": 3},
            ),
            "read",
        ),
        (
            BECMessage.DeviceInstructionMessage(
                device="samy",
                action="wait",
                parameter={
                    "type": "trigger",
                    "group": "scan_motor",
                    "wait_group": "scan_motor",
                },
                metadata={"readout_priority": "monitored", "DIID": 3},
            ),
            "trigger",
        ),
        (
            BECMessage.DeviceInstructionMessage(
                device="samy",
                action="wait",
                parameter={
                    "type": None,
                    "group": "scan_motor",
                    "wait_group": "scan_motor",
                },
                metadata={"readout_priority": "monitored", "DIID": 3},
            ),
            None,
        ),
    ],
)
def test_wait_for_devices(instructions, wait_type):
    worker = get_scan_worker()

    with mock.patch.object(worker, "_wait_for_idle") as idle_mock:
        with mock.patch.object(worker, "_wait_for_read") as read_mock:
            with mock.patch.object(worker, "_wait_for_trigger") as trigger_mock:
                if wait_type:
                    worker._wait_for_devices(instructions)

                if wait_type == "move":
                    idle_mock.assert_called_once_with(instructions)
                elif wait_type == "read":
                    read_mock.assert_called_once_with(instructions)
                elif wait_type == "trigger":
                    trigger_mock.assert_called_once_with(instructions)
                else:
                    with pytest.raises(DeviceMessageError) as exc_info:
                        worker._wait_for_devices(instructions)
                    assert exc_info.value.args[0] == "Unknown wait command"


@pytest.mark.parametrize(
    "device_status,devices,instr,abort",
    [
        (
            [
                BECMessage.DeviceReqStatusMessage(
                    device="samx",
                    success=True,
                    metadata={
                        "readout_priority": "monitored",
                        "DIID": 3,
                        "scanID": "scanID",
                        "RID": "requestID",
                    },
                )
            ],
            [("samx", 4)],
            BECMessage.DeviceInstructionMessage(
                device=["samx"],
                action="wait",
                parameter={"type": "move", "wait_group": "scan_motor"},
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 4,
                    "scanID": "scanID",
                    "RID": "requestID",
                },
            ),
            False,
        ),
        (
            [
                BECMessage.DeviceReqStatusMessage(
                    device="samx",
                    success=False,
                    metadata={
                        "readout_priority": "monitored",
                        "DIID": 3,
                        "scanID": "scanID",
                        "RID": "request",
                    },
                )
            ],
            [("samx", 4)],
            BECMessage.DeviceInstructionMessage(
                device=["samx"],
                action="wait",
                parameter={"type": "move", "wait_group": "scan_motor"},
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 4,
                    "scanID": "scanID",
                    "RID": "requestID",
                },
            ),
            False,
        ),
        (
            [
                BECMessage.DeviceReqStatusMessage(
                    device="samx",
                    success=False,
                    metadata={
                        "readout_priority": "monitored",
                        "DIID": 4,
                        "scanID": "scanID",
                        "RID": "requestID",
                    },
                )
            ],
            [("samx", 4)],
            BECMessage.DeviceInstructionMessage(
                device=["samx"],
                action="wait",
                parameter={"type": "move", "wait_group": "scan_motor"},
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 4,
                    "scanID": "scanID",
                    "RID": "requestID",
                },
            ),
            True,
        ),
        (
            [
                BECMessage.DeviceReqStatusMessage(
                    device="samx",
                    success=False,
                    metadata={
                        "readout_priority": "monitored",
                        "DIID": 3,
                        "scanID": "scanID",
                        "RID": "requestID",
                    },
                )
            ],
            [("samx", 4)],
            BECMessage.DeviceInstructionMessage(
                device=["samx"],
                action="wait",
                parameter={"type": "move", "wait_group": "scan_motor"},
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 4,
                    "scanID": "scanID",
                    "RID": "requestID",
                },
            ),
            False,
        ),
    ],
)
def test_check_for_failed_movements(device_status, devices, instr, abort):
    worker = get_scan_worker()
    if abort:
        with pytest.raises(ScanAbortion):
            worker.device_manager.producer._get_buffer[
                MessageEndpoints.device_readback("samx")
            ] = BECMessage.DeviceMessage(signals={"samx": {"value": 4}}, metadata={}).dumps()
            worker._check_for_failed_movements(device_status, devices, instr)
    else:
        worker._check_for_failed_movements(device_status, devices, instr)


@pytest.mark.parametrize(
    "msg1,msg2,req_msg",
    [
        (
            BECMessage.DeviceInstructionMessage(
                device="samx",
                action="set",
                parameter={"value": 10, "wait_group": "scan_motor"},
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 3,
                    "scanID": "scanID",
                    "RID": "requestID",
                },
            ),
            BECMessage.DeviceInstructionMessage(
                device=["samx"],
                action="wait",
                parameter={"type": "move", "wait_group": "scan_motor"},
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 4,
                    "scanID": "scanID",
                    "RID": "requestID",
                },
            ),
            BECMessage.DeviceReqStatusMessage(
                device="samx",
                success=False,
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 3,
                    "scanID": "scanID",
                    "RID": "requestID",
                },
            ),
        ),
        (
            BECMessage.DeviceInstructionMessage(
                device="samx",
                action="set",
                parameter={"value": 10, "wait_group": "scan_motor"},
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 3,
                    "scanID": "scanID",
                    "RID": "requestID",
                },
            ),
            BECMessage.DeviceInstructionMessage(
                device=["samx"],
                action="wait",
                parameter={"type": "move", "wait_group": "scan_motor"},
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 4,
                    "scanID": "scanID",
                    "RID": "requestID",
                },
            ),
            BECMessage.DeviceReqStatusMessage(
                device="samx",
                success=True,
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 3,
                    "scanID": "scanID",
                    "RID": "requestID",
                },
            ),
        ),
    ],
)
def test_wait_for_idle(msg1, msg2, req_msg: BECMessage.DeviceReqStatusMessage):
    worker = get_scan_worker()

    with mock.patch(
        "scan_server.scan_worker.ScanWorker._get_device_status",
        return_value=[req_msg.dumps()],
    ) as device_status:
        worker.device_manager.producer._get_buffer[
            MessageEndpoints.device_readback("samx")
        ] = BECMessage.DeviceMessage(signals={"samx": {"value": 4}}, metadata={}).dumps()

        worker._add_wait_group(msg1)
        if req_msg.content["success"]:
            worker._wait_for_idle(msg2)
        else:
            with pytest.raises(ScanAbortion):
                worker._wait_for_idle(msg2)


@pytest.mark.parametrize(
    "msg1,msg2,req_msg",
    [
        (
            BECMessage.DeviceInstructionMessage(
                device=["samx"],
                action="set",
                parameter={"value": 10, "wait_group": "scan_motor"},
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 3,
                    "scanID": "scanID",
                    "RID": "requestID",
                },
            ),
            BECMessage.DeviceInstructionMessage(
                device=["samx"],
                action="wait",
                parameter={"type": "move", "wait_group": "scan_motor"},
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 4,
                    "scanID": "scanID",
                    "RID": "requestID",
                },
            ),
            BECMessage.DeviceStatusMessage(
                device="samx",
                status=0,
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 3,
                    "scanID": "scanID",
                    "RID": "requestID",
                },
            ),
        ),
    ],
)
def test_wait_for_read(msg1, msg2, req_msg: BECMessage.DeviceReqStatusMessage):
    worker = get_scan_worker()

    with mock.patch(
        "scan_server.scan_worker.ScanWorker._get_device_status",
        return_value=[req_msg.dumps()],
    ) as device_status:
        with mock.patch.object(worker, "_check_for_interruption") as interruption_mock:
            assert worker._groups == {}
            worker._groups["scan_motor"] = {"samx": 3, "samy": 4}
            worker.device_manager.producer._get_buffer[
                MessageEndpoints.device_readback("samx")
            ] = BECMessage.DeviceMessage(signals={"samx": {"value": 4}}, metadata={}).dumps()
            worker._add_wait_group(msg1)
            worker._wait_for_read(msg2)
            assert worker._groups == {"scan_motor": {"samy": 4}}
            interruption_mock.assert_called_once()


@pytest.mark.parametrize(
    "instr",
    [
        (
            BECMessage.DeviceInstructionMessage(
                device=None,
                action="set",
                parameter={"time": 0.1},
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 3,
                    "scanID": "scanID",
                    "RID": "requestID",
                },
            )
        ),
    ],
)
def test_wait_for_trigger(instr):
    worker = get_scan_worker()
    worker._last_trigger = instr

    with mock.patch.object(worker, "_get_device_status") as status_mock:
        with mock.patch.object(worker, "_check_for_interruption") as interruption_mock:
            status_mock.return_value = [
                BECMessage.DeviceReqStatusMessage(
                    device="eiger",
                    success=True,
                    metadata={
                        "readout_priority": "monitored",
                        "DIID": 3,
                        "scanID": "scanID",
                        "RID": "requestID",
                    },
                ).dumps(),
            ]
            worker._wait_for_trigger(instr)
            status_mock.assert_called_once_with(MessageEndpoints.device_req_status, ["eiger"])
            interruption_mock.assert_called_once()


def test_wait_for_stage():
    worker = get_scan_worker()
    devices = ["samx", "samy"]
    with mock.patch.object(worker, "_get_device_status") as status_mock:
        with mock.patch.object(worker, "_check_for_interruption") as interruption_mock:
            worker._wait_for_stage(True, devices, {})
            status_mock.assert_called_once_with(MessageEndpoints.device_staged, devices)
            interruption_mock.assert_called_once()


def test_wait_for_device_server():
    worker = get_scan_worker()
    with mock.patch.object(worker.parent, "wait_for_service") as service_mock:
        worker._wait_for_device_server()
        service_mock.assert_called_once_with("DeviceServer")


@pytest.mark.parametrize(
    "instr",
    [
        (
            BECMessage.DeviceInstructionMessage(
                device=["samx"],
                action="set",
                parameter={"value": 10, "wait_group": "scan_motor", "time": 30},
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 3,
                    "scanID": "scanID",
                    "RID": "requestID",
                },
            )
        ),
    ],
)
def test_set_devices(instr):
    worker = get_scan_worker()
    with mock.patch.object(worker.device_manager.producer, "send") as send_mock:
        worker._set_devices(instr)
        send_mock.assert_called_once_with(MessageEndpoints.device_instructions(), instr.dumps())


@pytest.mark.parametrize(
    "instr",
    [
        (
            BECMessage.DeviceInstructionMessage(
                device=["samx"],
                action="trigger",
                parameter={"value": 10, "wait_group": "scan_motor", "time": 30},
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 3,
                    "scanID": "scanID",
                    "RID": "requestID",
                },
            )
        ),
    ],
)
def test_trigger_devices(instr):
    worker = get_scan_worker()
    with mock.patch.object(worker.device_manager.producer, "send") as send_mock:
        worker._trigger_devices(instr)
        devices = [dev.name for dev in worker.device_manager.devices.detectors()]

        send_mock.assert_called_once_with(
            MessageEndpoints.device_instructions(),
            BECMessage.DeviceInstructionMessage(
                device=devices,
                action="trigger",
                parameter={"value": 10, "wait_group": "scan_motor", "time": 30},
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 3,
                    "scanID": "scanID",
                    "RID": "requestID",
                },
            ).dumps(),
        )


@pytest.mark.parametrize(
    "instr",
    [
        (
            BECMessage.DeviceInstructionMessage(
                device=["samx"],
                action="trigger",
                parameter={"value": 10, "wait_group": "scan_motor", "time": 30},
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 3,
                    "scanID": "scanID",
                    "RID": "requestID",
                },
            )
        ),
    ],
)
def test_send_rpc(instr):
    worker = get_scan_worker()
    with mock.patch.object(worker.device_manager.producer, "send") as send_mock:
        worker._send_rpc(instr)
        send_mock.assert_called_once_with(MessageEndpoints.device_instructions(), instr.dumps())


@pytest.mark.parametrize(
    "instr",
    [
        (
            BECMessage.DeviceInstructionMessage(
                device=["samx"],
                action="read",
                parameter={"value": 10, "wait_group": "scan_motor", "time": 30},
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 3,
                    "scanID": "scanID",
                    "RID": "requestID",
                },
            )
        ),
        (
            BECMessage.DeviceInstructionMessage(
                device=[],
                action="read",
                parameter={"value": 10, "wait_group": "scan_motor", "time": 30},
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 3,
                    "scanID": "scanID",
                    "RID": "requestID",
                },
            )
        ),
    ],
)
def test_read_devices(instr):
    worker = get_scan_worker()
    instr_devices = instr.content["device"]
    if instr_devices is None:
        instr_devices = []
    worker.readout_priority.update({"monitored": instr_devices})
    devices = [dev.name for dev in worker._get_devices_from_instruction(instr)]
    with mock.patch.object(worker.device_manager.producer, "send") as send_mock:
        worker._read_devices(instr)

        if instr.content.get("device"):
            send_mock.assert_called_once_with(
                MessageEndpoints.device_instructions(),
                BECMessage.DeviceInstructionMessage(
                    device=["samx"],
                    action="read",
                    parameter=instr.content["parameter"],
                    metadata=instr.metadata,
                ).dumps(),
            )
        else:
            send_mock.assert_called_once_with(
                MessageEndpoints.device_instructions(),
                BECMessage.DeviceInstructionMessage(
                    device=devices,
                    action="read",
                    parameter=instr.content["parameter"],
                    metadata=instr.metadata,
                ).dumps(),
            )


@pytest.mark.parametrize(
    "instr, devices, parameter, metadata",
    [
        (
            BECMessage.DeviceInstructionMessage(
                device=["samx"],
                action="trigger",
                parameter={"value": 10, "wait_group": "scan_motor", "time": 30},
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 3,
                    "scanID": "scanID",
                    "RID": "requestID",
                },
            ),
            ["samx"],
            {"value": 10, "wait_group": "scan_motor", "time": 30},
            {"readout_priority": "monitored", "DIID": 3, "scanID": "scanID", "RID": "requestID"},
        ),
    ],
)
def test_kickoff_devices(instr, devices, parameter, metadata):
    worker = get_scan_worker()
    with mock.patch.object(worker.device_manager.producer, "send") as send_mock:
        worker._kickoff_devices(instr)
        send_mock.assert_called_once_with(
            MessageEndpoints.device_instructions(),
            BECMessage.DeviceInstructionMessage(
                device=devices,
                action="kickoff",
                parameter=parameter,
                metadata=metadata,
            ).dumps(),
        )


@pytest.mark.parametrize(
    "instr, devices",
    [
        (
            BECMessage.DeviceInstructionMessage(
                device=["samx"],
                action="trigger",
                parameter={"value": 10, "wait_group": "scan_motor", "time": 30},
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 3,
                    "scanID": "scanID",
                    "RID": "requestID",
                },
            ),
            None,
        ),
    ],
)
def test_publish_readback(instr, devices):
    worker = get_scan_worker()
    with mock.patch.object(worker, "_get_readback", return_value=[{}]) as get_readback:
        with mock.patch.object(worker.device_manager, "producer") as producer_mock:
            worker._publish_readback(instr)

            get_readback.assert_called_once_with(["samx"])
            pipe = producer_mock.pipeline()
            msg = BECMessage.DeviceMessage(
                signals={},
                metadata=instr.metadata,
            ).dumps()

            producer_mock.set_and_publish.assert_called_once_with(
                MessageEndpoints.device_read("samx"), msg, pipe
            )
            pipe.execute.assert_called_once()


def test_get_readback():
    worker = get_scan_worker()
    devices = ["samx"]
    with mock.patch.object(worker.device_manager, "producer") as producer_mock:
        worker._get_readback(devices)
        pipe = producer_mock.pipeline()
        producer_mock.get.assert_called_once_with(
            MessageEndpoints.device_readback("samx"), pipe=pipe
        )
        pipe.execute.assert_called_once()


def test_publish_data_as_read():
    worker = get_scan_worker()
    instr = BECMessage.DeviceInstructionMessage(
        device=["samx"],
        action="publish_data_as_read",
        parameter={"data": {}},
        metadata={
            "readout_priority": "monitored",
            "DIID": 3,
            "scanID": "scanID",
            "RID": "requestID",
        },
    )
    with mock.patch.object(worker.device_manager, "producer") as producer_mock:
        worker._publish_data_as_read(instr)
        msg = BECMessage.DeviceMessage(
            signals=instr.content["parameter"]["data"],
            metadata=instr.metadata,
        ).dumps()
        producer_mock.set_and_publish.assert_called_once_with(
            MessageEndpoints.device_read("samx"), msg
        )


def test_publish_data_as_read_multiple():
    worker = get_scan_worker()
    data = [{"samx": {}}, {"samy": {}}]
    devices = ["samx", "samy"]
    instr = BECMessage.DeviceInstructionMessage(
        device=devices,
        action="publish_data_as_read",
        parameter={"data": data},
        metadata={
            "readout_priority": "monitored",
            "DIID": 3,
            "scanID": "scanID",
            "RID": "requestID",
        },
    )
    with mock.patch.object(worker.device_manager, "producer") as producer_mock:
        worker._publish_data_as_read(instr)
        mock_calls = []
        for device, dev_data in zip(devices, data):
            msg = BECMessage.DeviceMessage(
                signals=dev_data,
                metadata=instr.metadata,
            ).dumps()
            mock_calls.append(mock.call(MessageEndpoints.device_read(device), msg))
        assert producer_mock.set_and_publish.mock_calls == mock_calls


def test_check_for_interruption():
    worker = get_scan_worker()
    worker.status = InstructionQueueStatus.STOPPED
    with pytest.raises(ScanAbortion) as exc_info:
        worker._check_for_interruption()


@pytest.mark.parametrize(
    "instr, corr_num_points, scan_id",
    [
        (
            BECMessage.DeviceInstructionMessage(
                device=None,
                action="open_scan",
                parameter={"num_points": 150, "scan_motors": ["samx", "samy"]},
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 18,
                    "scanID": "12345",
                    "scan_def_id": 100,
                    "pointID": 50,
                    "RID": 11,
                },
            ),
            201,
            False,
        ),
        (
            BECMessage.DeviceInstructionMessage(
                device=None,
                action="open_scan",
                parameter={"num_points": 150},
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 18,
                    "scanID": "12345",
                    "RID": 11,
                },
            ),
            150,
            True,
        ),
    ],
)
def test_open_scan(instr, corr_num_points, scan_id):
    worker = get_scan_worker()

    if not scan_id:
        assert worker.scan_id == None
    else:
        worker.scan_id = 111
        worker.scan_motors = ["bpm4i"]

    if "pointID" in instr.metadata:
        worker.max_point_id = instr.metadata["pointID"]

    assert worker.parent.producer.get(MessageEndpoints.scan_number()) == None

    with mock.patch.object(worker, "current_instruction_queue_item") as queue_mock:
        with mock.patch.object(worker, "_initialize_scan_info") as init_mock:
            with mock.patch.object(worker.scan_report_instructions, "append") as instr_append_mock:
                with mock.patch.object(worker, "_send_scan_status") as send_mock:
                    with mock.patch.object(
                        worker.current_instruction_queue_item.parent.queue_manager,
                        "send_queue_status",
                    ) as queue_status_mock:
                        active_rb = queue_mock.active_request_block
                        active_rb.scan_report_instructions = []
                        worker._open_scan(instr)

                        if not scan_id:
                            assert worker.scan_id == instr.metadata.get("scanID")
                            assert worker.scan_motors == [
                                worker.device_manager.devices["samx"],
                                worker.device_manager.devices["samy"],
                            ]
                        else:
                            assert worker.scan_id == 111
                            assert worker.scan_motors == ["bpm4i"]
                        init_mock.assert_called_once_with(active_rb, instr, corr_num_points)
                        assert active_rb.scan_report_instructions == [
                            {"table_wait": corr_num_points}
                        ]
                        queue_status_mock.assert_called_once()
                        send_mock.assert_called_once_with("open")


@pytest.mark.parametrize(
    "msg",
    [
        BECMessage.ScanQueueMessage(
            scan_type="grid_scan",
            parameter={"args": {"samx": (-5, 5, 3)}, "kwargs": {}, "num_points": 100},
            queue="primary",
            metadata={"RID": "something"},
        ),
    ],
)
def test_initialize_scan_info(msg):
    worker = get_scan_worker()
    scan_server = load_ScanServerMock()
    rb = RequestBlock(msg, assembler=ScanAssembler(parent=scan_server))
    assert rb.metadata == {"RID": "something"}

    with mock.patch.object(worker, "current_instruction_queue_item"):
        worker._initialize_scan_info(rb, msg, msg.content["parameter"].get("num_points"))
        assert worker.current_scan_info == {
            **msg.metadata,
            **msg.content["parameter"],
            "scan_number": 2,
            "dataset_number": 3,
            "exp_time": None,
            "settling_time": 0,
            "readout_time": 0,
            "acquisition_config": {"default": {"exp_time": 0, "readout_time": 0}},
            "scan_report_hint": rb.scan.scan_report_hint,
            "scan_report_devices": rb.scan.scan_report_devices,
            "num_points": 100,
            "scan_msgs": [],
            "enforce_sync": True,
            "frames_per_trigger": 1,
        }


@pytest.mark.parametrize(
    "msg,scan_id,max_point_id,exp_num_points",
    [
        (
            BECMessage.DeviceInstructionMessage(
                device=None,
                action="close_scan",
                parameter={},
                metadata={"readout_priority": "monitored", "DIID": 18, "scanID": "12345"},
            ),
            "12345",
            19,
            20,
        ),
        (
            BECMessage.DeviceInstructionMessage(
                device=None,
                action="close_scan",
                parameter={},
                metadata={"readout_priority": "monitored", "DIID": 18, "scanID": "12345"},
            ),
            "0987",
            200,
            19,
        ),
    ],
)
def test_close_scan(msg, scan_id, max_point_id, exp_num_points):
    worker = get_scan_worker()
    worker.scan_id = scan_id
    worker.current_scan_info["num_points"] = 19

    reset = bool(worker.scan_id == msg.metadata["scanID"])
    with mock.patch.object(worker, "_send_scan_status") as send_scan_status_mock:
        worker._close_scan(msg, max_point_id=max_point_id)
        if reset:
            send_scan_status_mock.assert_called_with("closed")
            assert worker.scan_id == None
        else:
            assert worker.scan_id == scan_id
    assert worker.current_scan_info["num_points"] == exp_num_points


@pytest.mark.parametrize(
    "msg",
    [
        BECMessage.DeviceInstructionMessage(
            device=None,
            action="close_scan",
            parameter={},
            metadata={"readout_priority": "monitored", "DIID": 18, "scanID": "12345"},
        ),
    ],
)
def test_stage_device(msg):
    worker = get_scan_worker()

    with mock.patch.object(worker, "_wait_for_stage") as wait_mock:
        with mock.patch.object(worker.device_manager.producer, "send") as send_mock:
            worker._stage_devices(msg)

            for dev in worker.device_manager.devices.enabled_devices:
                assert dev.name in worker._staged_devices
            send_mock.assert_called_once_with(
                MessageEndpoints.device_instructions(),
                DeviceMsg(
                    device=[dev.name for dev in worker.device_manager.devices.enabled_devices],
                    action="stage",
                    parameter=msg.content["parameter"],
                    metadata=msg.metadata,
                ).dumps(),
            )
            wait_mock.assert_called_once_with(
                staged=True,
                devices=[dev.name for dev in worker.device_manager.devices.enabled_devices],
                metadata=msg.metadata,
            )


@pytest.mark.parametrize(
    "msg, devices, parameter, metadata, cleanup",
    [
        (
            BECMessage.DeviceInstructionMessage(
                device=None,
                action="close_scan",
                parameter={"parameter": "param"},
                metadata={"readout_priority": "monitored", "DIID": 18, "scanID": "12345"},
            ),
            ["samx"],
            {"parameter": "param"},
            {"readout_priority": "monitored", "DIID": 18, "scanID": "12345"},
            False,
        ),
        (None, None, {}, {}, False),
        (None, None, {}, {}, True),
    ],
)
def test_unstage_device(msg, devices, parameter, metadata, cleanup):
    worker = get_scan_worker()
    if not devices:
        devices = [dev.name for dev in worker.device_manager.devices.enabled_devices]

    with mock.patch.object(worker.device_manager.producer, "send") as send_mock:
        with mock.patch.object(worker, "_wait_for_stage") as wait_mock:
            worker._unstage_devices(msg, devices, cleanup)

            send_mock.assert_called_once_with(
                MessageEndpoints.device_instructions(),
                DeviceMsg(
                    device=devices,
                    action="unstage",
                    parameter=parameter,
                    metadata=metadata,
                ).dumps(),
            )
            if cleanup:
                wait_mock.assert_not_called()
            else:
                wait_mock.assert_called_once_with(
                    staged=False,
                    devices=devices,
                    metadata=metadata,
                )


@pytest.mark.parametrize(
    "status,expire",
    [
        (
            "open",
            None,
        ),
        (
            "closed",
            1800,
        ),
        (
            "aborted",
            1800,
        ),
    ],
)
def test_send_scan_status(status, expire):
    worker = get_scan_worker()
    worker.current_scanID = str(uuid.uuid4())
    worker._send_scan_status(status)
    scan_info_msgs = [
        msg
        for msg in worker.device_manager.producer.message_sent
        if msg["queue"] == MessageEndpoints.public_scan_info(scanID=worker.current_scanID)
    ]
    assert len(scan_info_msgs) == 1
    assert scan_info_msgs[0]["expire"] == expire


@pytest.mark.parametrize("abortion", [False, True])
def test_process_instructions(abortion):
    worker = get_scan_worker()
    scan_server = load_ScanServerMock()
    scan_queue = ScanQueue(QueueManager(scan_server))
    queue = InstructionQueueMock(
        parent=scan_queue, assembler=ScanAssembler(parent=scan_server), worker=worker
    )

    with mock.patch.object(worker, "_wait_for_device_server") as wait_mock:
        with mock.patch.object(worker, "reset") as reset_mock:
            with mock.patch.object(worker, "_check_for_interruption") as interruption_mock:
                with mock.patch.object(queue.queue, "active_rb") as rb_mock:
                    with mock.patch.object(worker, "_instruction_step") as step_mock:
                        if abortion:
                            interruption_mock.side_effect = ScanAbortion
                            with pytest.raises(ScanAbortion) as exc_info:
                                worker._process_instructions(queue)
                        else:
                            worker._process_instructions(queue)

                        assert worker.max_point_id == 0
                        wait_mock.assert_called_once()

                        if not abortion:
                            assert interruption_mock.call_count == 4
                            assert worker._exposure_time == getattr(rb_mock.scan, "exp_time", None)
                            assert step_mock.call_count == 4
                            assert queue.is_active == False
                            assert queue.status == InstructionQueueStatus.COMPLETED
                            assert worker.current_instruction_queue_item == None
                            reset_mock.assert_called_once()

                        else:
                            assert worker._groups == {}
                            assert queue.stopped == True
                            assert interruption_mock.call_count == 1
                            assert queue.is_active == True
                            assert queue.status == InstructionQueueStatus.PENDING
                            assert worker.current_instruction_queue_item == queue


@pytest.mark.parametrize(
    "msg,method",
    [
        (
            BECMessage.DeviceInstructionMessage(
                device=None,
                action="open_scan",
                parameter={"readout_priority": {"monitored": [], "baseline": [], "ignored": []}},
                metadata={"readout_priority": "monitored", "DIID": 18, "scanID": "12345"},
            ),
            "_open_scan",
        ),
        (
            BECMessage.DeviceInstructionMessage(
                device=None,
                action="close_scan",
                parameter={},
                metadata={"readout_priority": "monitored", "DIID": 18, "scanID": "12345"},
            ),
            "_close_scan",
        ),
        (
            BECMessage.DeviceInstructionMessage(
                device=["samx"],
                action="wait",
                parameter={"type": "move", "wait_group": "scan_motor"},
                metadata={
                    "readout_priority": "monitored",
                    "DIID": 4,
                    "scanID": "12345",
                    "RID": "123456",
                },
            ),
            "_wait_for_devices",
        ),
        (
            BECMessage.DeviceInstructionMessage(
                device=None,
                action="trigger",
                parameter={"group": "trigger"},
                metadata={"readout_priority": "monitored", "DIID": 20, "pointID": 0},
            ),
            "_trigger_devices",
        ),
        (
            BECMessage.DeviceInstructionMessage(
                device="samx",
                action="set",
                parameter={
                    "value": 1.3681828686580249,
                    "wait_group": "scan_motor",
                },
                metadata={"readout_priority": "monitored", "DIID": 24},
            ),
            "_set_devices",
        ),
        (
            BECMessage.DeviceInstructionMessage(
                device=None,
                action="read",
                parameter={
                    "group": "primary",
                    "wait_group": "readout_primary",
                },
                metadata={"readout_priority": "monitored", "DIID": 30, "pointID": 1},
            ),
            "_read_devices",
        ),
        (
            BECMessage.DeviceInstructionMessage(
                device=None,
                action="stage",
                parameter={},
                metadata={"readout_priority": "monitored", "DIID": 17},
            ),
            "_stage_devices",
        ),
        (
            BECMessage.DeviceInstructionMessage(
                device=None,
                action="unstage",
                parameter={},
                metadata={"readout_priority": "monitored", "DIID": 17},
            ),
            "_unstage_devices",
        ),
        (
            BECMessage.DeviceInstructionMessage(
                device="samx",
                action="rpc",
                parameter={
                    "device": "lsamy",
                    "func": "readback.get",
                    "rpc_id": "61a7376c-36cf-41af-94b1-76c1ba821d47",
                    "args": [],
                    "kwargs": {},
                },
                metadata={"readout_priority": "monitored", "DIID": 9},
            ),
            "_send_rpc",
        ),
        (
            BECMessage.DeviceInstructionMessage(
                device="samx", action="kickoff", parameter={}, metadata={}
            ),
            "_kickoff_devices",
        ),
        (
            BECMessage.DeviceInstructionMessage(
                device=None,
                action="baseline_reading",
                parameter={},
                metadata={"readout_priority": "baseline", "DIID": 15},
            ),
            "_baseline_reading",
        ),
        (
            BECMessage.DeviceInstructionMessage(device=None, action="close_scan_def", parameter={}),
            "_close_scan",
        ),
        (
            BECMessage.DeviceInstructionMessage(
                device=None, action="publish_data_as_read", parameter={}
            ),
            "_publish_data_as_read",
        ),
        (
            BECMessage.DeviceInstructionMessage(
                device=None, action="scan_report_instruction", parameter={}
            ),
            "_process_scan_report_instruction",
        ),
    ],
)
def test_instruction_step(msg, method):
    worker = get_scan_worker()
    with mock.patch(f"scan_server.scan_worker.ScanWorker.{method}") as instruction_method:
        worker._instruction_step(msg)
        instruction_method.assert_called_once()


def test_reset():
    worker = get_scan_worker()
    worker._gropus = 1
    worker.current_scanID = 1
    worker.current_scan_info = 1
    worker.scan_id = 1
    worker.interception_msg = 1
    worker.scan_motors = 1

    worker.reset()

    assert worker._groups == {}
    assert worker.current_scanID == ""
    assert worker.current_scan_info == {}
    assert worker.scan_id == None
    assert worker.interception_msg == None
    assert worker.scan_motors == []


def test_cleanup():
    worker = get_scan_worker()
    with mock.patch.object(worker, "_unstage_devices") as unstage_mock:
        worker.cleanup()
        unstage_mock.assert_called_once_with(devices=list(worker._staged_devices), cleanup=True)


def test_shutdown():
    worker = get_scan_worker()
    with mock.patch.object(worker.signal_event, "set") as set_mock:
        with mock.patch.object(worker, "join") as join_mock:
            worker.shutdown()
            set_mock.assert_called_once()
            join_mock.assert_called_once()
