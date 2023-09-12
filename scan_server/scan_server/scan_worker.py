import datetime
import threading
import time
import traceback
from asyncio.log import logger
from typing import List

from bec_lib.core import (
    Alarms,
    BECMessage,
    Device,
    DeviceStatus,
    MessageEndpoints,
    bec_logger,
)

from .errors import DeviceMessageError, ScanAbortion
from .scan_queue import InstructionQueueItem, InstructionQueueStatus, RequestBlock

logger = bec_logger.logger

DeviceMsg = BECMessage.DeviceInstructionMessage
ScanStatusMsg = BECMessage.ScanStatusMessage


class ScanWorker(threading.Thread):
    """
    Scan worker receives device instructions and pre-processes them before sending them to the device server
    """

    def __init__(self, *, parent, queue_name: str = "primary"):
        super().__init__(daemon=True)
        self.queue_name = queue_name
        self.name = f"ScanWorker-{queue_name}"
        self.parent = parent
        self.device_manager = self.parent.device_manager
        self.connector = self.parent.connector
        self.status = InstructionQueueStatus.IDLE
        self.signal_event = threading.Event()
        self.scan_id = None
        self.scan_motors = []
        self.readout_priority = {}
        self.scan_type = None
        self.current_scanID = None
        self.current_scan_info = None
        self._staged_devices = set()
        self.max_point_id = 0
        self._exposure_time = None
        self.current_instruction_queue_item = None
        self._last_trigger = {}
        self.reset()

    @property
    def scan_report_instructions(self):
        req_block = self.current_instruction_queue_item.active_request_block
        return req_block.scan_report_instructions

    def _get_devices_from_instruction(self, instr: DeviceMsg) -> List[Device]:
        """Extract devices from instruction message

        Args:
            instr (DeviceMsg): DeviceInstructionMessage

        Returns:
            List[Device]: List of devices
        """
        devices = []
        if not instr.content.get("device"):
            group = instr.content["parameter"].get("group")
            if group == "primary":
                devices = self.device_manager.devices.monitored_devices(
                    readout_priority=self.readout_priority
                )
            elif group == "scan_motor":
                devices = self.scan_motors
        else:
            instr_devices = instr.content.get("device")
            if not isinstance(instr_devices, list):
                instr_devices = [instr_devices]
            devices = [self.device_manager.devices[dev] for dev in instr_devices]
        return devices

    def _add_wait_group(self, instr: DeviceMsg) -> None:
        """If needed, add a wait_group. This wait_group can later be used to
        wait for instructions to complete before continuing.

        Example:
            DeviceInstructionMessage(({'device': ['samx', 'samy'], 'action': 'read', 'parameter': {'group': 'scan_motor', 'wait_group': 'scan_motor'}}, {DIID': 0,...}))

            This instruction would create a new wait_group entry for the devices samx and samy, to finish DIID 0.

        Args:
            instr (DeviceMsg): DeviceInstructionMessage

        """
        wait_group = instr.content["parameter"].get("wait_group")
        action = instr.content["action"]
        if not wait_group or action == "wait":
            return

        devices = self._get_devices_from_instruction(instr)
        DIID = instr.metadata.get("DIID")
        if DIID is None:
            raise DeviceMessageError("Device message metadata does not contain a DIID entry.")

        if wait_group in self._groups:
            self._groups[wait_group].update({dev.name: DIID for dev in devices})
        else:
            self._groups[wait_group] = {dev.name: DIID for dev in devices}

    def _wait_for_devices(self, instr: DeviceMsg) -> None:
        wait_type = instr.content["parameter"].get("type")

        if wait_type == "move":
            self._wait_for_idle(instr)
        elif wait_type == "read":
            self._wait_for_read(instr)
        elif wait_type == "trigger":
            self._wait_for_trigger(instr)
        else:
            logger.error("Unknown wait command")
            raise DeviceMessageError("Unknown wait command")

    def _get_device_status(self, msg_cls: MessageEndpoints, devices: list) -> list:
        pipe = self.device_manager.producer.pipeline()
        for dev in devices:
            self.device_manager.producer.get(msg_cls(dev), pipe)
        return pipe.execute()

    def _check_for_failed_movements(self, device_status: list, devices: list, instr: DeviceMsg):
        if all(dev.content["success"] for dev in device_status):
            return
        ind = [dev.content["success"] for dev in device_status].index(False)
        failed_device = devices[ind]

        # make sure that this is not an old message
        matching_DIID = device_status[ind].metadata.get("DIID") == devices[ind][1]
        matching_RID = device_status[ind].metadata.get("RID") == instr.metadata["RID"]
        if matching_DIID and matching_RID:
            last_pos = BECMessage.DeviceMessage.loads(
                self.device_manager.producer.get(MessageEndpoints.device_readback(failed_device[0]))
            ).content["signals"][failed_device[0]]["value"]
            self.connector.raise_alarm(
                severity=Alarms.MAJOR,
                source=instr.content,
                content=f"Movement of device {failed_device[0]} failed whilst trying to reach the target position. Last recorded position: {last_pos}",
                alarm_type="MovementFailed",
                metadata=instr.metadata,
            )
            raise ScanAbortion

    def _wait_for_idle(self, instr: DeviceMsg) -> None:
        """Wait for devices to become IDLE

        Args:
            instr (DeviceInstructionMessage): Device instruction received from the scan assembler
        """
        start = datetime.datetime.now()

        wait_group = instr.content["parameter"].get("wait_group")

        if not wait_group or wait_group not in self._groups:
            return

        group_devices = [dev.name for dev in self._get_devices_from_instruction(instr)]
        wait_group_devices = [
            (dev_name, DIID)
            for dev_name, DIID in self._groups[wait_group].items()
            if dev_name in group_devices
        ]

        logger.debug(f"Waiting for devices: {wait_group}")

        while True:
            device_status = self._get_device_status(
                MessageEndpoints.device_req_status,
                [dev for dev, _ in wait_group_devices],
            )
            self._check_for_interruption()

            if None in device_status:
                continue

            device_status = [BECMessage.DeviceReqStatusMessage.loads(dev) for dev in device_status]
            devices_moved_successfully = all(dev.content["success"] for dev in device_status)
            matching_scanID = all(
                dev.metadata.get("scanID") == instr.metadata["scanID"] for dev in device_status
            )
            matching_requestID = all(
                dev.metadata.get("RID") == instr.metadata["RID"] for dev in device_status
            )
            matching_DIID = all(
                dev.metadata.get("DIID") == wait_group_devices[ii][1]
                for ii, dev in enumerate(device_status)
            )

            if (
                devices_moved_successfully
                and matching_scanID
                and matching_DIID
                and matching_requestID
            ):
                break

            if not devices_moved_successfully:
                self._check_for_failed_movements(device_status, wait_group_devices, instr)

        self._groups[wait_group] = {
            dev: DIID for dev, DIID in self._groups[wait_group].items() if dev not in group_devices
        }
        logger.debug("Finished waiting")
        logger.debug(datetime.datetime.now() - start)

    def _wait_for_read(self, instr: DeviceMsg) -> None:
        start = datetime.datetime.now()

        wait_group = instr.content["parameter"].get("wait_group")

        if not wait_group or wait_group not in self._groups:
            return

        group_devices = [dev.name for dev in self._get_devices_from_instruction(instr)]
        wait_group_devices = [
            (dev_name, DIID)
            for dev_name, DIID in self._groups[wait_group].items()
            if dev_name in group_devices
        ]

        logger.debug(f"Waiting for devices: {wait_group}")

        while True:
            device_status = self._get_device_status(
                MessageEndpoints.device_status, [dev for dev, _ in wait_group_devices]
            )
            self._check_for_interruption()
            device_status = [BECMessage.DeviceStatusMessage.loads(dev) for dev in device_status]

            if None in device_status:
                continue
            devices_are_idle = all(
                DeviceStatus(dev.content.get("status")) == DeviceStatus.IDLE
                for dev in device_status
            )
            matching_scanID = all(
                dev.metadata.get("scanID") == instr.metadata["scanID"] for dev in device_status
            )
            matching_DIID = all(
                dev.metadata.get("DIID") == wait_group_devices[ii][1]
                for ii, dev in enumerate(device_status)
            )
            if devices_are_idle and matching_scanID and matching_DIID:
                break
            # time.sleep(1e-4)

        self._groups[wait_group] = {
            dev: DIID for dev, DIID in self._groups[wait_group].items() if dev not in group_devices
        }
        logger.debug("Finished waiting")
        logger.debug(datetime.datetime.now() - start)

    def _wait_for_stage(self, staged: bool, devices: list, metadata: dict) -> None:
        while True:
            stage_status = self._get_device_status(MessageEndpoints.device_staged, devices)
            self._check_for_interruption()
            device_status = [BECMessage.DeviceStatusMessage.loads(dev) for dev in stage_status]

            if None in device_status:
                continue
            devices_are_ready = all(
                bool(dev.content.get("status")) == staged for dev in device_status
            )
            matching_scanID = all(
                dev.metadata.get("scanID") == metadata["scanID"] for dev in device_status
            )
            matching_DIID = all(
                dev.metadata.get("DIID") == metadata["DIID"] for dev in device_status
            )
            if devices_are_ready and matching_scanID and matching_DIID:
                break

    def _wait_for_device_server(self) -> None:
        self.parent.wait_for_service("DeviceServer")

    def _set_devices(self, instr: DeviceMsg) -> None:
        """Send device instruction to set a device to a specific value

        Args:
            instr (DeviceInstructionMessage): Device instruction received from the scan assembler
        """

        # send instruction
        self.device_manager.producer.send(MessageEndpoints.device_instructions(), instr.dumps())

    def _trigger_devices(self, instr: DeviceMsg) -> None:
        devices = [dev.name for dev in self.device_manager.devices.detectors()]
        self._last_trigger = instr
        self.device_manager.producer.send(
            MessageEndpoints.device_instructions(),
            DeviceMsg(
                device=devices,
                action="trigger",
                parameter=instr.content["parameter"],
                metadata=instr.metadata,
            ).dumps(),
        )

    def _wait_for_trigger(self, instr: DeviceMsg) -> None:
        trigger_time = float(instr.content["parameter"]["time"]) * self.current_scan_info.get(
            "frames_per_trigger", 1
        )
        time.sleep(trigger_time)
        devices = [dev.name for dev in self.device_manager.devices.detectors()]
        metadata = self._last_trigger.metadata
        while True:
            stage_status = self._get_device_status(MessageEndpoints.device_req_status, devices)
            self._check_for_interruption()
            device_status = [BECMessage.DeviceReqStatusMessage.loads(dev) for dev in stage_status]

            if None in device_status:
                continue
            devices_are_ready = all(
                bool(dev.content.get("success")) is True for dev in device_status
            )
            matching_scanID = all(
                dev.metadata.get("scanID") == metadata["scanID"] for dev in device_status
            )
            matching_DIID = all(
                dev.metadata.get("DIID") == metadata["DIID"] for dev in device_status
            )
            if devices_are_ready and matching_scanID and matching_DIID:
                break

    def _pre_scan(self, instr: DeviceMsg) -> None:
        devices = [dev.name for dev in self.device_manager.devices.enabled_devices]
        self.device_manager.producer.send(
            MessageEndpoints.device_instructions(),
            DeviceMsg(
                device=devices,
                action="pre_scan",
                parameter=instr.content["parameter"],
                metadata=instr.metadata,
            ).dumps(),
        )
        self._wait_for_pre_scan(instr)

    def _wait_for_pre_scan(self, instr: DeviceMsg) -> None:
        devices = [dev.name for dev in self.device_manager.devices.enabled_devices]
        metadata = instr.metadata
        while True:
            stage_status = self._get_device_status(MessageEndpoints.device_req_status, devices)
            self._check_for_interruption()
            device_status = [BECMessage.DeviceReqStatusMessage.loads(dev) for dev in stage_status]

            if None in device_status:
                continue
            devices_are_ready = all(
                bool(dev.content.get("success")) is True for dev in device_status
            )
            matching_scanID = all(
                dev.metadata.get("scanID") == metadata["scanID"] for dev in device_status
            )
            matching_DIID = all(
                dev.metadata.get("DIID") == metadata["DIID"] for dev in device_status
            )
            if devices_are_ready and matching_scanID and matching_DIID:
                break

    def _send_rpc(self, instr: DeviceMsg) -> None:
        self.device_manager.producer.send(MessageEndpoints.device_instructions(), instr.dumps())

    def _read_devices(self, instr: DeviceMsg) -> None:
        if instr.metadata.get("cached"):
            self._publish_readback(instr)
            return

        producer = self.device_manager.producer

        devices = instr.content.get("device")
        if devices is None:
            devices = [
                dev.name
                for dev in self.device_manager.devices.monitored_devices(
                    readout_priority=self.readout_priority
                )
            ]
        producer.send(
            MessageEndpoints.device_instructions(),
            DeviceMsg(
                device=devices,
                action="read",
                parameter=instr.content["parameter"],
                metadata=instr.metadata,
            ).dumps(),
        )
        return

    def _publish_readback(self, instr: DeviceMsg, devices: list = None) -> None:
        producer = self.device_manager.producer
        if not devices:
            devices = instr.content.get("device")

        # cached readout
        readouts = self._get_readback(devices)
        pipe = producer.pipeline()
        for readout, device in zip(readouts, devices):
            msg = BECMessage.DeviceMessage(signals=readout, metadata=instr.metadata).dumps()
            producer.set_and_publish(
                MessageEndpoints.device_read(device),
                msg,
                pipe,
            )
        return pipe.execute()

    def _get_readback(self, devices: list) -> list:
        producer = self.device_manager.producer
        # cached readout
        pipe = producer.pipeline()
        for dev in devices:
            producer.get(MessageEndpoints.device_readback(dev), pipe=pipe)
        return pipe.execute()

    def _publish_data_as_read(self, instr: DeviceMsg):
        producer = self.device_manager.producer
        data = instr.content["parameter"]["data"]
        devices = instr.content["device"]
        if not isinstance(devices, list):
            devices = [devices]
        if not isinstance(data, list):
            data = [data]
        for device, dev_data in zip(devices, data):
            msg = BECMessage.DeviceMessage(signals=dev_data, metadata=instr.metadata).dumps()
            producer.set_and_publish(MessageEndpoints.device_read(device), msg)

    def _kickoff_devices(self, instr: DeviceMsg) -> None:
        # logger.info("kickoff")
        self.device_manager.producer.send(
            MessageEndpoints.device_instructions(),
            DeviceMsg(
                device=instr.content.get("device"),
                action="kickoff",
                parameter=instr.content["parameter"],
                metadata=instr.metadata,
            ).dumps(),
        )

    def _complete_devices(self, instr: DeviceMsg) -> None:
        self.device_manager.producer.send(
            MessageEndpoints.device_instructions(),
            DeviceMsg(
                device=instr.content.get("device"),
                action="complete",
                parameter=instr.content["parameter"],
                metadata=instr.metadata,
            ).dumps(),
        )

    def _baseline_reading(self, instr: DeviceMsg) -> None:
        baseline_devices = [
            dev.name
            for dev in self.device_manager.devices.baseline_devices(
                readout_priority=self.readout_priority
            )
        ]
        params = instr.content["parameter"]
        self.device_manager.producer.send(
            MessageEndpoints.device_instructions(),
            DeviceMsg(
                device=baseline_devices,
                action="read",
                parameter=params,
                metadata=instr.metadata,
            ).dumps(),
        )

    def _check_for_interruption(self) -> None:
        if self.status == InstructionQueueStatus.PAUSED:
            self._send_scan_status("paused")
        while self.status == InstructionQueueStatus.PAUSED:
            time.sleep(0.1)
        if self.status == InstructionQueueStatus.STOPPED:
            raise ScanAbortion

    def _open_scan(self, instr: DeviceMsg) -> None:
        if not self.scan_id:
            self.scan_id = instr.metadata.get("scanID")
            if instr.content["parameter"].get("scan_motors") is not None:
                self.scan_motors = [
                    self.device_manager.devices[dev]
                    for dev in instr.content["parameter"].get("scan_motors")
                ]
                self.readout_priority = instr.content["parameter"].get("readout_priority", {})
            self.scan_type = instr.content["parameter"].get("scan_type")

        if not instr.metadata.get("scan_def_id"):
            self.max_point_id = 0
        num_points = self.max_point_id + instr.content["parameter"]["num_points"]
        if self.max_point_id:
            num_points += 1

        active_rb = self.current_instruction_queue_item.active_request_block

        self._initialize_scan_info(active_rb, instr, num_points)

        # only append the table_wait if the scan is not using scan_progress
        if not self.scan_report_instructions or not self.scan_report_instructions[-1].get(
            "scan_progress"
        ):
            self.scan_report_instructions.append({"table_wait": num_points})
        self.current_instruction_queue_item.parent.queue_manager.send_queue_status()

        self._send_scan_status("open")

    def _initialize_scan_info(self, active_rb: RequestBlock, instr: DeviceMsg, num_points: int):
        metadata = active_rb.metadata
        self.current_scan_info = {**instr.metadata, **instr.content["parameter"]}
        self.current_scan_info.update(metadata)
        self.current_scan_info.update(
            {
                "scan_number": self.parent.scan_number,
                "dataset_number": self.parent.dataset_number,
                "exp_time": self._exposure_time,
                "frames_per_trigger": active_rb.scan.frames_per_trigger,
                "settling_time": active_rb.scan.settling_time,
                "readout_time": active_rb.scan.readout_time,
                "acquisition_config": active_rb.scan.acquisition_config,
                "scan_report_hint": active_rb.scan.scan_report_hint,
                "scan_report_devices": active_rb.scan.scan_report_devices,
                "enforce_sync": active_rb.scan.enforce_sync,
                "num_points": num_points,
            }
        )
        self.current_scan_info["scan_msgs"] = [
            str(scan_msg) for scan_msg in self.current_instruction_queue_item.scan_msgs
        ]

    def _process_scan_report_instruction(self, instr):
        self.scan_report_instructions.append(instr.content["parameter"])
        self.current_instruction_queue_item.parent.queue_manager.send_queue_status()

    def _close_scan(self, instr: DeviceMsg, max_point_id: int) -> None:
        scan_id = instr.metadata.get("scanID")

        if self.scan_id != scan_id:
            return

        # reset the scan ID now that the scan will be closed
        self.scan_id = None

        scan_info = self.current_scan_info
        if scan_info.get("scan_type") == "fly":
            # flyers do not increase the point_id but instead set the num_points directly
            num_points = self.current_instruction_queue_item.active_request_block.scan.num_pos
            self.current_scan_info["num_points"] = num_points

        else:
            # point_id starts at 0
            scan_info["num_points"] = max_point_id + 1

        self._send_scan_status("closed")

    def _stage_devices(self, instr: DeviceMsg) -> None:
        devices = [dev.name for dev in self.device_manager.devices.enabled_devices]
        self._staged_devices.update(devices)
        self.device_manager.producer.send(
            MessageEndpoints.device_instructions(),
            DeviceMsg(
                device=devices,
                action="stage",
                parameter=instr.content["parameter"],
                metadata=instr.metadata,
            ).dumps(),
        )
        self._wait_for_stage(staged=True, devices=devices, metadata=instr.metadata)

    def _unstage_devices(
        self, instr: DeviceMsg = None, devices: list = None, cleanup=False
    ) -> None:
        if not devices:
            devices = [dev.name for dev in self.device_manager.devices.enabled_devices]
        parameter = {} if not instr else instr.content["parameter"]
        metadata = {} if not instr else instr.metadata
        self._staged_devices.difference_update(devices)
        self.device_manager.producer.send(
            MessageEndpoints.device_instructions(),
            DeviceMsg(
                device=devices,
                action="unstage",
                parameter=parameter,
                metadata=metadata,
            ).dumps(),
        )
        if not cleanup:
            self._wait_for_stage(staged=False, devices=devices, metadata=metadata)

    def _send_scan_status(self, status: str):
        current_scan_info_print = self.current_scan_info.copy()
        if current_scan_info_print.get("positions", []):
            current_scan_info_print["positions"] = "..."
        logger.info(
            f"New scan status: {self.current_scanID} / {status} / {current_scan_info_print}"
        )
        msg = ScanStatusMsg(
            scanID=self.current_scanID,
            status=status,
            info=self.current_scan_info,
        ).dumps()
        expire = None if status in ["open", "paused"] else 1800
        pipe = self.device_manager.producer.pipeline()
        self.device_manager.producer.set(
            MessageEndpoints.public_scan_info(self.current_scanID),
            msg,
            pipe=pipe,
            expire=expire,
        )
        self.device_manager.producer.set_and_publish(MessageEndpoints.scan_status(), msg, pipe=pipe)
        pipe.execute()

    def _process_instructions(self, queue: InstructionQueueItem) -> None:
        """
        Process scan instructions and send DeviceInstructions to OPAAS.
        For now this is an in-memory communication. In the future however,
        we might want to pass it through a dedicated Kafka topic.
        Args:
            queue: instruction queue

        Returns:

        """
        self.current_instruction_queue_item = queue

        start = time.time()
        self.max_point_id = 0

        # make sure the device server is ready to receive data
        self._wait_for_device_server()

        queue.is_active = True
        try:
            for instr in queue:
                if instr is None:
                    continue
                self._check_for_interruption()
                self._exposure_time = getattr(queue.active_request_block.scan, "exp_time", None)
                self._instruction_step(instr)
        except ScanAbortion as exc:
            self._groups = {}
            if queue.stopped or not (queue.return_to_start and queue.active_request_block):
                raise ScanAbortion from exc
            queue.stopped = True
            cleanup = queue.active_request_block.scan.return_to_start()
            self.status = InstructionQueueStatus.RUNNING
            for instr in cleanup:
                self._check_for_interruption()
                instr.metadata["scanID"] = queue.queue.active_rb.scanID
                instr.metadata["queueID"] = queue.queue_id
                self._instruction_step(instr)
            raise ScanAbortion from exc
        queue.is_active = False
        queue.status = InstructionQueueStatus.COMPLETED
        self.current_instruction_queue_item = None

        logger.info(f"QUEUE ITEM finished after {time.time()-start:.2f} seconds")
        self.reset()

    def _instruction_step(self, instr: DeviceMsg):
        logger.debug(instr)
        action = instr.content.get("action")
        scan_def_id = instr.metadata.get("scan_def_id")
        if self.current_scanID != instr.metadata.get("scanID"):
            self.current_scanID = instr.metadata.get("scanID")

        if "pointID" in instr.metadata:
            self.max_point_id = instr.metadata["pointID"]

        self._add_wait_group(instr)

        logger.debug(f"Device instruction: {instr}")
        self._check_for_interruption()

        if action == "open_scan":
            self._open_scan(instr)
        elif action == "close_scan" and scan_def_id is None:
            self._close_scan(instr, self.max_point_id)
        elif action == "open_scan_def":
            pass
        elif action == "close_scan_def":
            self._close_scan(instr, self.max_point_id)
        elif action == "wait":
            self._wait_for_devices(instr)
        elif action == "trigger":
            self._trigger_devices(instr)
        elif action == "set":
            self._set_devices(instr)
        elif action == "read":
            self._read_devices(instr)
        elif action == "kickoff":
            self._kickoff_devices(instr)
        elif action == "complete":
            self._complete_devices(instr)
        elif action == "baseline_reading":
            self._baseline_reading(instr)
        elif action == "rpc":
            self._send_rpc(instr)
        elif action == "stage":
            self._stage_devices(instr)
        elif action == "unstage":
            self._unstage_devices(instr)
        elif action == "pre_scan":
            self._pre_scan(instr)
        elif action == "publish_data_as_read":
            self._publish_data_as_read(instr)
        elif action == "scan_report_instruction":
            self._process_scan_report_instruction(instr)

        else:
            logger.warning(f"Unknown device instruction: {instr}")

    def reset(self):
        """reset the scan worker and its member variables"""
        self._groups = {}
        self.current_scanID = ""
        self.current_scan_info = {}
        self.scan_id = None
        self.interception_msg = None
        self.scan_motors = []

    def cleanup(self):
        """perform cleanup instructions"""
        self._unstage_devices(devices=list(self._staged_devices), cleanup=True)

    def run(self):
        try:
            while not self.signal_event.is_set():
                try:
                    for queue in self.parent.queue_manager.queues[self.queue_name]:
                        self._process_instructions(queue)
                        if not queue.stopped:
                            queue.append_to_queue_history()

                except ScanAbortion:
                    queue.queue.increase_scan_number()
                    if queue.return_to_start:
                        self._send_scan_status("aborted")
                    else:
                        self._send_scan_status("halted")
                    queue.status = InstructionQueueStatus.STOPPED
                    queue.append_to_queue_history()
                    self.cleanup()
                    self.parent.queue_manager.queues[self.queue_name].abort()
                    self.reset()

        # pylint: disable=broad-except
        except Exception as exc:
            content = traceback.format_exc()
            logger.error(content)
            self.connector.raise_alarm(
                severity=Alarms.MAJOR,
                source="ScanWorker",
                content=content,
                alarm_type=exc.__class__.__name__,
                metadata={},
            )
            self.parent.queue_manager.queues[self.queue_name].abort()
            self.reset()
        finally:
            self.connector.shutdown()

    def shutdown(self):
        """shutdown the scan worker"""
        self.signal_event.set()
        self.join()
