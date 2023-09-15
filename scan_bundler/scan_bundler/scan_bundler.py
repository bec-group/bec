import collections
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from bec_lib.core import BECMessage, BECService, BECStatus
from bec_lib.core import DeviceManagerBase as DeviceManager
from bec_lib.core import MessageEndpoints, bec_logger
from bec_lib.core.connector import ConnectorBase

from .bec_emitter import BECEmitter
from .bluesky_emitter import BlueskyEmitter

logger = bec_logger.logger


class ScanBundler(BECService):
    def __init__(self, config, connector_cls: ConnectorBase) -> None:
        super().__init__(config, connector_cls, unique_service=True)

        self.device_manager = None
        self._start_device_manager()
        self._start_device_read_consumer()
        self._start_scan_queue_consumer()
        self._start_scan_status_consumer()

        self.sync_storage = {}
        self.monitored_devices = {}
        self.baseline_devices = {}
        self.device_storage = {}
        self.scan_motors = {}
        self.readout_priority = {}
        self.storage_initialized = set()
        self.current_queue = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.executor_tasks = collections.deque(maxlen=100)
        self.scanID_history = collections.deque(maxlen=10)
        self._lock = threading.Lock()
        self._emitter = []
        self._initialize_emitters()
        self.status = BECStatus.RUNNING

    def _initialize_emitters(self):
        self._emitter = [BECEmitter(self), BlueskyEmitter(self)]

    def run_emitter(self, emitter_method: Callable, *args, **kwargs):
        for emi in self._emitter:
            try:
                getattr(emi, emitter_method)(*args, **kwargs)
            except Exception:
                content = traceback.format_exc()
                logger.error(f"Failed to run emitter: {content}")

    def _start_device_manager(self):
        self.device_manager = DeviceManager(self.connector)
        self.device_manager.initialize(self.bootstrap_server)

    def _start_device_read_consumer(self):
        self._device_read_consumer = self.connector.consumer(
            pattern=MessageEndpoints.device_read("*"),
            cb=self._device_read_callback,
            parent=self,
            name="device_read_consumer",
        )
        self._device_read_consumer.start()

    def _start_scan_queue_consumer(self):
        self._scan_queue_consumer = self.connector.consumer(
            MessageEndpoints.scan_queue_status(),
            cb=self._scan_queue_callback,
            group_id="scan_bundler",
            parent=self,
            name="scan_queue_consumer",
        )
        self._scan_queue_consumer.start()

    def _start_scan_status_consumer(self):
        self._scan_status_consumer = self.connector.consumer(
            MessageEndpoints.scan_status(),
            cb=self._scan_status_callback,
            group_id="scan_bundler",
            parent=self,
            name="scan_status_consumer",
        )
        self._scan_status_consumer.start()

    @staticmethod
    def _device_read_callback(msg, parent, **_kwargs):
        # pylint: disable=protected-access
        dev = msg.topic.decode().split(MessageEndpoints._device_read + "/")[-1].split(":sub")[0]
        msgs = BECMessage.DeviceMessage.loads(msg.value)
        logger.debug(f"Received reading from device {dev}")
        if not isinstance(msgs, list):
            msgs = [msgs]
        task = parent.executor.submit(parent._add_device_to_storage, msgs, dev)
        parent.executor_tasks.append(task)

    @staticmethod
    def _scan_queue_callback(msg, parent, **_kwargs):
        msg = BECMessage.ScanQueueStatusMessage.loads(msg.value)
        logger.trace(msg)
        parent.current_queue = msg.content["queue"]["primary"].get("info")

    @staticmethod
    def _scan_status_callback(msg, parent, **_kwargs):
        msg = BECMessage.ScanStatusMessage.loads(msg.value)
        parent.handle_scan_status_message(msg)

    def handle_scan_status_message(self, msg: BECMessage.ScanStatusMessage) -> None:
        """handle scan status messages"""
        logger.info(f"Received new scan status: {msg}")
        scanID = msg.content["scanID"]
        self.cleanup_storage()
        if not scanID in self.sync_storage:
            self._initialize_scan_container(msg)
            if scanID not in self.scanID_history:
                self.scanID_history.append(scanID)
        if msg.content.get("status") != "open":
            self._scan_status_modification(msg)

    def _scan_status_modification(self, msg: BECMessage.ScanStatusMessage):
        status = msg.content.get("status")
        if status not in ["closed", "aborted", "paused", "halted"]:
            logger.error(f"Unknown scan status {status}")
            return

        scanID = msg.content.get("scanID")
        if not scanID:
            logger.warning(f"Received scan status update without scanID: {msg}")
            return
        if self.sync_storage.get(scanID):
            self.sync_storage[scanID]["status"] = status
        else:
            self.sync_storage[scanID] = {"info": {}, "status": status, "sent": set()}
            self.storage_initialized.add(scanID)
            if scanID not in self.scanID_history:
                self.scanID_history.append(scanID)

    def _initialize_scan_container(self, scan_msg: BECMessage.ScanStatusMessage):
        if scan_msg.content.get("status") != "open":
            return

        scanID = scan_msg.content["scanID"]
        scan_info = scan_msg.content["info"]
        scan_motors = list(set(self.device_manager.devices[m] for m in scan_info["scan_motors"]))
        self.scan_motors[scanID] = scan_motors
        self.readout_priority[scanID] = scan_info["readout_priority"]
        if not scanID in self.storage_initialized:
            self.sync_storage[scanID] = {"info": scan_info, "status": "open", "sent": set()}
            self.monitored_devices[scanID] = {
                "devices": self.device_manager.devices.monitored_devices(
                    readout_priority=self.readout_priority[scanID]
                ),
                "pointID": {},
            }
            self.baseline_devices[scanID] = {
                "devices": self.device_manager.devices.baseline_devices(
                    readout_priority=self.readout_priority[scanID]
                ),
                "done": {
                    dev.name: False
                    for dev in self.device_manager.devices.baseline_devices(
                        readout_priority=self.readout_priority[scanID]
                    )
                },
            }
            self.storage_initialized.add(scanID)
            self.run_emitter("on_init", scanID)
            return

    def _step_scan_update(self, scanID, device, signal, metadata):
        if "pointID" not in metadata:
            return
        with self._lock:
            dev = {device: signal}
            pointID = metadata["pointID"]
            monitored_devices = self.monitored_devices[scanID]

            self.sync_storage[scanID][pointID] = {
                **self.sync_storage[scanID].get(pointID, {}),
                **dev,
            }

            if monitored_devices["pointID"].get(pointID) is None:
                monitored_devices["pointID"][pointID] = {
                    dev.name: False for dev in monitored_devices["devices"]
                }
            monitored_devices["pointID"][pointID][device] = True

            monitored_devices_completed = list(monitored_devices["pointID"][pointID].values())

            all_monitored_devices_completed = bool(
                all(monitored_devices_completed)
                and (
                    len(monitored_devices_completed)
                    == len(self.monitored_devices[scanID]["devices"])
                )
            )

            if all_monitored_devices_completed and self.sync_storage[scanID].get(pointID):
                self._update_monitor_signals(scanID, pointID)
                self._send_scan_point(scanID, pointID)

    def _fly_scan_update(self, scanID, device, signal, metadata):
        if "pointID" not in metadata:
            return
        with self._lock:
            dev = {device: signal}
            pointID = metadata["pointID"]
            if self.sync_storage[scanID].get("info", {}).get("enforce_sync") is False:
                monitored_devices = self.monitored_devices[scanID]

                self.sync_storage[scanID][pointID] = {
                    **self.sync_storage[scanID].get(pointID, {}),
                    **dev,
                }

                if monitored_devices["pointID"].get(pointID) is None:
                    monitored_devices["pointID"][pointID] = {
                        dev.name: False for dev in monitored_devices["devices"]
                    }
                monitored_devices["pointID"][pointID][device] = True

                monitored_devices_completed = list(monitored_devices["pointID"][pointID].values())

                all_monitored_devices_completed = bool(
                    all(monitored_devices_completed)
                    and (
                        len(monitored_devices_completed)
                        == len(self.monitored_devices[scanID]["devices"])
                    )
                )
                if all_monitored_devices_completed and self.sync_storage[scanID].get(pointID):
                    self._update_monitor_signals(scanID, pointID)
                    self._send_scan_point(scanID, pointID)
            else:
                self.sync_storage[scanID][pointID] = {
                    **self.sync_storage[scanID].get(pointID, {}),
                    **signal,
                }

                if self.sync_storage[scanID].get(pointID):
                    self._update_monitor_signals(scanID, pointID)
                    self._send_scan_point(scanID, pointID)

    def _baseline_update(self, scanID, device, signal):
        with self._lock:
            dev = {device: signal}
            baseline_devices_status = self.baseline_devices[scanID]["done"]
            baseline_devices_status[device] = True

            self.sync_storage[scanID]["baseline"] = {
                **self.sync_storage[scanID].get("baseline", {}),
                **dev,
            }

            if not all(status for status in baseline_devices_status.values()):
                return

            logger.info(f"Sending baseline readings for scanID {scanID}.")
            logger.debug("Baseline: ", self.sync_storage[scanID]["baseline"])
            self.run_emitter("on_baseline_emit", scanID)
            self.baseline_devices[scanID]["done"] = {
                dev.name: False
                for dev in self.device_manager.devices.baseline_devices(
                    readout_priority=self.readout_priority[scanID]
                )
            }

    def _get_scan_status_history(self, length):
        return [
            BECMessage.ScanStatusMessage.loads(msg)
            for msg in self.producer.lrange(
                MessageEndpoints.scan_status() + "_list", length * -1, -1
            )
        ]

    def _wait_for_scanID(self, scanID, timeout_time=10):
        elapsed_time = 0
        while not scanID in self.storage_initialized:
            msgs = self._get_scan_status_history(5)
            for msg in msgs:
                if msg.content["scanID"] == scanID:
                    self.handle_scan_status_message(msg)
            if scanID in self.sync_storage:
                if self.sync_storage[scanID]["status"] in ["closed", "aborted"]:
                    logger.info(
                        f"Received reading for {self.sync_storage[scanID]['status']} scan {scanID}."
                    )
                    return
            time.sleep(0.05)
            elapsed_time += 0.05
            if elapsed_time > timeout_time:
                raise TimeoutError(f"Reached timeout whilst waiting for scanID {scanID} to appear.")

    def _add_device_to_storage(self, msgs, device, timeout_time=10):
        for msg in msgs:
            metadata = msg.metadata

            scanID = metadata.get("scanID")
            if not scanID:
                logger.error("Received device message without scanID")
                return

            signal = msg.content.get("signals")
            if not signal:
                logger.error("Received device message without signals")
                return

            try:
                self._wait_for_scanID(scanID, timeout_time=timeout_time)
            except TimeoutError:
                logger.warning(f"Could not find a matching scanID {scanID} in sync_storage.")
                return

            if self.sync_storage[scanID]["status"] in ["aborted", "closed"]:
                # check if the sync_storage has been initialized properly.
                # In case of post-scan initialization, scan info is not available
                if not self.sync_storage[scanID]["info"].get("scan_type"):
                    return
            self.device_storage[device] = signal
            readout_priority = metadata.get("readout_priority")
            if readout_priority == "monitored":
                if self.sync_storage[scanID]["info"]["scan_type"] == "step":
                    self._step_scan_update(scanID, device, signal, metadata)
                elif self.sync_storage[scanID]["info"]["scan_type"] == "fly":
                    self._fly_scan_update(scanID, device, signal, metadata)
                else:
                    raise RuntimeError(
                        f"Unknown scan type {self.sync_storage[scanID]['info']['scan_type']}"
                    )

            elif readout_priority == "baseline":
                self._baseline_update(scanID, device, signal)

    def _update_monitor_signals(self, scanID, pointID) -> None:
        if self.sync_storage[scanID]["info"]["scan_type"] == "fly":
            # for fly scans, take all primary and monitor signals
            devices = self.monitored_devices[scanID]["devices"]

            readings = self._get_last_device_readback(devices)

            for read, dev in zip(readings, devices):
                self.sync_storage[scanID][pointID][dev.name] = read

    def _get_last_device_readback(self, devices: list) -> list:
        pipe = self.producer.pipeline()
        for dev in devices:
            self.producer.get(MessageEndpoints.device_readback(dev.name), pipe)
        read_raw = pipe.execute()

        return [BECMessage.DeviceMessage.loads(read).content["signals"] for read in read_raw]

    def cleanup_storage(self):
        """remove old scanIDs to free memory"""
        remove_scanIDs = []
        for scanID, entry in self.sync_storage.items():
            if entry.get("status") not in ["closed", "aborted"]:
                continue
            if scanID in self.scanID_history:
                continue
            remove_scanIDs.append(scanID)

        for scanID in remove_scanIDs:
            for storage in [
                "sync_storage",
                "monitored_devices",
                "baseline_devices",
                "scan_motors",
                "readout_priority",
            ]:
                try:
                    getattr(self, storage).pop(scanID)
                except KeyError:
                    logger.warning(f"Failed to remove {scanID} from {storage}.")
            # self.bluesky_emitter.cleanup_storage(scanID)
            self.run_emitter("on_cleanup", scanID)
            self.storage_initialized.remove(scanID)

    def _send_scan_point(self, scanID, pointID) -> None:
        logger.info(f"Sending point {pointID} for scanID {scanID}.")
        logger.debug(f"{pointID}, {self.sync_storage[scanID][pointID]}")

        self.run_emitter("on_scan_point_emit", scanID, pointID)

        if not pointID in self.sync_storage[scanID]["sent"]:
            self.sync_storage[scanID]["sent"].add(pointID)
        else:
            logger.warning(f"Resubmitting existing pointID {pointID} for scanID {scanID}")

    def shutdown(self):
        self.device_manager.shutdown()
