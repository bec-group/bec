import inspect
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from typing import Any

import ophyd
from bec_lib.core import Alarms, BECMessage, BECService, MessageEndpoints, bec_logger
from bec_lib.core.BECMessage import BECStatus
from bec_lib.core.connector import ConnectorBase
from bec_lib.core.devicemanager import OnFailure
from ophyd import Staged
from ophyd.utils import errors as ophyd_errors

from device_server.devices import is_serializable, rgetattr
from device_server.devices.devicemanager import DeviceManagerDS

logger = bec_logger.logger

consumer_stop = threading.Event()


class DisabledDeviceError(Exception):
    pass


class InvalidDeviceError(Exception):
    pass


class DeviceServer(BECService):
    """DeviceServer using ophyd as a service
    This class is intended to provide a thin wrapper around ophyd and the devicemanager. It acts as the entry point for other services
    """

    def __init__(self, config, connector_cls: ConnectorBase) -> None:
        super().__init__(config, connector_cls, unique_service=True)
        self._tasks = []
        self.device_manager = None
        self.threads = []
        self.sig_thread = None
        self.sig_thread = self.connector.consumer(
            MessageEndpoints.scan_queue_modification(),
            cb=self.consumer_interception_callback,
            parent=self,
        )
        self.sig_thread.start()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._start_device_manager()

    def _start_device_manager(self):
        self.device_manager = DeviceManagerDS(self.connector, status_cb=self.update_status)
        self.device_manager.initialize(self.bootstrap_server)

    def start(self) -> None:
        """start the device server"""
        if consumer_stop.is_set():
            consumer_stop.clear()

        self.threads = [
            self.connector.consumer(
                MessageEndpoints.device_instructions(),
                event=consumer_stop,
                cb=self.instructions_callback,
                parent=self,
            ),
        ]
        for thread in self.threads:
            thread.start()
        self.status = BECStatus.RUNNING

    def update_status(self, status: BECStatus):
        self.status = status

    def stop(self) -> None:
        """stop the device server"""
        consumer_stop.set()
        for thread in self.threads:
            thread.join()
        self.status = BECStatus.IDLE

    def shutdown(self) -> None:
        """shutdown the device server"""
        super().shutdown()
        self.stop()
        self.sig_thread.signal_event.set()
        self.sig_thread.join()
        self.device_manager.shutdown()

    def _update_device_metadata(self, instr) -> None:
        devices = instr.content["device"]
        if not isinstance(devices, list):
            devices = [devices]
        for dev in devices:
            self.device_manager.devices.get(dev).metadata = instr.metadata

    @staticmethod
    def consumer_interception_callback(msg, *, parent, **_kwargs) -> None:
        """callback for receiving scan modifications / interceptions"""
        mvalue = BECMessage.ScanQueueModificationMessage.loads(msg.value)
        if mvalue is None:
            logger.warning("Failed to parse scan queue modification message.")
            return
        logger.info(f"Receiving: {mvalue.content}")
        if mvalue.content.get("action") in ["pause", "abort"]:
            parent.stop_devices()

    def stop_devices(self) -> None:
        """stop all enabled devices"""
        logger.info("Stopping devices after receiving 'abort' request.")
        self.status = BECStatus.BUSY
        for dev in self.device_manager.devices.enabled_devices:
            if not dev.enabled_set:
                # don't stop devices that we haven't set
                continue
            if hasattr(dev.obj, "stop"):
                dev.obj.stop()
        self.status = BECStatus.RUNNING

    def _assert_device_is_enabled(self, instructions: BECMessage.DeviceInstructionMessage) -> None:
        devices = instructions.content["device"]

        if isinstance(devices, str):
            devices = [devices]

        for dev in devices:
            if not self.device_manager.devices[dev].enabled:
                raise DisabledDeviceError(f"Cannot access disabled device {dev}.")

    def _assert_device_is_valid(self, instructions: BECMessage.DeviceInstructionMessage) -> None:
        devices = instructions.content["device"]
        if not devices:
            raise InvalidDeviceError("At least one device must be specified.")
        if isinstance(devices, str):
            devices = [devices]
        for dev in devices:
            if dev not in self.device_manager.devices:
                raise InvalidDeviceError(f"There is no device with the name {dev}.")

    def handle_device_instructions(self, msg: str) -> None:
        """Parse a device instruction message and handle the requested action. Action
        types are set, read, rpc, kickoff or trigger.

        Args:
            msg (str): A DeviceInstructionMessage string containing the action and its parameters

        """
        action = None
        try:
            instructions = BECMessage.DeviceInstructionMessage.loads(msg)
            if not instructions.content["device"]:
                return
            action = instructions.content["action"]
            self._assert_device_is_valid(instructions)
            if action != "rpc":
                # rpc has its own error handling
                self._assert_device_is_enabled(instructions)
            self._update_device_metadata(instructions)

            if action == "set":
                self._set_device(instructions)
            elif action == "read":
                start = time.time()
                self._read_device(instructions)
                logger.info(f"finished read after {time.time()-start}")
            elif action == "rpc":
                self._run_rpc(instructions)
            elif action == "kickoff":
                self._kickoff_device(instructions)
            elif action == "complete":
                self._complete_device(instructions)
            elif action == "trigger":
                self._trigger_device(instructions)
            elif action == "stage":
                start = time.time()
                self._stage_device(instructions)
                logger.info(f"finished stage after {time.time()-start}")
            elif action == "unstage":
                self._unstage_device(instructions)
            elif action == "pre_scan":
                self._pre_scan(instructions)
            else:
                logger.warning(f"Received unknown device instruction: {instructions}")
        except ophyd_errors.LimitError as limit_error:
            content = traceback.format_exc()
            logger.error(content)
            self.connector.raise_alarm(
                severity=Alarms.MAJOR,
                source=instructions.content,
                content=content,
                alarm_type=limit_error.__class__.__name__,
                metadata=instructions.metadata,
            )
        except Exception as exc:  # pylint: disable=broad-except
            if action == "rpc":
                self._send_rpc_exception(exc, instructions)
            else:
                content = traceback.format_exc()
                self.connector.log_error({"source": msg, "message": content})
                logger.error(content)
                self.connector.raise_alarm(
                    severity=Alarms.MAJOR,
                    source=instructions.content,
                    content=content,
                    alarm_type=exc.__class__.__name__,
                    metadata=instructions.metadata,
                )

    @staticmethod
    def instructions_callback(msg, *, parent, **_kwargs) -> None:
        """callback for handling device instructions"""
        parent.executor.submit(parent.handle_device_instructions, msg.value)

    def _get_result_from_rpc(self, rpc_var: Any, instr_params: dict) -> Any:
        if callable(rpc_var):
            args = tuple(instr_params.get("args", ()))
            kwargs = instr_params.get("kwargs", {})
            if args and kwargs:
                res = rpc_var(*args, **kwargs)
            elif args:
                res = rpc_var(*args)
            elif kwargs:
                res = rpc_var(**kwargs)
            else:
                res = rpc_var()
        else:
            res = rpc_var
        if not is_serializable(res):
            if isinstance(res, ophyd.StatusBase):
                return res
            elif isinstance(res, list) and instr_params.get("func") in ["stage", "unstage"]:
                return [obj._staged for obj in res]
            res = None
            self.connector.raise_alarm(
                severity=Alarms.WARNING,
                alarm_type="TypeError",
                source=instr_params,
                content=f"Return value of rpc call {instr_params} is not serializable.",
                metadata={},
            )
        return res

    def _run_rpc(self, instr: BECMessage.DeviceInstructionMessage) -> None:
        save_stdout = sys.stdout
        result = StringIO()
        sys.stdout = result
        try:
            instr_params = instr.content.get("parameter")
            self._assert_device_is_enabled(instr)
            rpc_var = rgetattr(
                self.device_manager.devices[instr.content["device"]].obj,
                instr_params.get("func"),
            )
            res = self._get_result_from_rpc(rpc_var, instr_params)
            if isinstance(res, ophyd.StatusBase):
                res.__dict__["instruction"] = instr
                res.add_callback(self._status_callback)
                res = {
                    "type": "status",
                    "RID": instr.metadata.get("RID"),
                    "success": res.success,
                    "timeout": res.timeout,
                    "done": res.done,
                    "settle_time": res.settle_time,
                }
            elif isinstance(res, list) and isinstance(res[0], ophyd.Staged):
                res = [str(stage) for stage in res]
            else:
                print(f"return value: {res}")
            # send result to client
            self.producer.set(
                MessageEndpoints.device_rpc(instr_params.get("rpc_id")),
                BECMessage.DeviceRPCMessage(
                    device=instr.content["device"],
                    return_val=res,
                    out=result.getvalue(),
                    success=True,
                ).dumps(),
                expire=1800,
            )
            logger.trace(res)
        except KeyboardInterrupt as kbi:
            sys.stdout = save_stdout
            raise KeyboardInterrupt from kbi

        except Exception as exc:  # pylint: disable=broad-except
            # send error to client
            self._send_rpc_exception(exc, instr)

        finally:
            sys.stdout = save_stdout

    def _send_rpc_exception(self, exc: Exception, instr: BECMessage.DeviceInstructionMessage):
        exc_formatted = {
            "error": exc.__class__.__name__,
            "msg": exc.args,
            "traceback": traceback.format_exc(),
        }
        logger.info(f"Received exception: {exc_formatted}, {exc}")
        instr_params = instr.content.get("parameter")
        self.producer.set(
            MessageEndpoints.device_rpc(instr_params.get("rpc_id")),
            BECMessage.DeviceRPCMessage(
                device=instr.content["device"],
                return_val=None,
                out=exc_formatted,
                success=False,
            ).dumps(),
        )

    def _trigger_device(self, instr: BECMessage.DeviceInstructionMessage) -> None:
        logger.debug(f"Trigger device: {instr}")
        devices = instr.content["device"]
        if not isinstance(devices, list):
            devices = [devices]
        for dev in devices:
            obj = self.device_manager.devices.get(dev)
            obj.metadata = instr.metadata
            status = obj.obj.trigger()
            status.__dict__["instruction"] = instr
            status.add_callback(self._status_callback)

    def _kickoff_device(self, instr: BECMessage.DeviceInstructionMessage) -> None:
        logger.debug(f"Kickoff device: {instr}")
        obj = self.device_manager.devices.get(instr.content["device"]).obj
        kickoff_args = inspect.getfullargspec(obj.kickoff).args
        kickoff_parameter = instr.content["parameter"].get("configure", {})
        if len(kickoff_args) > 1:
            obj.kickoff(metadata=instr.metadata, **kickoff_parameter)
            return
        obj.configure(kickoff_parameter)
        status = obj.kickoff()
        status.__dict__["instruction"] = instr
        status.add_callback(self._status_callback)

    def _complete_device(self, instr: BECMessage.DeviceInstructionMessage) -> None:
        obj = self.device_manager.devices.get(instr.content["device"]).obj
        status = obj.complete()
        status.__dict__["instruction"] = instr
        status.add_callback(self._status_callback)

    def _set_device(self, instr: BECMessage.DeviceInstructionMessage) -> None:
        device_obj = self.device_manager.devices.get(instr.content["device"])
        if not device_obj.enabled_set:
            raise DisabledDeviceError(
                f"Setting the device {device_obj.name} is currently disabled."
            )
        logger.debug(f"Setting device: {instr}")
        val = instr.content["parameter"]["value"]
        obj = self.device_manager.devices.get(instr.content["device"]).obj
        # self.device_manager.add_req_done_sub(obj)
        status = obj.set(val)
        status.__dict__["instruction"] = instr
        status.add_callback(self._status_callback)

    def _pre_scan(self, instr: BECMessage.DeviceInstructionMessage) -> None:
        devices = instr.content["device"]
        if not isinstance(devices, list):
            devices = [devices]
        pipe = self.producer.pipeline()
        for dev in devices:
            obj = self.device_manager.devices.get(dev)
            obj.metadata = instr.metadata
            if hasattr(obj.obj, "pre_scan"):
                obj.obj.pre_scan()
            dev_msg = BECMessage.DeviceReqStatusMessage(
                device=dev,
                success=True,
                metadata=instr.metadata,
            ).dumps()
            self.producer.set_and_publish(MessageEndpoints.device_req_status(dev), dev_msg, pipe)
        pipe.execute()

    def _status_callback(self, status):
        pipe = self.producer.pipeline()
        if hasattr(status, "device"):
            device_name = status.device.root.name
        else:
            device_name = status.obj.root.name
        dev_msg = BECMessage.DeviceReqStatusMessage(
            device=device_name,
            success=status.success,
            metadata=status.instruction.metadata,
        ).dumps()
        logger.debug(f"req status for device {device_name}: {status.success}")
        self.producer.set_and_publish(
            MessageEndpoints.device_req_status(device_name), dev_msg, pipe
        )
        response = status.instruction.metadata.get("response")
        if response:
            self.producer.lpush(
                MessageEndpoints.device_req_status(status.instruction.metadata["RID"]),
                dev_msg,
                pipe,
                expire=18000,
            )

        pipe.execute()

    def _read_device(self, instr: BECMessage.DeviceInstructionMessage) -> None:
        # check performance -- we might have to change it to a background thread
        devices = instr.content["device"]
        if not isinstance(devices, list):
            devices = [devices]

        start = time.time()
        pipe = self.producer.pipeline()
        for dev in devices:
            self.device_manager.devices.get(dev).metadata = instr.metadata
            obj = self.device_manager.devices.get(dev).obj
            try:
                signals = obj.read()
            except Exception as exc:
                self.device_manager.connector.raise_alarm(
                    severity=Alarms.WARNING,
                    alarm_type="Warning",
                    source="DeviceServer",
                    content=f"Failed to read device {dev}.",
                    metadata={},
                )
                ds_dev = self.device_manager.devices.get(dev)
                if ds_dev.on_failure == OnFailure.RAISE:
                    raise exc

                if ds_dev.on_failure == OnFailure.RETRY:
                    # try to read it again, may have been only a glitch
                    signals = obj.read()
                elif ds_dev.on_failure == OnFailure.BUFFER:
                    # if possible, fall back to past readings
                    logger.warning(f"Failed to read {dev}. Trying to load an old value.")
                    old_msg = BECMessage.DeviceMessage.loads(
                        self.producer.get(MessageEndpoints.device_read(dev))
                    )
                    if not old_msg:
                        raise exc
                    signals = old_msg.content["signals"]

            self.producer.set_and_publish(
                MessageEndpoints.device_read(dev),
                BECMessage.DeviceMessage(signals=signals, metadata=instr.metadata).dumps(),
                pipe,
            )
            self.producer.set_and_publish(
                MessageEndpoints.device_readback(dev),
                BECMessage.DeviceMessage(signals=signals, metadata=instr.metadata).dumps(),
                pipe,
            )
            self.producer.set(
                MessageEndpoints.device_status(dev),
                BECMessage.DeviceStatusMessage(
                    device=dev, status=0, metadata=instr.metadata
                ).dumps(),
                pipe,
            )
        pipe.execute()
        logger.debug(
            f"Elapsed time for reading and updating status info: {(time.time()-start)*1000} ms"
        )

    def _stage_device(self, instr: BECMessage.DeviceInstructionMessage) -> None:
        devices = instr.content["device"]
        if not isinstance(devices, list):
            devices = [devices]

        pipe = self.producer.pipeline()
        for dev in devices:
            obj = self.device_manager.devices[dev].obj
            if hasattr(obj, "_staged"):
                # pylint: disable=protected-access
                if obj._staged == Staged.yes:
                    logger.info(f"Device {obj.name} was already staged and will be first unstaged.")
                    self.device_manager.devices[dev].obj.unstage()
                self.device_manager.devices[dev].obj.stage()
            self.producer.set(
                MessageEndpoints.device_staged(dev),
                BECMessage.DeviceStatusMessage(
                    device=dev, status=1, metadata=instr.metadata
                ).dumps(),
                pipe,
            )
        pipe.execute()

    def _unstage_device(self, instr: BECMessage.DeviceInstructionMessage) -> None:
        devices = instr.content["device"]
        if not isinstance(devices, list):
            devices = [devices]

        pipe = self.producer.pipeline()
        for dev in devices:
            obj = self.device_manager.devices[dev].obj
            if hasattr(obj, "_staged"):
                # pylint: disable=protected-access
                if obj._staged == Staged.yes:
                    self.device_manager.devices[dev].obj.unstage()
                else:
                    logger.debug(f"Device {obj.name} was already unstaged.")
            self.producer.set(
                MessageEndpoints.device_staged(dev),
                BECMessage.DeviceStatusMessage(
                    device=dev, status=0, metadata=instr.metadata
                ).dumps(),
                pipe,
            )
        pipe.execute()
