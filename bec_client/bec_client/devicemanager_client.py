import functools
import inspect
import logging
import sys
import time
import uuid

import bec_utils.BECMessage as BMessage
from bec_utils import Device, DeviceManagerBase, MessageEndpoints

from bec_client.callbacks import ScanRequestError

logger = logging.getLogger(__name__)

"""
Device (bluesky interface):
* trigger
* read
* describe
* stage
* unstage
* pause
* resume

Signal:
* trigger
* get
* put
* set
* value
* read
* describe
* limits
* low limit
* high limit


Positioner:
* trigger
* read
* set 
* stop
* settle_time
* timeout
* egu
* limits
* low_limit
* high_limit
* move
* position
* moving







device status
* connected
* enabled
* status (idle, moving etc)



instead of motor_is_moving, subscribe to SUB_DONE for receiving successful movements and SUB_REQ_DONE for "requested move finished". The latter is cleared after each request
"""


def rpc(fcn):
    """Decorator to perform rpc calls."""

    @functools.wraps(fcn)
    def wrapper(self, *args, **kwargs):
        device, func_call = self._get_rpc_func_name(fcn=fcn)

        if kwargs.get("cached", False):
            return fcn(self, *args, **kwargs)
        return self._run_rpc_call(device, func_call, *args, **kwargs)

    return wrapper


class RPCBase:
    def __init__(self, name: str, info: dict = None, parent=None) -> None:
        self.name = name
        if info is None:
            info = {}
        self._info = info.get("device_info")
        self.parent = parent
        self._enabled = False
        self._custom_rpc_methods = {}
        if self._info:
            self._parse_info()

    def _run(self, *args, **kwargs):
        device, func_call = self._get_rpc_func_name(fcn_name=self.name, use_parent=True)
        return self._run_rpc_call(device, func_call, args, kwargs)

    @property
    def root(self):
        parent = self
        while not isinstance(parent.parent, DMClient):
            parent = parent.parent
        return parent

    def _run_rpc_call(self, device, func_call, *args, **kwargs):
        rpc_id = str(uuid.uuid4())
        requestID = str(uuid.uuid4())  # TODO: move this to the API server
        params = {
            "device": device,
            "rpc_id": rpc_id,
            "func": func_call,
            "args": args,
            "kwargs": kwargs,
        }
        msg = BMessage.ScanQueueMessage(
            scan_type="device_rpc",
            parameter=params,
            queue="primary",
            metadata={"RID": requestID},
        )
        self.root.parent.producer.send(MessageEndpoints.scan_queue_request(), msg.dumps())
        scan_queue = self.root.parent.parent.queue
        while scan_queue.scan_queue_requests.get(requestID) is None:
            time.sleep(0.1)
        scan_queue_request = scan_queue.scan_queue_requests.get(requestID)
        while scan_queue_request.decision_pending:
            time.sleep(0.1)
        if not all(scan_queue_request.accepted):
            raise ScanRequestError(
                f"Function call was rejected by the server: {scan_queue_request.response.content['message']}"
            )
        while True:
            msg = self.root.parent.producer.get(MessageEndpoints.device_rpc(rpc_id))
            if msg:
                break
            time.sleep(0.1)
        msg = BMessage.DeviceRPCMessage.loads(msg)
        return msg.content.get("return_val")

    def _get_rpc_func_name(self, fcn_name=None, fcn=None, use_parent=False):
        if not fcn_name:
            fcn_name = fcn.__name__
        full_func_call = ".".join([self._compile_function_path(use_parent=use_parent), fcn_name])
        device = full_func_call.split(".")[0]
        func_call = ".".join(full_func_call.split(".")[1:])
        return (device, func_call)

    def _compile_function_path(self, use_parent=False) -> str:
        if use_parent:
            parent = self.parent
        else:
            parent = self
        func_call = []
        while not isinstance(parent, DMClient):
            func_call.append(parent.name)
            parent = parent.parent
        return ".".join(func_call[::-1])

    def _parse_info(self):
        if self._info.get("signals"):
            for signal_name in self._info.get("signals"):
                setattr(self, signal_name, Signal(signal_name, parent=self))
        if self._info.get("subdevices"):
            for dev in self._info.get("subdevices"):
                base_class = dev["device_info"].get("device_base_class")
                if base_class == "positioner":
                    setattr(self, dev.get("name"), Positioner(signal_name, parent=self))
                elif base_class == "device":
                    setattr(self, dev.get("name"), Device(signal_name, parent=self))

        for user_access_name, descr in self._info.get("custom_user_access").items():

            if "type" in descr:
                self._custom_rpc_methods[user_access_name] = RPCBase(
                    name=user_access_name, info=descr, parent=self
                )
                setattr(
                    self,
                    user_access_name,
                    self._custom_rpc_methods[user_access_name]._run,
                )
            else:
                self._custom_rpc_methods[user_access_name] = RPCBase(
                    name=user_access_name,
                    info={"device_info": {"custom_user_access": descr}},
                    parent=self,
                )
                setattr(
                    self,
                    user_access_name,
                    self._custom_rpc_methods[user_access_name],
                )


class DeviceBase(RPCBase, Device):
    @property
    def enabled(self):
        return self.root._enabled

    @enabled.setter
    def enabled(self, val):
        self.root._enabled = val

    @rpc
    def trigger(self, rpc_id: str):
        pass

    @rpc
    def read(self, cached=False, use_readback=False):
        if use_readback:
            val = self.parent.producer.get(MessageEndpoints.device_readback(self.name))
        else:
            val = self.parent.producer.get(MessageEndpoints.device_read(self.name))
        if val:
            return BMessage.DeviceMessage.loads(val).content["signals"].get(self.name)
        else:
            return None

    @rpc
    def describe(self):
        pass

    @rpc
    def stage(self):
        pass

    @rpc
    def unstage(self):
        pass

    @rpc
    def summary(self):
        pass


class Signal(DeviceBase):
    @rpc
    def get(self):
        pass

    @rpc
    def put(self, val):
        pass

    @rpc
    def set(self, val):
        pass

    @rpc
    def value(self):
        pass

    @rpc
    def limits(self):
        pass

    @rpc
    def low_limit(self):
        pass

    @rpc
    def high_limit(self):
        pass


class Positioner(DeviceBase):
    @rpc
    def set(self, val):
        pass

    @rpc
    def stop(self):
        pass

    @rpc
    def settle_time(self):
        pass

    @rpc
    def timeout(self):
        pass

    @rpc
    def egu(self):
        pass

    @rpc
    def limits(self):
        pass

    @rpc
    def low_limit(self):
        pass

    @rpc
    def high_limit(self):
        pass

    @rpc
    def move(self):
        pass

    @rpc
    def position(self):
        pass

    @rpc
    def moving(self):
        pass


class DMClient(DeviceManagerBase):
    def __init__(self, parent, scibec_url):
        super().__init__(parent.connector, scibec_url)
        self.parent = parent

    def _load_config_device(self):
        if self._is_config_valid():
            start = time.time()
            for name, dev in self._config.items():
                logger.info(
                    f"Adding device {name}: {'ENABLED' if dev['status']['enabled'] else 'DISABLED'}"
                )
                obj = ClientDevice(name, self, dev["status"]["enabled"])
                self.devices._add_device(name, obj)
            print(time.time() - start)

    def _get_device_info(self, device_name) -> BMessage.DeviceInfoMessage:
        msg = BMessage.DeviceInfoMessage.loads(
            self.producer.get(MessageEndpoints.device_info(device_name))
        )
        return msg

    def _load_session(self):
        if self._is_config_valid():
            for dev in self._session["devices"]:
                msg = self._get_device_info(dev.get("name"))
                self._add_device(dev, msg)

    def _add_device(self, dev: dict, msg: BMessage.DeviceInfoMessage):
        name = msg.content["device"]
        info = msg.content["info"]

        base_class = info["device_info"]["device_base_class"]

        if base_class == "device":
            logger.info(f"Adding new device {name}")
            obj = DeviceBase(name, info, parent=self)
        elif base_class == "positioner":
            logger.info(f"Adding new positioner {name}")
            obj = Positioner(name, info, parent=self)
        elif base_class == "signal":
            logger.info(f"Adding new signal {name}")
            obj = Signal(name, info, parent=self)
        else:
            logger.error(f"Trying to add new device {name} of type {base_class}")

        for key, val in dev.items():
            obj.__setattr__(key, val)
        self.devices._add_device(name, obj)