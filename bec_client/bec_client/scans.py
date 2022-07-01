import asyncio
import logging
import uuid
from contextlib import ContextDecorator

import msgpack
from bec_utils import BECMessage as BMessage
from bec_utils import MessageEndpoints
from bec_utils.connector import ConsumerConnector
from cytoolz import partition

from .callbacks import (
    live_updates_readback,
    live_updates_readback_progressbar,
    live_updates_table,
)
from .devicemanager_client import Device
from .scan_queue import ScanReport

logger = logging.getLogger("scans")


class ScanObject:
    def __init__(self, scan_name: str, scan_info: dict, parent=None) -> None:
        self.scan_name = scan_name
        self.scan_info = scan_info
        self.parent = parent

        # run must be an anonymous function to allow for multiple doc strings
        self.run = lambda *args, **kwargs: asyncio.run(self._run(*args, **kwargs))

    async def _run(self, *args, **kwargs):
        with self.parent._sighandler:
            # handle reserved kwargs:
            hide_report_kwarg = False
            if "hide_report" in kwargs:
                hide_report_kwarg = kwargs["hide_report"]
            hide_report = hide_report_kwarg or self.parent.scans._hide_report

            if self.parent.scans._scan_group is not None:
                if "md" not in kwargs:
                    kwargs["md"] = {}
                kwargs["md"]["queue_group"] = self.parent.scans._scan_group
            if self.parent.scans._scan_def_id is not None:
                if "md" not in kwargs:
                    kwargs["md"] = {}
                kwargs["md"]["scan_def_id"] = self.parent.scans._scan_def_id

            request = Scans._prepare_scan_request(self.scan_name, self.scan_info, *args, **kwargs)
            requestID = str(uuid.uuid4())  # TODO: move this to the API server
            request.metadata["RID"] = requestID

            scan_report = ScanReport.from_request(request, client=self.parent)

            scan_report_type = self._get_scan_report_type(hide_report)
            if scan_report_type == "readback":
                consumer = self._start_consumer(request)
                self._send_scan_request(request)
                await live_updates_readback_progressbar(
                    self.parent.devicemanager,
                    request,
                )
                consumer.shutdown()
            elif scan_report_type == "table":
                self._send_scan_request(request)
                if (
                    self.parent.scans._scan_def_id is None
                    or request.content["scan_type"] == "close_scan_def"
                ):
                    await live_updates_table(self.parent, request)
            else:
                self._send_scan_request(request)
            return scan_report

    def _get_scan_report_type(self, hide_report) -> str:
        if hide_report:
            return None
        else:
            return self.scan_info.get("scan_report_hint")

    def _start_consumer(self, request: BMessage.ScanQueueMessage) -> ConsumerConnector:
        consumer = self.parent.devicemanager.connector.consumer(
            [
                MessageEndpoints.device_readback(dev)
                for dev in request.content["parameter"]["args"].keys()
            ],
            threaded=False,
            cb=(lambda msg: msg),
        )
        return consumer

    def _send_scan_request(self, request: BMessage.ScanQueueMessage) -> None:
        self.parent.devicemanager.producer.send(
            MessageEndpoints.scan_queue_request(), request.dumps()
        )


class Scans:
    def __init__(self, parent):
        self.parent = parent
        self._available_scans = {}
        self._import_scans()
        self._scan_group = None
        self._scan_def_id = None
        self._scan_group_ctx = ScanGroup(parent=self)
        self._scan_def_ctx = ScanDef(parent=self)
        self._hide_report = None
        self._hide_report_ctx = HideReport(parent=self)

    def _import_scans(self):

        available_scans = msgpack.loads(
            self.parent.producer.get(MessageEndpoints.available_scans())
        )
        for scan_name, scan_info in available_scans.items():
            self._available_scans[scan_name] = ScanObject(scan_name, scan_info, parent=self.parent)
            setattr(
                self,
                scan_name,
                self._available_scans[scan_name].run,
            )
            setattr(getattr(self, scan_name), "__doc__", scan_info.get("doc"))

    @staticmethod
    def get_arg_type(in_type: str):
        if in_type == "float":
            return (float, int)
        elif in_type == "int":
            return int
        elif in_type == "list":
            return list
        elif in_type == "boolean":
            return bool
        elif in_type == "str":
            return str
        elif in_type == "dict":
            return dict
        elif in_type == "device":
            return Device
        else:
            TypeError(f"Unknown type {in_type}")

    @staticmethod
    def _prepare_scan_request(
        scan_name: str, scan_info: dict, *args, **kwargs
    ) -> BMessage.ScanQueueMessage:
        """Prepare scan request message with given scan arguments

        Args:
            scan_name (str): scan name (matching a scan name on the scan server)
            scan_info (dict): dictionary describing the scan (e.g. doc string, required kwargs etc.)

        Raises:
            TypeError: Raised if not all required keyword arguments have been specified.
            TypeError: Raised if the number of args do fit into the required bundling pattern.
            TypeError: Raised if an argument is not of the required type as specified in scan_info.

        Returns:
            BMessage.ScanQueueMessage: _description_
        """
        arg_input = scan_info.get("arg_input")
        if arg_input is not None:
            arg_bundle_size = len(arg_input)
            if len(arg_input) > 0:
                if len(args) % len(arg_input) != 0:
                    raise TypeError(
                        f"{scan_info.get('doc')}\n {scan_name} takes multiples of {len(arg_input)} arguments ({len(args)} given).",
                    )
                if not all(req_kwarg in kwargs for req_kwarg in scan_info.get("required_kwargs")):
                    raise TypeError(
                        f"{scan_info.get('doc')}\n Not all required keyword arguments have been specified. The required arguments are: {scan_info.get('required_kwargs')}"
                    )
                for ii, arg in enumerate(args):
                    if not isinstance(arg, Scans.get_arg_type(arg_input[ii % len(arg_input)])):
                        raise TypeError(
                            f"{scan_info.get('doc')}\n Argument {ii} must be of type {arg_input[ii%len(arg_input)]}, not {type(arg).__name__}."
                        )
        else:
            logger.warning("Could not check arguments against scan input types.")
            arg_bundle_size = len(args)
        md = {}
        if "md" in kwargs:
            md = kwargs.pop("md")
        params = {
            "args": Scans._parameter_bundler(args, arg_bundle_size),
            "kwargs": kwargs,
        }
        return BMessage.ScanQueueMessage(
            scan_type=scan_name, parameter=params, queue="primary", metadata=md
        )

    @staticmethod
    def _parameter_bundler(args, bundle_size):
        """

        Args:
            args:
            bundle_size: number of parameters per bundle

        Returns:

        """
        params = {}
        for cmds in partition(bundle_size, args):
            cmds_serialized = [cmd.name if hasattr(cmd, "name") else cmd for cmd in cmds]
            params[cmds_serialized[0]] = cmds_serialized[1:]
        return params

    @property
    def scan_group(self):
        return self._scan_group_ctx

    @property
    def scan_def(self):
        return self._scan_def_ctx

    @property
    def hide_report(self):
        return self._hide_report_ctx


class ScanGroup(ContextDecorator):
    def __init__(self, parent=None) -> None:
        super().__init__()
        self.parent = parent

    def __enter__(self):
        group_id = str(uuid.uuid4())
        self.parent._scan_group = group_id
        return self

    def __exit__(self, *exc):
        self.parent.close_scan_group()
        self.parent._scan_group = None


class ScanDef(ContextDecorator):
    def __init__(self, parent=None) -> None:
        super().__init__()
        self.parent = parent

    def __enter__(self):
        scan_def_id = str(uuid.uuid4())
        self.parent._scan_def_id = scan_def_id
        self.parent.open_scan_def()
        return self

    def __exit__(self, *exc):
        self.parent.close_scan_def()
        self.parent._scan_def_id = None


class HideReport(ContextDecorator):
    def __init__(self, parent=None) -> None:
        super().__init__()
        self.parent = parent

    def __enter__(self):
        if self.parent._hide_report is None:
            self.parent._hide_report = True
        return self

    def __exit__(self, *exc):
        self.parent._hide_report = None
