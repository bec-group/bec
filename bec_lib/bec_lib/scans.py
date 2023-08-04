from __future__ import annotations

import builtins
import uuid
from contextlib import ContextDecorator
from typing import TYPE_CHECKING, Callable

import msgpack
from cytoolz import partition
from typeguard import typechecked

from bec_lib.core import BECMessage, MessageEndpoints, bec_logger
from bec_lib.core.connector import ConsumerConnector
from bec_lib.core.utils import scan_to_csv

from .callback_handler import CallbackRegister
from .devicemanager_client import Device
from .scan_manager import ScanReport

if TYPE_CHECKING:
    from bec_client import BECClient

logger = bec_logger.logger


class ScanObject:
    def __init__(self, scan_name: str, scan_info: dict, client: BECClient = None) -> None:
        self.scan_name = scan_name
        self.scan_info = scan_info
        self.client = client

        # run must be an anonymous function to allow for multiple doc strings
        self.run = lambda *args, **kwargs: self._run(*args, **kwargs)

    def _run(self, *args, callback: Callable = None, async_callback: Callable = None, **kwargs):
        if self.client.alarm_handler.alarms_stack:
            logger.warning("The alarm stack is not empty but will be cleared now.")
            self.client.clear_all_alarms()
        scans = self.client.scans

        # handle reserved kwargs:
        hide_report_kwarg = kwargs.get("hide_report", False)
        hide_report = hide_report_kwarg or scans._hide_report

        metadata = self.client.metadata.copy()
        if not "sample_name" in metadata:
            sample_name = self.client.get_global_var("sample_name")
            if sample_name is not None:
                metadata["sample_name"] = sample_name

        if "md" in kwargs:
            metadata.update(kwargs["md"])

        if scans._scan_group:
            metadata["queue_group"] = scans._scan_group
        if scans._scan_def_id:
            metadata["scan_def_id"] = scans._scan_def_id
        if scans._dataset_id_on_hold:
            metadata["dataset_id_on_hold"] = scans._dataset_id_on_hold

        kwargs["md"] = metadata

        request = Scans.prepare_scan_request(self.scan_name, self.scan_info, *args, **kwargs)
        requestID = str(uuid.uuid4())  # TODO: move this to the API server
        request.metadata["RID"] = requestID

        self._send_scan_request(request)

        report = ScanReport.from_request(request, client=self.client)
        report.request.callbacks.register_many("scan_segment", callback, sync=True)
        report.request.callbacks.register_many("scan_segment", async_callback, sync=False)

        if scans._scan_export and scans._scan_export.scans is not None:
            scans._scan_export.scans.append(report)

        if not hide_report and self.client.live_updates:
            scan_report_type = self._get_scan_report_type(hide_report)
            # call process_requests even if report_type is None
            self.client.live_updates.process_request(request, scan_report_type, callback)

        self.client.callbacks.poll()

        return report

    def _get_scan_report_type(self, hide_report) -> str:
        if hide_report:
            return None
        return self.scan_info.get("scan_report_hint")

    def _start_consumer(self, request: BECMessage.ScanQueueMessage) -> ConsumerConnector:
        consumer = self.client.device_manager.connector.consumer(
            [
                MessageEndpoints.device_readback(dev)
                for dev in request.content["parameter"]["args"].keys()
            ],
            threaded=False,
            cb=(lambda msg: msg),
        )
        return consumer

    def _send_scan_request(self, request: BECMessage.ScanQueueMessage) -> None:
        self.client.device_manager.producer.send(
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
        self._dataset_id_on_hold = None
        self._dataset_id_on_hold_ctx = DatasetIdOnHold(parent=self)
        self._scan_export = None

    def _import_scans(self):
        available_scans = msgpack.loads(
            self.parent.producer.get(MessageEndpoints.available_scans())
        )
        for scan_name, scan_info in available_scans.items():
            self._available_scans[scan_name] = ScanObject(scan_name, scan_info, client=self.parent)
            setattr(
                self,
                scan_name,
                self._available_scans[scan_name].run,
            )
            setattr(getattr(self, scan_name), "__doc__", scan_info.get("doc"))

    @staticmethod
    def get_arg_type(in_type: str):
        """translate type string into python type"""
        # pylint: disable=too-many-return-statements
        if in_type == "float":
            return (float, int)
        if in_type == "int":
            return int
        if in_type == "list":
            return list
        if in_type == "boolean":
            return bool
        if in_type == "str":
            return str
        if in_type == "dict":
            return dict
        if in_type == "device":
            return Device
        raise TypeError(f"Unknown type {in_type}")

    @staticmethod
    def prepare_scan_request(
        scan_name: str, scan_info: dict, *args, **kwargs
    ) -> BECMessage.ScanQueueMessage:
        """Prepare scan request message with given scan arguments

        Args:
            scan_name (str): scan name (matching a scan name on the scan server)
            scan_info (dict): dictionary describing the scan (e.g. doc string, required kwargs etc.)

        Raises:
            TypeError: Raised if not all required keyword arguments have been specified.
            TypeError: Raised if the number of args do fit into the required bundling pattern.
            TypeError: Raised if an argument is not of the required type as specified in scan_info.

        Returns:
            BECMessage.ScanQueueMessage: _description_
        """
        arg_input = scan_info.get("arg_input", [])
        arg_bundle_size = scan_info.get("arg_bundle_size")
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

        metadata = {}
        if "md" in kwargs:
            metadata = kwargs.pop("md")
        params = {
            "args": Scans._parameter_bundler(args, arg_bundle_size),
            "kwargs": kwargs,
        }
        return BECMessage.ScanQueueMessage(
            scan_type=scan_name, parameter=params, queue="primary", metadata=metadata
        )

    @staticmethod
    def _parameter_bundler(args, bundle_size):
        """

        Args:
            args:
            bundle_size: number of parameters per bundle

        Returns:

        """
        if not bundle_size:
            return args
        params = {}
        for cmds in partition(bundle_size, args):
            cmds_serialized = [cmd.name if hasattr(cmd, "name") else cmd for cmd in cmds]
            params[cmds_serialized[0]] = cmds_serialized[1:]
        return params

    @property
    def scan_group(self):
        """Context manager / decorator for defining scan groups"""
        return self._scan_group_ctx

    @property
    def scan_def(self):
        """Context manager / decorator for defining new scans"""
        return self._scan_def_ctx

    @property
    def hide_report(self):
        """Context manager / decorator for hiding the report"""
        return self._hide_report_ctx

    @property
    def dataset_id_on_hold(self):
        """Context manager / decorator for setting the dataset id on hold"""
        return self._dataset_id_on_hold_ctx

    def scan_export(self, output_file: str):
        """Context manager / decorator for exporting scans"""
        return ScanExport(output_file)


class ScanGroup(ContextDecorator):
    def __init__(self, parent: Scans = None) -> None:
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
    def __init__(self, parent: Scans = None) -> None:
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
    def __init__(self, parent: Scans = None) -> None:
        super().__init__()
        self.parent = parent

    def __enter__(self):
        if self.parent._hide_report is None:
            self.parent._hide_report = True
        return self

    def __exit__(self, *exc):
        self.parent._hide_report = None


class DatasetIdOnHold(ContextDecorator):
    def __init__(self, parent: Scans = None) -> None:
        super().__init__()
        self.parent = parent
        self._call_count = 0

    def __enter__(self):
        self._call_count += 1
        if self.parent._dataset_id_on_hold is None:
            self.parent._dataset_id_on_hold = True
        return self

    def __exit__(self, *exc):
        self._call_count -= 1
        if self._call_count:
            return
        self.parent._dataset_id_on_hold = None
        queue = self.parent.parent.queue
        queue.next_dataset_number += 1


class Metadata:
    @typechecked
    def __init__(self, metadata: dict) -> None:
        """Context manager for updating metadata

        Args:
            metadata (dict): Metadata dictionary
        """
        self.client = self._get_client()
        self._metadata = metadata
        self._orig_metadata = None

    def _get_client(self):
        return builtins.__dict__["bec"]

    def __enter__(self):
        self._orig_metadata = self.client.metadata.copy()
        self.client.metadata.update(self._metadata)
        return self

    def __exit__(self, *exc):
        self.client.metadata = self._orig_metadata


class ScanExport:
    def __init__(self, output_file: str) -> None:
        """Context manager for exporting scans

        Args:
            output_file (str): Output file name
        """
        self.output_file = output_file
        self.client = self._get_client()
        self.client.scans._scan_export = self
        self.scans = None

    def _get_client(self):
        return builtins.__dict__["bec"]

    def __enter__(self):
        self.scans = []
        return self

    def __exit__(self, *exc):
        try:
            for scan in self.scans:
                scan.wait()
        finally:
            scan_to_csv(self.scans, self.output_file)
            self.scans = None
