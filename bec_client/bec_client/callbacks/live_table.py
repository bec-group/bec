from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable, List

import numpy as np
from bec_utils import BECMessage, bec_logger

from bec_client.prettytable import PrettyTable
from bec_client.progressbar import ScanProgressBar

from .utils import LiveUpdatesBase, check_alarms

if TYPE_CHECKING:
    from bec_client.bec_client import BECClient

logger = bec_logger.logger


def sort_devices(devices, scan_devices) -> list:
    """sort the devices to ensure that the table starts with scan motors"""
    for scan_dev in list(scan_devices)[::-1]:
        devices.remove(scan_dev)
        devices.insert(0, scan_dev)
    return devices


class LiveUpdatesTable(LiveUpdatesBase):
    """Live updates for scans using a table and a scan progess bar.

    Args:
        bec (BECClient): client instance
        request (BECMessage.ScanQueueMessage): The scan request that should be monitored

    Raises:
        TimeoutError: Raised if no queue item is added before reaching a predefined timeout.
        RuntimeError: Raised if more points than requested are returned.
        ScanRequestError: Raised if the scan was rejected by the server.
    """

    MAX_DEVICES = 10

    def __init__(
        self, bec: BECClient, request: BECMessage.ScanQueueMessage, callbacks: List[Callable] = None
    ) -> None:
        super().__init__(bec, request, callbacks)
        self.scan_queue_request = None
        self.scan_item = None
        self.dev_values = None
        self.point_data = None
        self.point_id = 0
        self.table = None

    async def wait_for_scan_to_start(self):
        """wait until the scan starts"""
        while True:
            queue_pos = self.scan_item.queue.queue_position
            self.check_alarms()
            if self.scan_item.status == "closed":
                break
            if queue_pos is None:
                logger.debug(f"Could not find queue entry for scanID {self.scan_item.scanID}")
                continue
            if queue_pos == 0:
                break
            print(
                f"Scan is enqueued and is waiting for execution. Current position in queue: {queue_pos + 1}.",
                end="\r",
                flush=True,
            )
            await asyncio.sleep(0.1)
        while not self.scan_item.scan_number:
            await asyncio.sleep(0.05)

    async def wait_for_scan_item_to_finish(self):
        """wait for scan completion"""
        while True:
            if self.scan_item.end_time:
                if self.scan_item.open_queue_group:
                    break
                if self.scan_item.queue.queue_position is None:
                    break
            self.check_alarms()
            await asyncio.sleep(0.1)

    def check_alarms(self):
        check_alarms(self.bec)

    async def resume(self, request):
        super().__init__(self.bec, request)
        await self.process_request()

    @property
    def devices(self):
        """get the devices for the callback"""
        if self.point_data.metadata["scan_type"] == "step":
            return self.get_devices_from_scan_data(self.scan_item.data[0])
        if self.point_data.metadata["scan_type"] == "fly":
            devices = list(self.point_data.content["data"].keys())
            if len(devices) > self.MAX_DEVICES:
                return devices[0 : self.MAX_DEVICES]
            return devices
        return None

    def get_devices_from_scan_data(self, data: BECMessage.ScanMessage) -> list:
        """extract interesting devices from a scan request"""
        device_manager = self.bec.device_manager
        scan_devices = data.metadata.get("scan_report_devices")
        primary_devices = device_manager.devices.primary_devices(
            [device_manager.devices[dev] for dev in scan_devices]
        )
        devices = [hint for dev in primary_devices for hint in dev._hints]
        devices = sort_devices(devices, scan_devices)
        if len(devices) > self.MAX_DEVICES:
            return devices[0 : self.MAX_DEVICES]
        return devices

    def _prepare_table(self) -> PrettyTable:
        header = self._get_header()
        max_len = max(len(head) for head in header)
        return PrettyTable(header, padding=max_len)

    def _get_header(self) -> List:
        header = ["seq. num"]
        for dev in self.devices:
            obj = self.bec.device_manager.devices[dev]
            header.extend(obj._hints)
        return header

    async def update_scan_item(self):
        """get the current scan item"""
        while self.scan_queue_request.scan is None:
            self.check_alarms()
            await asyncio.sleep(0.1)
        self.scan_item = self.scan_queue_request.scan

    async def core(self):
        req_ID = self.scan_queue_request.requestID
        while True:
            request_block = [
                req for req in self.scan_item.queue.request_blocks if req["RID"] == req_ID
            ][0]
            if not request_block["is_scan"]:
                break
            if request_block["report_instructions"]:
                break
            self.check_alarms()

        for instr in request_block["report_instructions"]:
            await self._run_table_update(instr["table_wait"])

    async def _run_table_update(self, target_num_points):
        with ScanProgressBar(
            scan_number=self.scan_item.scan_number, clear_on_exit=True
        ) as progressbar:
            while True:
                self.check_alarms()
                self.point_data = self.scan_item.data.get(self.point_id)
                if self.scan_item.num_points:
                    progressbar.max_points = self.scan_item.num_points
                    if target_num_points == 0:
                        target_num_points = self.scan_item.num_points

                progressbar.update(self.point_id)
                if self.point_data:
                    if not self.table:
                        self.dev_values = (len(self._get_header()) - 1) * [0]
                        self.table = self._prepare_table()
                        print(self.table.get_header_lines())

                    self.point_id += 1
                    if self.point_id % 100 == 0:
                        print(self.table.get_header_lines())
                    ind = 0
                    for dev in self.devices:
                        obj = self.bec.device_manager.devices[dev]
                        for hint in obj._hints:
                            signal = self.point_data.content["data"].get(dev, {}).get(hint)
                            self.dev_values[ind] = signal.get("value") if signal else -999
                            ind += 1
                    print(self.table.get_row(self.point_id, *self.dev_values))
                    self.emit_point(self.point_data.content, metadata=self.point_data.metadata)
                    progressbar.update(self.point_id)
                else:
                    logger.debug("waiting for new data point")
                    await asyncio.sleep(0.1)

                if not self.scan_item.num_points:
                    continue
                # if self.scan_item.open_scan_defs:
                #     continue

                if self.point_id == target_num_points:
                    break
                if self.point_id > self.scan_item.num_points:
                    raise RuntimeError("Received more points than expected.")

    def close_table(self):
        elapsed_time = self.scan_item.end_time - self.scan_item.start_time
        print(
            self.table.get_footer(
                f"Scan {self.scan_item.scan_number} finished. Scan ID {self.scan_item.scanID}. Elapsed time: {elapsed_time:.2f} s"
            )
        )

    async def process_request(self):
        if self.request.content["scan_type"] == "close_scan_def":
            await self.wait_for_scan_item_to_finish()
            self.close_table()
            return

        await self.wait_for_request_acceptance()
        await asyncio.wait_for(self.update_scan_item(), timeout=15)
        await self.wait_for_scan_to_start()

        if self.table:
            self.table = None
        else:
            print(f"\nStarting scan {self.scan_item.scan_number}.")

        await self.core()

    async def run(self):
        if self.request.content["scan_type"] == "open_scan_def":
            await self.wait_for_request_acceptance()
            return
        await self.process_request()
        await self.wait_for_scan_item_to_finish()
        self.close_table()
