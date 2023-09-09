import asyncio

from bec_lib.core import BECMessage, MessageEndpoints, bec_logger

from bec_client.progressbar import ScanProgressBar

from .live_table import LiveUpdatesTable

logger = bec_logger.logger


class LiveUpdatesScanProgress(LiveUpdatesTable):
    """Live updates for scans using a progress bar based on the progress of one or more devices"""

    REPORT_TYPE = "scan_progress"

    async def _run_update(self, device_names: str):
        with ScanProgressBar(
            scan_number=self.scan_item.scan_number, clear_on_exit=False
        ) as progressbar:
            while True:
                if await self._update_progressbar(progressbar, device_names):
                    break

    async def _update_progressbar(self, progressbar: ScanProgressBar, device_names: str) -> bool:
        """
        Update the progressbar based on the device status message. Returns True if the scan is finished.
        """
        self.check_alarms()
        status = self.bec.producer.get(MessageEndpoints.device_progress(device_names[0]))
        if not status:
            logger.debug("waiting for new data point")
            await asyncio.sleep(0.1)
            return False
        status = BECMessage.DeviceStatusMessage.loads(status)
        if status.metadata.get("scanID") != self.scan_item.scanID:
            logger.debug("waiting for new data point")
            await asyncio.sleep(0.1)
            return False

        point_id = status.content["status"].get("value")
        if point_id is None:
            logger.debug("waiting for new data point")
            await asyncio.sleep(0.1)
            return False

        max_value = status.content["status"].get("max_value")
        if max_value and max_value != progressbar.max_points:
            progressbar.max_points = max_value

        progressbar.update(point_id)
        # process sync callbacks
        self.bec.callbacks.poll()
        self.scan_item.poll_callbacks()

        if point_id == max_value:
            return True
        return False
