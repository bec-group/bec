from __future__ import annotations

import time
from math import inf

from bec_utils import BECMessage, MessageEndpoints, bec_errors, bec_logger

from bec_client.queue_items import QueueStorage
from bec_client.request_items import RequestStorage
from bec_client.scan_items import ScanStorage

logger = bec_logger.logger


class ScanReport:
    def __init__(self) -> None:
        self._client = None
        self.request = None

    @classmethod
    def from_request(cls, request: BECMessage.ScanQueueMessage, client=None):
        """create new scan report from request"""
        scan_report = cls()
        scan_report._client = client

        client.queue.request_storage.update_with_request(request)
        scan_report.request = client.queue.request_storage.find_request_by_ID(
            request.metadata["RID"]
        )
        return scan_report

    @property
    def scan(self):
        """get the scan item"""
        return self.request.scan

    def wait(self, timeout=None):
        """wait until the request is completed"""
        timeout = timeout if timeout is not None else inf
        queue_item = None
        elapsed_time = 0
        sleep_time = 0.1
        while not queue_item:
            queue_item = self._client.queue.queue_storage.find_queue_item_by_requestID(
                self.request.requestID
            )
            elapsed_time += sleep_time
            time.sleep(sleep_time)
            if elapsed_time > timeout:
                raise TimeoutError
        while True:
            queue_history = self._client.queue.queue_storage.queue_history()
            for queue in queue_history:
                if not queue.content["queueID"] == queue_item.queueID:
                    continue
                if queue.content["status"] not in ["STOPPED", "COMPLETED"]:
                    continue
                if queue.content["status"] == "COMPLETED":
                    return
                if queue.content["status"] == "STOPPED":
                    raise bec_errors.ScanAbortion
            elapsed_time += sleep_time
            time.sleep(sleep_time)
            if elapsed_time > timeout:
                raise TimeoutError


class ScanManager:
    def __init__(self, connector):
        self.connector = connector
        self.producer = self.connector.producer()
        self.queue_storage = QueueStorage(scan_manager=self)
        self.request_storage = RequestStorage(scan_manager=self)
        self.scan_storage = ScanStorage(scan_manager=self)

        self._scan_queue_consumer = self.connector.consumer(
            topics=MessageEndpoints.scan_queue_status(),
            cb=self._scan_queue_status_callback,
            parent=self,
        )
        self._scan_queue_request_consumer = self.connector.consumer(
            topics=MessageEndpoints.scan_queue_request(),
            cb=self._scan_queue_request_callback,
            parent=self,
        )
        self._scan_queue_request_response_consumer = self.connector.consumer(
            topics=MessageEndpoints.scan_queue_request_response(),
            cb=self._scan_queue_request_response_callback,
            parent=self,
        )
        self._scan_status_consumer = self.connector.consumer(
            topics=MessageEndpoints.scan_status(),
            cb=self._scan_status_callback,
            parent=self,
        )

        self._scan_segment_consumer = self.connector.consumer(
            topics=MessageEndpoints.scan_segment(),
            cb=self._scan_segment_callback,
            parent=self,
        )

        self._scan_queue_consumer.start()
        self._scan_queue_request_consumer.start()
        self._scan_queue_request_response_consumer.start()
        self._scan_status_consumer.start()
        self._scan_segment_consumer.start()

    def update_with_queue_status(self, queue: BECMessage.ScanQueueStatusMessage) -> None:
        """update storage with a new queue status message"""
        self.queue_storage.update_with_status(queue)
        self.scan_storage.update_with_queue_status(queue)

    def request_scan_interruption(self, deferred_pause=True, scanID: str = None) -> None:
        """request a scan interruption

        Args:
            deferred_pause (bool, optional): Request a deferred pause. If False, a pause will be requested. Defaults to True.
            scanID (str, optional): ScanID. Defaults to None.

        """
        if scanID is None:
            scanID = self.scan_storage.current_scanID
        if not any(scanID):
            return self.request_scan_abortion()

        action = "deferred_pause" if deferred_pause else "pause"
        logger.info(f"Requesting {action}")
        return self.producer.send(
            MessageEndpoints.scan_queue_modification_request(),
            BECMessage.ScanQueueModificationMessage(
                scanID=scanID, action=action, parameter={}
            ).dumps(),
        )

    def request_scan_abortion(self, scanID=None):
        """request a scan abortion

        Args:
            scanID (str, optional): ScanID. Defaults to None.

        """
        if scanID is None:
            scanID = self.scan_storage.current_scanID
        logger.info("Requesting scan abortion")
        self.producer.send(
            MessageEndpoints.scan_queue_modification_request(),
            BECMessage.ScanQueueModificationMessage(
                scanID=scanID, action="abort", parameter={}
            ).dumps(),
        )

    def request_scan_continuation(self, scanID=None):
        """request a scan continuation

        Args:
            scanID (str, optional): ScanID. Defaults to None.

        """
        if scanID is None:
            scanID = self.scan_storage.current_scanID
        logger.info("Requesting scan continuation")
        self.producer.send(
            MessageEndpoints.scan_queue_modification_request(),
            BECMessage.ScanQueueModificationMessage(
                scanID=scanID, action="continue", parameter={}
            ).dumps(),
        )

    def request_queue_reset(self):
        """request a scan queue reset"""
        logger.info("Requesting a queue reset")
        self.producer.send(
            MessageEndpoints.scan_queue_modification_request(),
            BECMessage.ScanQueueModificationMessage(
                scanID=None, action="clear", parameter={}
            ).dumps(),
        )

    @property
    def current_scan_number(self):
        return int(self.producer.get(MessageEndpoints.scan_number()))

    @staticmethod
    def _scan_queue_status_callback(msg, *, parent: ScanManager, **_kwargs) -> None:
        queue_status = BECMessage.ScanQueueStatusMessage.loads(msg.value)
        if not queue_status:
            return
        parent.update_with_queue_status(queue_status)

    @staticmethod
    def _scan_queue_request_callback(msg, *, parent: ScanManager, **_kwargs) -> None:
        request = BECMessage.ScanQueueMessage.loads(msg.value)
        parent.request_storage.update_with_request(request)

    @staticmethod
    def _scan_queue_request_response_callback(msg, *, parent: ScanManager, **_kwargs) -> None:
        response = BECMessage.RequestResponseMessage.loads(msg.value)
        logger.debug(response)
        parent.request_storage.update_with_response(response)

    @staticmethod
    def _scan_status_callback(msg, *, parent: ScanManager, **_kwargs) -> None:
        scan = BECMessage.ScanStatusMessage.loads(msg.value)
        parent.scan_storage.update_with_scan_status(scan)

    @staticmethod
    def _scan_segment_callback(msg, *, parent: ScanManager, **_kwargs) -> None:
        scan_msgs = BECMessage.ScanMessage.loads(msg.value)
        if not isinstance(scan_msgs, list):
            scan_msgs = [scan_msgs]
        for scan_msg in scan_msgs:
            parent.scan_storage.add_scan_segment(scan_msg)

    def shutdown(self):
        """stop the scan manager's threads"""
        self._scan_queue_consumer.shutdown()
        self._scan_queue_request_consumer.shutdown()
        self._scan_queue_request_response_consumer.shutdown()
        self._scan_status_consumer.shutdown()
        self._scan_segment_consumer.shutdown()
