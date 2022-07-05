from bec_utils import BECMessage, bec_logger

from .errors import ScanAbortion

logger = bec_logger.logger


class ScanAssembler:
    """
    ScanAssembler receives scan messages and translates the scan message into device instructions.
    """

    def __init__(self, *, parent):
        self.parent = parent
        self.device_manager = self.parent.device_manager
        self.connector = self.parent.connector
        self.scan_manager = (
            self.parent.scan_manager
        )  # TODO should these be the same dict, or a copy?

    def assemble_device_instructions(self, msg: BECMessage.ScanQueueMessage):
        scan = msg.content.get("scan_type")
        cls_name = self.scan_manager.available_scans[scan]["class"]
        scan_cls = self.scan_manager.scan_dict[cls_name]

        logger.info(f"Preparing instructions of request of type {scan} / {scan_cls.__name__}")
        try:
            scan_instance = scan_cls(
                device_manager=self.device_manager,
                parameter=msg.content.get("parameter"),
                metadata=msg.metadata,
            )
            return scan_instance
        except Exception as exc:
            logger.error(f"Failed to initialize the scan class of type {scan_cls.__name__}")
            raise ScanAbortion from exc
