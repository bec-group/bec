class MessageEndpoints:
    # devices feedback
    _device_status = "internal/devices/status"
    _device_read = "internal/devices/read"
    _device_last_read = "internal/devices/last_read"
    _device_readback = "internal/devices/readback"
    _device_req_status = "internal/devices/req_status"

    # device config
    _device_config_request = "internal/devices/config_request"
    _device_config = "internal/devices/config"
    _device_info = "internal/devices/info"

    # scan queue
    _scan_queue_modification = "internal/queue/queue_modification"
    _scan_queue_modification_request = "internal/queue/queue_modification_request"
    _scan_queue_insert = "internal/queue/queue_insert"
    _scan_queue_request = "internal/queue/queue_request"
    _scan_queue_request_response = "internal/queue/queue_request_response"
    _scan_queue_status = "internal/queue/queue_status"

    # scan info
    _scan_status = "scans/scan_status"
    _available_scans = "scans/available_scans"
    _scan_segment = "scans/scan_segment"
    _bluesky_events = "scans/bluesky-events"

    # instructions
    _device_instructions = "internal/devices/instructions"
    _device_rpc = "internal/devices/rpc"
    _pre_scan_macros = "internal/pre_scan_macros"
    _post_scan_macros = "internal/post_scan_macros"

    # log
    _log = "internal/log"
    _alarms = "internal/alarms"

    # service
    _services_status = "internal/services/status"

    ##########

    # devices feedback
    @classmethod
    def device_status(self, device: str):
        return f"{self._device_status}/{device}"

    @classmethod
    def device_read(self, device: str):
        return f"{self._device_read}/{device}"

    @classmethod
    def device_last_read(self, device: str):
        return f"{self._device_last_read}/{device}"

    @classmethod
    def device_readback(self, device: str):
        return f"{self._device_readback}/{device}"

    @classmethod
    def device_req_status(self, device: str):
        return f"{self._device_req_status}/{device}"

    # device config
    @classmethod
    def device_config_request(self):
        return self._device_config_request

    @classmethod
    def device_config(self):
        return self._device_config

    @classmethod
    def device_info(self, device: str):
        return f"{self._device_info}/{device}"

    # scan queue
    @classmethod
    def scan_queue_modification(self):
        return self._scan_queue_modification

    @classmethod
    def scan_queue_modification_request(self):
        return self._scan_queue_modification_request

    @classmethod
    def scan_queue_insert(self):
        return self._scan_queue_insert

    @classmethod
    def scan_queue_request(self):
        return self._scan_queue_request

    @classmethod
    def scan_queue_request_response(self):
        return self._scan_queue_request_response

    @classmethod
    def scan_queue_status(self):
        return self._scan_queue_status

    # scan info
    @classmethod
    def scan_status(self):
        return self._scan_status

    @classmethod
    def available_scans(self):
        return self._available_scans

    @classmethod
    def bluesky_events(self):
        return self._bluesky_events

    @classmethod
    def scan_segment(self):
        return self._scan_segment

    # instructions
    @classmethod
    def device_instructions(self):
        return self._device_instructions

    @classmethod
    def device_rpc(self, rpc_id: str):
        return f"{self._device_rpc}/{rpc_id}"

    @classmethod
    def pre_scan_macros(self):
        return self._pre_scan_macros

    @classmethod
    def post_scan_macros(self):
        return self._post_scan_macros

    # log
    @classmethod
    def log(self):
        return self._log

    @classmethod
    def alarm(self):
        return self._alarms

    @classmethod
    def service_status(self, service_id: str):
        return f"{self._services_status}/{service_id}"
