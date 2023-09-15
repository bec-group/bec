"""
SCAN PLUGINS

All new scans should be derived from ScanBase. ScanBase provides various methods that can be customized and overriden
but they are executed in a specific order:

- self.initialize                        # initialize the class if needed
- self.read_scan_motors                  # used to retrieve the start position (and the relative position shift if needed)
- self.prepare_positions                 # prepare the positions for the scan. The preparation is split into multiple sub fuctions:
    - self._calculate_positions          # calculate the positions
    - self._set_positions_offset         # apply the previously retrieved scan position shift (if needed)
    - self._check_limits                 # tests to ensure the limits won't be reached
- self.open_scan                         # send an open_scan message including the scan name, the number of points and the scan motor names
- self.stage                             # stage all devices for the upcoming acquisiton
- self.run_baseline_readings             # read all devices to get a baseline for the upcoming scan
- self.scan_core                         # run a loop over all position
    - self._at_each_point(ind, pos)      # called at each position with the current index and the target positions as arguments
- self.finalize                          # clean up the scan, e.g. move back to the start position; wait everything to finish
- self.unstage                           # unstage all devices that have been staged before
- self.cleanup                           # send a close scan message and perform additional cleanups if needed
"""

import time

from scan_server.scans import FlyScanBase, ScanArgType, ScanAbortion
from bec_lib.core import MessageEndpoints, BECMessage

from bec_lib.core import bec_logger

logger = bec_logger.logger


class SgalilGrid(FlyScanBase):
    scan_name = "sgalil_grid"
    scan_report_hint = "scan_progress"
    required_kwargs = []
    arg_input = []
    arg_bundle_size = len(arg_input)
    enforce_sync = False

    def __init__(
        self,
        start_y: float,
        end_y: float,
        interval_y: int,
        start_x: float,
        end_x: float,
        interval_x: int,
        *args,
        exp_time: float = 0.1,
        readout_time: float = 0.1,
        **kwargs
    ):
        """
        SGalil-based grid scan.

        Args:
            start_y (float): start position of y axis (fast axis)
            end_y (float): end position of y axis (fast axis)
            interval_y (int): number of points in y axis
            start_x (float): start position of x axis (slow axis)
            end_x (float): end position of x axis (slow axis)
            interval_x (int): number of points in x axis
            exp_time (float): exposure time in seconds. Default is 0.1s
            readout_time (float): readout time in seconds, minimum of 3e-3s (3ms)

        Exp:
           scans.sgalil_grid(start_y = val1, end_y= val1, interval_y = val1, start_x = val1, end_x = val1, interval_x = val1, exp_time = 0.02, readout_time = 3e-3)


        """
        super().__init__(*args, **kwargs)
        # Always scan from positive  x & y to negative x & y
        if start_y > end_y:
            self.start_y = start_y
            self.end_y = end_y
        else:
            self.start_y = end_y
            self.end_y = start_y
        if start_x > end_x:
            self.start_x = start_x
            self.end_x = end_x
        else:
            self.start_x = end_x
            self.end_x = start_x
        self.interval_y = interval_y
        self.interval_x = interval_x
        self.exp_time = exp_time
        self.readout_time = readout_time
        self.num_pos = int(interval_x * interval_y)
        self.scan_motors = ["samx", "samy"]
        # Scan progress related variables
        self.timeout_progress = 0
        self.progress_point = 0
        self.timeout_scan_abortion = 10  # 42 # duty cycles of scan segment update
        self.sleep_time = 1

    def scan_report_instructions(self):
        if not self.scan_report_hint:
            yield None
            return
        yield from self.stubs.scan_report_instruction({"scan_progress": ["mcs"]})

    def pre_scan(self):
        yield from self._move_and_wait([self.start_x, self.start_y])
        yield from self.stubs.pre_scan()
        # TODO move to start position

    def scan_progress(self) -> int:
        """Timeout of the progress bar. This gets updated in the frequency of scan segments"""
        raw_msg = self.device_manager.producer.get(MessageEndpoints.device_progress("mcs"))
        if not raw_msg:
            self.timeout_progress += 1
            return self.timeout_progress
        msg = BECMessage.DeviceStatusMessage.loads(raw_msg)
        if not msg:
            self.timeout_progress += 1
            return self.timeout_progress
        # TODO which update is that!
        updated_progress = int(msg.content["status"]["value"])
        if updated_progress == int(self.progress_point):
            self.timeout_progress += 1
            return self.timeout_progress
        else:
            self.timeout_progress = 0
            self.progress_point = updated_progress
            return self.timeout_progress

    def scan_core(self):
        """
        This is the main event loop.
        """

        # set up the delay generators
        status_ddg_detectors_burst = yield from self.stubs.send_rpc_and_wait(
            "ddg_detectors",
            "burst_enable",
            count=self.interval_y,
            delay=0,
            period=(self.exp_time + self.readout_time),
            config="first",
        )
        status_ddg_mcs_burst = yield from self.stubs.send_rpc_and_wait(
            "ddg_mcs",
            "burst_enable",
            count=self.interval_y,
            delay=0,
            period=(self.exp_time + self.readout_time),
            config="first",
        )
        # Disable burst mod on DDF for fsh and EN of MCS card
        status_ddg_fsh_burst = yield from self.stubs.send_rpc_and_wait(
            "ddg_fsh",
            "burst_disable")
        # Set width of FSH opening to 0
        status_ddg_fsh_ttlwidth = yield from self.stubs.send_rpc_and_wait(
            "ddg_fsh", "set_channels", "width", 0, channels = ['channelCD'],
        )

        # TODO disable fsh ddg bc SGalil trigger it directly
        # Setup triggering
        status_ddg_detectors_source = yield from self.stubs.send_rpc_and_wait(
            "ddg_detectors", "source.set", 2
        )
        status_ddg_mcs_source = yield from self.stubs.send_rpc_and_wait("ddg_mcs", "source.set", 1)
        # Setup mcs_points per line
        # status_mcs_points_per_line = yield from self.stubs.send_rpc_and_wait(
        #     "mcs", "num_use_all.set", self.interval_y + 1
        # )
        # status_mcs_lines = yield from self.stubs.send_rpc_and_wait(
        #     "mcs", "num_lines.set", self.interval_x
        # )
        
        # status_ddg_mcs_ttlwidth = yield from self.stubs.send_rpc_and_wait(
        #     "ddg_mcs", "set_channels", "width", 3e-3
        # )
        status_ddg_mcs_ttldelay = yield from self.stubs.send_rpc_and_wait(
            "ddg_mcs", "set_channels", "delay", 0
        )

        # wait for the delay generators to finish setting up
        status_ddg_detectors_source.wait()
        status_ddg_mcs_source.wait()
        trigger_ddg_fsh = yield from self.stubs.send_rpc_and_wait("ddg_fsh", "trigger")
        #trigger_ddg_fsh.wait()
        # status_mcs_points_per_line.wait()
        # status_mcs_lines.wait()

        yield from self.stubs.kickoff(
            device="samx",
            parameter={
                "start_y": self.start_y,
                "end_y": self.end_y,
                "interval_y": self.interval_y,
                "start_x": self.start_x,
                "end_x": self.end_x,
                "interval_x": self.interval_x,
                "exp_time": self.exp_time,
                "readout_time": self.readout_time,
            },
        )
        target_diid = self.DIID - 1
        while True:
            # readout the primary device and wait for the fly scan to finish
            yield from self.stubs.read_and_wait(
                group="primary", wait_group="readout_primary", pointID=self.pointID
            )
            self.pointID += 1
            status = self.stubs.get_req_status(
                device="samx", RID=self.metadata["RID"], DIID=target_diid
            )
            if status:
                break
            time.sleep(self.sleep_time)
            if self.scan_progress() > int(self.timeout_scan_abortion / self.sleep_time):
                logger.info(f'would have raised a scan abortion here')
                # raise ScanAbortion()

            # try:
            #     logger.info(f'Scan progress check {self.scan_progress()} and {int(self.timeout_scan_abortion/self.sleep_time)}')
            #     logger.info(f'Potential scan abortion {self.scan_progress() > int(self.timeout_scan_abortion/self.sleep_time)}')
            #     if self.scan_progress() > int(self.timeout_scan_abortion/self.sleep_time):
            #         logger.info('Testing Scan abortion, would have raised here!')
            # except Exception as exc:
            #     logger.info(f'{exc}')
