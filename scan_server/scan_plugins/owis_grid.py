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
import uuid

from scan_server.scans import FlyScanBase, ScanArgType, ScanAbortion
from bec_lib.core import MessageEndpoints, BECMessage

from bec_lib.core import bec_logger

logger = bec_logger.logger


class OwisGrid(FlyScanBase):
    scan_name = "owis_grid"
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

        # Keep the shutter open for longer to allow acquisitions to fly in
        self.shutter_additional_width = 0.15


        ########### Owis stage parameters
        # scanning related parameters
        self.stepping_y = abs(self.start_y - self.end_y) / interval_y
        self.stepping_x = abs(self.start_x - self.end_x) / interval_x
        
        #Standard parameter for owis stages!!
        self.high_velocity = 10 # mm/s
        self.high_acc_time = 0.2  # s 
        self.base_velocity = 0.0625
        self.add_pre_move_time = 0.0 #s
        self.backlash_distance = 0.125

        self.sign = 1
        
        
        #Relevant parameters for scan
        self.target_velocity = self.stepping_y / (self.exp_time+ self.readout_time)
        self.acc_time = (self.target_velocity -self.base_velocity)/(self.high_velocity-self.base_velocity)*self.high_acc_time
        self.premove_distance = 0.5*(self.target_velocity+self.base_velocity)*self.acc_time  + self.add_pre_move_time*self.target_velocity
        self.time_offset_snake = self.backlash_distance/self.target_velocity

        #Checks and set acc_time and premove for the designated scan
        if self.target_velocity > self.high_velocity or self.target_velocity < self.base_velocity:
            raise ScanAbortion(f'Requested velocity of {self.target_velocity} exceeds {self.high_velocity}')
        

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
            delay=0.01,
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

        status_ddg_fsh_ttlwidth = yield from self.stubs.send_rpc_and_wait(
            "ddg_fsh", "burst_disable"
        )

        # Set width of FSH opening to 0
        status_ddg_fsh_ttlwidth = yield from self.stubs.send_rpc_and_wait(
            "ddg_fsh", "set_channels", 
            "width", 0, 
            channels = ['channelCD'],
        )
        yield from self.stubs.send_rpc_and_wait(
            "ddg_fsh", "set_channels", 
            "width", 0, 
            channels = ['channelEF', "channelGH"],
        )
        time.sleep(0.05)
        trigger_ddg_fsh = yield from self.stubs.send_rpc_and_wait("ddg_fsh", "trigger")
        time.sleep(0.05)
        status_ddg_fsh_ttlwidth = yield from self.stubs.send_rpc_and_wait(
            "ddg_fsh", "set_channels", 
            "width", (self.interval_y * (self.exp_time+self.readout_time) + self.shutter_additional_width ), 
            channels = ['channelCD'],
        )
        status_ddg_fsh_ttlwidth = yield from self.stubs.send_rpc_and_wait(
            "ddg_fsh", "set_channels", 
            "width", 0, 
            channels = ['channelAB'],
        )
        
        #Software trigger on 
        status_ddg_mcs_ttldelay = yield from self.stubs.send_rpc_and_wait(
            "ddg_mcs", "set_channels", "delay", 0
        )

        # Set ddg_mcs on ext trigger from ddg_detectors
        status_ddg_mcs_source = yield from self.stubs.send_rpc_and_wait("ddg_mcs", "source.set", 1)
        # Set ddg_detectors and ddg_fsh to software trigger
        status_ddg_detectors_source = yield from self.stubs.send_rpc_and_wait("ddg_detectors", "source.set", 5)
        status_ddg_fsh_source = yield from self.stubs.send_rpc_and_wait("ddg_fsh", "source.set", 5)

        status_ddg_mcs_source.wait()
        status_ddg_detectors_source.wait()
        status_ddg_fsh_source.wait()
        

        #Set motor speed

        status_prepos = yield from self.stubs.send_rpc_and_wait(f'samy', "move", (self.start_y-self.premove_distance))
        status_prepos.wait()
        
        status_speed = yield from self.stubs.send_rpc_and_wait(f'samy', "velocity.put", self.target_velocity)
        status_acc = yield from self.stubs.send_rpc_and_wait(f'samy', "acceleration.put", self.acc_time)
        # Read out primary devices once at start and once at end of fly scan
        yield from self.stubs.read_and_wait(
            group="primary", wait_group="readout_primary", pointID=self.pointID
        )
        self.pointID += 1
        start=time.time()
        for ii in range(self.interval_x):
            #Set speed and acceleration
            logger.info(f'Start point, run {ii}: {time.time()-start}')
            status_speed = yield from self.stubs.send_rpc_and_wait(f'samy', "velocity.put", self.target_velocity)
            logger.info(f'Time passed velocity: {time.time()-start}')
            status_acc = yield from self.stubs.send_rpc_and_wait(f'samy', "acceleration.put", self.acc_time)
            logger.info(f'Time passed acceleration: {time.time()-start}')
            # yield from self.stubs.set(device = 'samy.velocity', value = self.target_velocity)
            # yield from self.stubs.set(device = 'samy.acceleration', value = self.acc_time)
            #time.sleep(0.01)

            # Start motion and send triggers
            yield from self.stubs.set(device = "samy", value =  (self.end_y+(self.sign*self.premove_distance)), wait_group= 'flyer')
            trigger_ddg_fsh = yield from self.stubs.send_rpc_and_wait("ddg_fsh", "trigger")
            #logger.info(self.acc_time)
            # if ii%2==0:
            time.sleep(self.acc_time)
            # else:
            #     time.sleep(self.acc_time + self.time_offset_snake)
            logger.info(f'{time.time()-start}, after sleep of {self.acc_time}')
            trigger_ddg_detectors = yield from self.stubs.send_rpc_and_wait("ddg_detectors", "trigger")
            #Wait for motion to finish
            yield from self.stubs.wait(device = "samy", wait_group="flyer", wait_type="move")
            logger.info(f'Finished Scan after {time.time()-start}')
            #Step yaxis
            #yield from self.stubs.set(device =f'samx', value =(self.start_x + ii*self.stepping_x), wait_group = 'flyer')
            yield from self.stubs.set(device =f'samx', value =(self.start_x - ii*self.stepping_x), wait_group = 'motion')

            #TODO fly scans -> swapping start and end
            # stored = self.start_y
            # self.start_y = self.end_y
            # self.end_y = stored
            # self.sign*=(-1)*self.sign
            logger.info(f'Time before velocity after scan: {time.time()-start}')
            status_speed = yield from self.stubs.send_rpc_and_wait(f'samy', "velocity.put", self.high_velocity)
            logger.info(f'Time after velocity: {time.time()-start}')
            status_acc = yield from self.stubs.send_rpc_and_wait(f'samy', "acceleration.put", self.high_acc_time)
            logger.info(f'Time after acceleration: {time.time()-start}')
            # yield from self.stubs.set(device = 'samy.velocity', value = self.high_velocity)
            # yield from self.stubs.set(device = 'samy.acceleration', value = self.high_acc_time)

            # Move back to start
            logger.info(f'Start moving back {time.time()-start}')
            status_prepos = yield from self.stubs.send_rpc_and_wait(f'samy', "move", (self.start_y-self.premove_distance))
            
            status_prepos.wait()
            logger.info(f'Finished moving {time.time()-start}')

        status_speed = yield from self.stubs.send_rpc_and_wait(f'samy', "velocity.put", self.high_velocity)
        status_acc = yield from self.stubs.send_rpc_and_wait(f'samy', "acceleration.put", self.high_acc_time)

        yield from self.stubs.read_and_wait(
            group="primary", wait_group="readout_primary", pointID=self.pointID
        )
        self.pointID += 1

        # while True:
        #     # readout the primary device and wait for the fly scan to finish
        #     yield from self.stubs.read_and_wait(
        #         group="primary", wait_group="readout_primary", pointID=self.pointID
        #     )
        #     self.pointID += 1
        #     status = self.stubs.get_req_status(
        #         device="samx", RID=self.metadata["RID"], DIID=target_diid
        #     )
        #     if status:
        #         break
        #     time.sleep(self.sleep_time)
        #     if self.scan_progress() > int(self.timeout_scan_abortion / self.sleep_time):
        #         logger.info(f'would have raised a scan abortion here')
        #         raise ScanAbortion()

        #     try:
        #         logger.info(f'Scan progress check {self.scan_progress()} and {int(self.timeout_scan_abortion/self.sleep_time)}')
        #         logger.info(f'Potential scan abortion {self.scan_progress() > int(self.timeout_scan_abortion/self.sleep_time)}')
        #         if self.scan_progress() > int(self.timeout_scan_abortion/self.sleep_time):
        #             logger.info('Testing Scan abortion, would have raised here!')
        #     except Exception as exc:
        #         logger.info(f'{exc}')
