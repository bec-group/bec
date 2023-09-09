import ast
import enum
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
from bec_lib.core import BECMessage, DeviceManagerBase, MessageEndpoints, bec_logger

from .errors import LimitError, ScanAbortion
from .path_optimization import PathOptimizerMixin
from .scan_stubs import ScanStubs

DeviceMsg = BECMessage.DeviceInstructionMessage
ScanMsg = BECMessage.ScanQueueMessage

logger = bec_logger.logger


class ScanArgType(str, enum.Enum):
    DEVICE = "device"
    FLOAT = "float"
    INT = "int"
    BOOL = "boolean"
    STR = "str"
    LIST = "list"
    DICT = "dict"


def unpack_scan_args(scan_args: Dict[str, Any]) -> list:
    """unpack_scan_args unpacks the scan arguments and returns them as a tuple.

    Args:
        scan_args (Dict[str, Any]): scan arguments

    Returns:
        list: list of arguments
    """
    args = []
    if not scan_args:
        return args
    if not isinstance(scan_args, dict):
        return scan_args
    for cmd_name, cmd_args in scan_args.items():
        args.append(cmd_name)
        args.extend(cmd_args)
    return args


def get_2D_raster_pos(axis, snaked=True):
    """get_2D_raster_post calculates and returns the positions for a 2D

    snaked==True:
        ->->->->-
        -<-<-<-<-
        ->->->->-
    snaked==False:
        ->->->->-
        ->->->->-
        ->->->->-

    Args:
        axis (list): list of positions for each axis
        snaked (bool, optional): If true, the positions will be calculcated for a snake scan. Defaults to True.

    Returns:
        array: calculated positions
    """

    x_grid, y_grid = np.meshgrid(axis[0], axis[1])
    if snaked:
        y_grid.T[::2] = np.fliplr(y_grid.T[::2])
    x_flat = x_grid.T.ravel()
    y_flat = y_grid.T.ravel()
    positions = np.vstack((x_flat, y_flat)).T
    return positions


# pylint: disable=too-many-arguments
def get_fermat_spiral_pos(
    m1_start, m1_stop, m2_start, m2_stop, step=1, spiral_type=0, center=False
):
    """get_fermat_spiral_pos calculates and returns the positions for a Fermat spiral scan.

    Args:
        m1_start (float): start position motor 1
        m1_stop (float): end position motor 1
        m2_start (float): start position motor 2
        m2_stop (float): end position motor 2
        step (float, optional): Step size. Defaults to 1.
        spiral_type (float, optional): Angular offset in radians that determines the shape of the spiral.
        A spiral with spiral_type=2 is the same as spiral_type=0. Defaults to 0.
        center (bool, optional): Add a center point. Defaults to False.

    Returns:
        array: calculated positions in the form [[m1, m2], ...]
    """
    positions = []
    phi = 2 * np.pi * ((1 + np.sqrt(5)) / 2.0) + spiral_type * np.pi

    start = int(not center)

    length_axis1 = abs(m1_stop - m1_start)
    length_axis2 = abs(m2_stop - m2_start)
    n_max = int(length_axis1 * length_axis2 * 3.2 / step / step)

    for ii in range(start, n_max):
        radius = step * 0.57 * np.sqrt(ii)
        if abs(radius * np.sin(ii * phi)) > length_axis1 / 2:
            continue
        if abs(radius * np.cos(ii * phi)) > length_axis2 / 2:
            continue
        positions.extend([(radius * np.sin(ii * phi), radius * np.cos(ii * phi))])
    return np.array(positions)


def get_round_roi_scan_positions(lx: float, ly: float, dr: float, nth: int, cenx=0, ceny=0):
    """
    get_round_roi_scan_positions calculates and returns the positions for a round scan in a rectangular region of interest.

    Args:
        lx (float): length in x
        ly (float): length in y
        dr (float): step size
        nth (int): number of angles in the inner ring
        cenx (int, optional): center in x. Defaults to 0.
        ceny (int, optional): center in y. Defaults to 0.

    Returns:
        array: calculated positions in the form [[x, y], ...]
    """
    positions = []
    nr = 1 + int(np.floor(max([lx, ly]) / dr))
    for ir in range(1, nr + 2):
        rr = ir * dr
        dth = 2 * np.pi / (nth * ir)
        pos = [
            (rr * np.cos(ith * dth) + cenx, rr * np.sin(ith * dth) + ceny)
            for ith in range(nth * ir)
            if np.abs(rr * np.cos(ith * dth)) < lx / 2 and np.abs(rr * np.sin(ith * dth)) < ly / 2
        ]
        positions.extend(pos)
    return np.array(positions)


def get_round_scan_positions(r_in: float, r_out: float, nr: int, nth: int, cenx=0, ceny=0):
    """
    get_round_scan_positions calculates and returns the positions for a round scan.

    Args:
        r_in (float): inner radius
        r_out (float): outer radius
        nr (int): number of radii
        nth (int): number of angles in the inner ring
        cenx (int, optional): center in x. Defaults to 0.
        ceny (int, optional): center in y. Defaults to 0.

    Returns:
        array: calculated positions in the form [[x, y], ...]

    """
    positions = []
    dr = (r_in - r_out) / nr
    for ir in range(1, nr + 2):
        rr = r_in + ir * dr
        dth = 2 * np.pi / (nth * ir)
        positions.extend(
            [
                (rr * np.sin(ith * dth) + cenx, rr * np.cos(ith * dth) + ceny)
                for ith in range(nth * ir)
            ]
        )
    return np.array(positions)


class RequestBase(ABC):
    """
    Base class for all scan requests.
    """

    scan_name = ""
    scan_report_hint = None
    arg_input = [ScanArgType.DEVICE]
    arg_bundle_size = len(arg_input)
    required_kwargs = []
    return_to_start_after_abort = False

    def __init__(
        self,
        *args,
        device_manager: DeviceManagerBase = None,
        parameter=None,
        metadata=None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.parameter = parameter if parameter is not None else {}
        self.caller_args = self.parameter.get("args", {})
        self.caller_kwargs = self.parameter.get("kwargs", {})
        self.metadata = metadata
        self.device_manager = device_manager
        self.DIID = 0
        self.scan_motors = []
        self.positions = []
        self._pre_scan_macros = []
        self._scan_report_devices = None
        self._get_scan_motors()
        self.readout_priority = {"monitored": [], "baseline": [], "ignored": []}
        self.update_readout_priority()
        if metadata is None:
            self.metadata = {}
        self.stubs = ScanStubs(
            producer=self.device_manager.producer,
            device_msg_callback=self.device_msg_metadata,
        )

    @property
    def scan_report_devices(self):
        """devices to be included in the scan report"""
        if self._scan_report_devices is None:
            return self.scan_motors
        return self._scan_report_devices

    @scan_report_devices.setter
    def scan_report_devices(self, devices: list):
        self._scan_report_devices = devices

    def device_msg_metadata(self):
        default_metadata = {"readout_priority": "monitored", "DIID": self.DIID}
        metadata = {**default_metadata, **self.metadata}
        self.DIID += 1
        return metadata

    @staticmethod
    def _get_func_name_from_macro(macro: str):
        return ast.parse(macro).body[0].name

    def run_pre_scan_macros(self):
        """run pre scan macros if any"""
        macros = self.device_manager.producer.lrange(MessageEndpoints.pre_scan_macros(), 0, -1)
        for macro in macros:
            macro = macro.decode().strip()
            func_name = self._get_func_name_from_macro(macro)
            exec(macro)
            eval(func_name)(self.device_manager.devices, self)

    def initialize(self):
        self.run_pre_scan_macros()

    def _check_limits(self):
        logger.debug("check limits")
        for ii, dev in enumerate(self.scan_motors):
            low_limit, high_limit = (
                self.device_manager.devices[dev]._config["deviceConfig"].get("limits", [0, 0])
            )
            if low_limit >= high_limit:
                # if both limits are equal or low > high, no restrictions ought to be applied
                return
            for pos in self.positions:
                pos_axis = pos[ii]
                if not low_limit <= pos_axis <= high_limit:
                    raise LimitError(
                        f"Target position {pos} for motor {dev} is outside of range: [{low_limit}, {high_limit}]"
                    )

    def _get_scan_motors(self):
        if len(self.caller_args) == 0:
            return
        if self.arg_bundle_size:
            self.scan_motors = list(self.caller_args.keys())
            return
        for motor in self.caller_args:
            if motor not in self.device_manager.devices:
                continue
            self.scan_motors.append(motor)

    def update_readout_priority(self):
        """update the readout priority for this request. Typically the monitored devices should also include the scan motors."""
        self.readout_priority["monitored"] = self.scan_motors

    @abstractmethod
    def run(self):
        pass


class ScanBase(RequestBase, PathOptimizerMixin):
    """
    Base class for all scans. The following methods are called in the following order during the scan
    1. initialize
        - run_pre_scan_macros
    2. read_scan_motors
    3. prepare_positions
        - _calculate_positions
        - _optimize_trajectory
        - _set_position_offset
        - _check_limits
    4. open_scan
    5. stage
    6. run_baseline_reading
    7. scan_core
    8. finalize
    9. unstage
    10. cleanup

    A subclass of ScanBase must implement the following methods:
    - _calculate_positions

    Attributes:
        scan_name (str): name of the scan
        scan_report_hint (str): hint for the scan report
        scan_type (str): scan type. Can be "step" or "fly"
        arg_input (list): list of scan argument types
        arg_bundle_size (int): number of arguments that are bundled together
        required_kwargs (list): list of required kwargs
        return_to_start_after_abort (bool): if True, the scan will return to the start position after an abort
    """

    scan_name = ""
    scan_report_hint = None
    scan_type = "step"
    arg_input = [ScanArgType.DEVICE]
    arg_bundle_size = len(arg_input)
    required_kwargs = ["required"]
    return_to_start_after_abort = True

    # perform pre-move action before the pre_scan trigger is sent
    pre_move = True

    # synchronize primary readings with the scan motor readings. Set to False if the master device
    # does not provide single readouts or are too fast to be synchronized with primary readings
    enforce_sync = True

    def __init__(
        self,
        *args,
        device_manager: DeviceManagerBase = None,
        parameter: dict = None,
        exp_time: float = 0,
        readout_time: float = 0,
        acquisition_config: dict = None,
        settling_time: float = 0,
        relative: bool = False,
        burst_at_each_point: int = 1,
        frames_per_trigger: int = 1,
        optim_trajectory: str = None,
        metadata: dict = None,
        **kwargs,
    ):
        super().__init__(
            *args,
            device_manager=device_manager,
            parameter=parameter,
            metadata=metadata,
            **kwargs,
        )
        self.DIID = 0
        self.pointID = 0
        self.exp_time = exp_time
        self.readout_time = readout_time
        self.acquisition_config = acquisition_config
        self.settling_time = settling_time
        self.relative = relative
        self.burst_at_each_point = burst_at_each_point
        self.frames_per_trigger = frames_per_trigger
        self.optim_trajectory = optim_trajectory
        self.burst_index = 0

        self.start_pos = np.repeat(0, len(self.scan_motors)).tolist()
        self.positions = []
        self.num_pos = None

        if self.scan_name == "":
            raise ValueError("scan_name cannot be empty")

        if acquisition_config is None or "default" not in acquisition_config:
            self.acquisition_config = {
                "default": {
                    "exp_time": self.exp_time,
                    "readout_time": self.readout_time,
                }
            }

    def read_scan_motors(self):
        """read the scan motors"""
        yield from self.stubs.read_and_wait(device=self.scan_motors, wait_group="scan_motor")

    @abstractmethod
    def _calculate_positions(self) -> None:
        """Calculate the positions"""
        pass

    def _optimize_trajectory(self):
        if not self.optim_trajectory:
            return
        if self.optim_trajectory == "corridor":
            self.positions = self.optimize_corridor(self.positions)
            return
        return

    def prepare_positions(self):
        """prepare the positions for the scan"""
        self._calculate_positions()
        self._optimize_trajectory()
        self.num_pos = len(self.positions) * self.burst_at_each_point
        yield from self._set_position_offset()
        self._check_limits()

    def open_scan(self):
        """open the scan"""
        positions = self.positions if isinstance(self.positions, list) else self.positions.tolist()
        yield from self.stubs.open_scan(
            scan_motors=self.scan_motors,
            readout_priority=self.readout_priority,
            num_pos=self.num_pos,
            positions=positions,
            scan_name=self.scan_name,
            scan_type=self.scan_type,
        )

    def stage(self):
        """call the stage procedure"""
        yield from self.stubs.stage()

    def run_baseline_reading(self):
        """perform a reading of all baseline devices"""
        yield from self.stubs.baseline_reading()

    def _set_position_offset(self):
        self.start_pos = []
        for dev in self.scan_motors:
            val = yield from self.stubs.send_rpc_and_wait(dev, "read")
            self.start_pos.append(val[dev].get("value"))
        if self.relative:
            self.positions += self.start_pos

    def close_scan(self):
        """close the scan"""
        yield from self.stubs.close_scan()

    def scan_core(self):
        """perform the scan core procedure"""
        for ind, pos in self._get_position():
            for self.burst_index in range(self.burst_at_each_point):
                yield from self._at_each_point(ind, pos)
            self.burst_index = 0

    def return_to_start(self):
        """return to the start position"""
        yield from self._move_and_wait(self.start_pos)

    def finalize(self):
        """finalize the scan"""
        yield from self.return_to_start()
        yield from self.stubs.wait(wait_type="read", group="primary", wait_group="readout_primary")

    def unstage(self):
        """call the unstage procedure"""
        yield from self.stubs.unstage()

    def cleanup(self):
        """call the cleanup procedure"""
        yield from self.close_scan()

    def _at_each_point(self, ind=None, pos=None):
        yield from self._move_and_wait(pos)
        if ind > 0:
            yield from self.stubs.wait(
                wait_type="read", group="primary", wait_group="readout_primary"
            )
        time.sleep(self.settling_time)
        yield from self.stubs.trigger(group="trigger", pointID=self.pointID)
        yield from self.stubs.wait(wait_type="trigger", group="trigger", wait_time=self.exp_time)
        yield from self.stubs.read(
            group="primary", wait_group="readout_primary", pointID=self.pointID
        )
        yield from self.stubs.wait(
            wait_type="read", group="scan_motor", wait_group="readout_primary"
        )

        self.pointID += 1

    def _move_and_wait(self, pos):
        if not isinstance(pos, list) and not isinstance(pos, np.ndarray):
            pos = [pos]
        if len(pos) == 0:
            return
        for ind, val in enumerate(self.scan_motors):
            yield from self.stubs.set(device=val, value=pos[ind], wait_group="scan_motor")

        yield from self.stubs.wait(wait_type="move", group="scan_motor", wait_group="scan_motor")

    def _get_position(self):
        for ind, pos in enumerate(self.positions):
            yield (ind, pos)

    def scan_report_instructions(self):
        yield None

    def pre_scan(self):
        """
        pre scan procedure. This method is called before the scan_core method and can be used to
        perform additional tasks before the scan is started. This
        """
        if self.pre_move and len(self.positions) > 0:
            yield from self._move_and_wait(self.positions[0])
        yield from self.stubs.pre_scan()

    def run(self):
        """run the scan. This method is called by the scan server and is the main entry point for the scan."""
        self.initialize()
        yield from self.read_scan_motors()
        yield from self.prepare_positions()
        yield from self.scan_report_instructions()
        yield from self.open_scan()
        yield from self.stage()
        yield from self.run_baseline_reading()
        yield from self.pre_scan()
        yield from self.scan_core()
        yield from self.finalize()
        yield from self.unstage()
        yield from self.cleanup()

    @classmethod
    def scan(cls, *args, **kwargs):
        scan = cls(args, **kwargs)
        yield from scan.run()


class FlyScanBase(ScanBase):
    scan_type = "fly"
    pre_move = False

    def _get_flyer_status(self) -> List:
        flyer = self.scan_motors[0]
        producer = self.device_manager.producer

        pipe = producer.pipeline()
        producer.lrange(MessageEndpoints.device_req_status(self.metadata["RID"]), 0, -1, pipe)
        producer.get(MessageEndpoints.device_readback(flyer), pipe)
        return pipe.execute()

    def scan_core(self):
        yield from self.stubs.kickoff(
            device=self.scan_motors[0],
            parameter=self.caller_kwargs,
        )
        yield from self.stubs.complete(device=self.scan_motors[0])
        target_diid = self.DIID - 1

        while True:
            status = self.stubs.get_req_status(
                device=self.scan_motors[0], RID=self.metadata["RID"], DIID=target_diid
            )
            progress = self.stubs.get_device_progress(
                device=self.scan_motors[0], RID=self.metadata["RID"]
            )
            if progress:
                self.num_pos = progress
            if status:
                break
            time.sleep(1)

    def _calculate_positions(self) -> None:
        pass

    def read_scan_motors(self):
        yield None

    def prepare_positions(self):
        yield None


class ScanStub(RequestBase):
    pass


class OpenScanDef(ScanStub):
    scan_name = "open_scan_def"
    scan_report_hint = None

    def run(self):
        yield from self.stubs.open_scan_def()


class CloseScanDef(ScanStub):
    scan_name = "close_scan_def"
    scan_report_hint = "table"

    def run(self):
        yield from self.stubs.close_scan_def()


class CloseScanGroup(ScanStub):
    scan_name = "close_scan_group"

    def run(self):
        yield from self.stubs.close_scan_group()


class DeviceRPC(ScanStub):
    scan_name = "device_rpc"
    arg_input = [
        ScanArgType.DEVICE,
        ScanArgType.STR,
        ScanArgType.LIST,
        ScanArgType.DICT,
    ]
    arg_bundle_size = len(arg_input)
    scan_report_hint = None

    def _get_scan_motors(self):
        pass

    def run(self):
        # different to calling self.device_rpc, this procedure will not wait for a reply and therefore not check any errors.
        yield from self.stubs.rpc(device=self.parameter.get("device"), parameter=self.parameter)


class Move(RequestBase):
    scan_name = "mv"
    arg_input = [ScanArgType.DEVICE, ScanArgType.FLOAT]
    arg_bundle_size = len(arg_input)
    scan_report_hint = None
    required_kwargs = ["relative"]

    def __init__(self, *args, relative=False, **kwargs):
        """
        Move device(s) to an absolute position
        Args:
            *args (Device, float): pairs of device / position arguments
            relative (bool): if True, move relative to current position

        Returns:
            ScanReport

        Examples:
            >>> scans.mv(dev.samx, 1, dev.samy,2)
        """
        super().__init__(**kwargs)
        self.relative = relative
        self.start_pos = np.repeat(0, len(self.scan_motors)).tolist()

    def _calculate_positions(self):
        self.positions = np.asarray([[val[0] for val in self.caller_args.values()]], dtype=float)

    def _at_each_point(self, pos=None):
        for ii, motor in enumerate(self.scan_motors):
            yield from self.stubs.set(
                device=motor,
                value=self.positions[0][ii],
                wait_group="scan_motor",
                metadata={"response": True},
            )

    def cleanup(self):
        pass

    def _set_position_offset(self):
        self.start_pos = []
        for dev in self.scan_motors:
            val = yield from self.stubs.send_rpc_and_wait(dev, "read")
            self.start_pos.append(val[dev].get("value"))
        if not self.relative:
            return
        self.positions += self.start_pos

    def prepare_positions(self):
        self._calculate_positions()
        yield from self._set_position_offset()
        self._check_limits()

    def scan_report_instructions(self):
        if not self.scan_report_hint:
            yield None
            return
        yield from self.stubs.scan_report_instruction(
            {
                "readback": {
                    "RID": self.metadata["RID"],
                    "devices": self.scan_motors,
                    "start": self.start_pos,
                    "end": self.positions[0],
                }
            }
        )

    def run(self):
        self.initialize()
        yield from self.prepare_positions()
        yield from self.scan_report_instructions()
        yield from self._at_each_point()


class UpdatedMove(Move):
    """
    Move device(s) to an absolute position and show live updates. This is a blocking call. For non-blocking use Move.
    Args:
        *args (Device, float): pairs of device / position arguments
        relative (bool): if True, move relative to current position

    Returns:
        ScanReport

    Examples:
        >>> scans.umv(dev.samx, 1, dev.samy,2)
    """

    scan_name = "umv"
    scan_report_hint = "readback"

    def _at_each_point(self, pos=None):
        for ii, motor in enumerate(self.scan_motors):
            yield from self.stubs.set(
                device=motor,
                value=self.positions[0][ii],
                wait_group="scan_motor",
            )

        for motor in self.scan_motors:
            yield from self.stubs.wait(wait_type="move", device=motor, wait_group="scan_motor")


class Scan(ScanBase):
    scan_name = "grid_scan"
    scan_report_hint = "table"
    arg_input = [
        ScanArgType.DEVICE,
        ScanArgType.FLOAT,
        ScanArgType.FLOAT,
        ScanArgType.INT,
    ]
    arg_bundle_size = len(arg_input)
    required_kwargs = ["relative"]

    def __init__(
        self,
        *args,
        exp_time=0,
        settling_time=0,
        relative=False,
        burst_at_each_point=1,
        **kwargs,
    ):
        """
        Scan two motors in a grid.

        Args:
            *args (Device, float, float, int): pairs of device / start / stop / steps arguments
            exp_time (float): exposure time in seconds. Default is 0.
            settling_time (float): settling time in seconds. Default is 0.
            relative (bool): if True, the motors will be moved relative to their current position. Default is False.
            burst_at_each_point (int): number of exposures at each point. Default is 1.

        Returns:
            ScanReport

        Examples:
            >>> scans.grid_scan(dev.motor1, -5, 5, 10, dev.motor2, -5, 5, 10, exp_time=0.1, relative=True)

        """
        super().__init__(**kwargs)
        self.relative = relative
        self.exp_time = exp_time
        self.settling_time = settling_time
        self.burst_at_each_point = burst_at_each_point
        self.axis = []

    def _calculate_positions(self):
        for _, val in self.caller_args.items():
            self.axis.append(np.linspace(val[0], val[1], val[2]))
        if len(self.axis) > 1:
            self.positions = get_2D_raster_pos(self.axis)
        else:
            self.positions = np.vstack(tuple(self.axis)).T


class FermatSpiralScan(ScanBase):
    scan_name = "fermat_scan"
    scan_report_hint = "table"
    required_kwargs = ["step", "relative"]
    arg_input = [ScanArgType.DEVICE, ScanArgType.FLOAT, ScanArgType.FLOAT]
    arg_bundle_size = len(arg_input)

    def __init__(
        self,
        *args,
        step=0.1,
        exp_time=0,
        settling_time=0,
        relative=False,
        burst_at_each_point=1,
        spiral_type=0,
        optim_trajectory=None,
        **kwargs,
    ):
        """
        A scan following Fermat's spiral.

        Args:
            *args: pairs of device / start position / end position / steps arguments
            step (float): step size in motor units. Default is 0.1.
            exp_time (float): exposure time in seconds. Default is 0.
            settling_time (float): settling time in seconds. Default is 0.
            relative (bool): if True, the motors will be moved relative to their current position. Default is False.
            burst_at_each_point (int): number of exposures at each point. Default is 1.
            spiral_type (float): type of spiral to use. Default is 0.
            optim_trajectory (str): trajectory optimization method. Default is None. Options are "corridor" and "none".

        Returns:
            ScanReport

        Examples:
            >>> scans.fermat_scan(dev.motor1, -5, 5, dev.motor2, -5, 5, step=0.5, exp_time=0.1, relative=True, optim_trajectory="corridor")

        """
        super().__init__(**kwargs)
        self.axis = []
        self.step = step
        self.exp_time = exp_time
        self.settling_time = settling_time
        self.relative = relative
        self.burst_at_each_point = burst_at_each_point
        self.spiral_type = spiral_type
        self.optim_trajectory = optim_trajectory

    def _calculate_positions(self):
        params = list(self.caller_args.values())
        self.positions = get_fermat_spiral_pos(
            params[0][0],
            params[0][1],
            params[1][0],
            params[1][1],
            step=self.step,
            spiral_type=self.spiral_type,
            center=False,
        )


class RoundScan(ScanBase):
    scan_name = "round_scan"
    scan_report_hint = "table"
    required_kwargs = ["relative"]
    arg_input = [
        ScanArgType.DEVICE,
        ScanArgType.DEVICE,
        ScanArgType.FLOAT,
        ScanArgType.FLOAT,
        ScanArgType.INT,
        ScanArgType.INT,
    ]
    arg_bundle_size = len(arg_input)

    def __init__(self, *args, relative=False, burst_at_each_point=1, **kwargs):
        """
        A scan following a round shell-like pattern.

        Args:
            *args: motor1, motor2, inner ring, outer ring, number of rings, number of positions in the first ring
            relative (bool): if True, the motors will be moved relative to their current position. Default is False.
            burst_at_each_point (int): number of exposures at each point. Default is 1.

        Returns:
            ScanReport

        Examples:
            >>> scans.round_scan(dev.motor1, dev.motor2, 0, 25, 5, 3, exp_time=0.1, relative=True)

        """
        super().__init__(**kwargs)
        self.relative = relative
        self.burst_at_each_point = burst_at_each_point
        self.axis = []

    def _get_scan_motors(self):
        caller_args = list(self.caller_args.items())[0]
        self.scan_motors = [caller_args[0], caller_args[1][0]]

    def _calculate_positions(self):
        params = list(self.caller_args.values())[0]
        self.positions = get_round_scan_positions(
            r_in=params[1], r_out=params[2], nr=params[3], nth=params[4]
        )


class ContLineScan(ScanBase):
    scan_name = "cont_line_scan"
    scan_report_hint = "table"
    required_kwargs = ["steps", "relative"]
    arg_input = [ScanArgType.DEVICE, ScanArgType.FLOAT, ScanArgType.FLOAT]
    arg_bundle_size = len(arg_input)
    scan_type = "step"

    def __init__(
        self,
        *args,
        exp_time=0,
        steps=10,
        relative=False,
        offset=100,
        burst_at_each_point=1,
        **kwargs,
    ):
        """
        A line scan for one or more motors.

        Args:
            *args (Device, float, float): pairs of device / start position / end position
            exp_time (float): exposure time in seconds. Default is 0.
            steps (int): number of steps. Default is 10.
            relative (bool): if True, the motors will be moved relative to their current position. Default is False.
            burst_at_each_point (int): number of exposures at each point. Default is 1.
            offset (float): offset in motor units. Default is 100.

        Returns:
            ScanReport

        Examples:
            >>> scans.cont_line_scan(dev.motor1, -5, 5, steps=10, exp_time=0.1, relative=True)

        """
        super().__init__(**kwargs)
        self.axis = []
        self.exp_time = exp_time
        self.steps = steps
        self.relative = relative
        self.burst_at_each_point = burst_at_each_point
        self.offset = offset

    def _calculate_positions(self) -> None:
        for _, val in self.caller_args.items():
            ax_pos = np.linspace(val[0], val[1], self.steps)
            self.axis.append(ax_pos)
        self.positions = np.array(list(zip(*self.axis)))

    def _at_each_point(self):
        yield from self.stubs.trigger(group="trigger", pointID=self.pointID)
        yield from self.stubs.read(group="primary", wait_group="primary", pointID=self.pointID)
        self.pointID += 1

    def scan_core(self):
        yield from self._move_and_wait(self.positions[0] - self.offset)
        # send the slow motor on its way
        yield from self.stubs.set(
            device=self.scan_motors[0],
            value=self.positions[-1][0],
            wait_group="scan_motor",
        )

        while self.pointID < len(self.positions[:]):
            cont_motor_positions = self.device_manager.devices[self.scan_motors[0]].readback()

            if not cont_motor_positions:
                continue

            cont_motor_positions = cont_motor_positions.get("value")
            logger.debug(f"Current position of {self.scan_motors[0]}: {cont_motor_positions}")
            if np.isclose(cont_motor_positions, self.positions[self.pointID][0], atol=0.5):
                logger.debug(f"reading point {self.pointID}")
                yield from self._at_each_point()
                continue
            if cont_motor_positions > self.positions[self.pointID][0]:
                raise ScanAbortion(f"Skipped point {self.pointID + 1}")


class RoundScanFlySim(ScanBase):
    scan_name = "round_scan_fly"
    scan_report_hint = "table"
    scan_type = "fly"
    pre_move = False
    required_kwargs = ["relative"]
    arg_input = [
        ScanArgType.DEVICE,
        ScanArgType.FLOAT,
        ScanArgType.FLOAT,
        ScanArgType.INT,
        ScanArgType.INT,
    ]
    arg_bundle_size = len(arg_input)

    def __init__(self, *args, **kwargs):
        """
        A fly scan following a round shell-like pattern.

        Args:
            *args: motor1, motor2, inner ring, outer ring, number of rings, number of positions in the first ring
            relative: Start from an absolute or relative position
            burst: number of acquisition per point

        Returns:
            ScanReport

        Examples:
            >>> scans.round_scan_fly(dev.flyer_sim, 0, 50, 5, 3, exp_time=0.1, relative=True)

        """
        super().__init__(**kwargs)
        self.axis = []

    def _get_scan_motors(self):
        self.scan_motors = list(self.caller_args.keys())
        self.flyer = list(self.caller_args.keys())[0]

    def prepare_positions(self):
        self._calculate_positions()
        self.num_pos = len(self.positions) * self.burst_at_each_point
        self._check_limits()
        yield None

    def finalize(self):
        yield

    def _calculate_positions(self):
        params = list(self.caller_args.values())[0]
        self.positions = get_round_scan_positions(
            r_in=params[0], r_out=params[1], nr=params[2], nth=params[3]
        )

    def scan_core(self):
        yield from self.stubs.kickoff(
            device=self.flyer,
            parameter={
                "num_pos": self.num_pos,
                "positions": self.positions.tolist(),
                "exp_time": self.exp_time,
            },
        )
        target_DIID = self.DIID - 1

        while True:
            yield from self.stubs.read_and_wait(group="primary", wait_group="readout_primary")
            msg = self.device_manager.producer.get(MessageEndpoints.device_status(self.flyer))
            if msg:
                status = BECMessage.DeviceStatusMessage.loads(msg)
                device_is_idle = status.content.get("status", 1) == 0
                matching_RID = self.metadata.get("RID") == status.metadata.get("RID")
                matching_DIID = target_DIID == status.metadata.get("DIID")
                if device_is_idle and matching_RID and matching_DIID:
                    break

            time.sleep(1)
            logger.debug("reading monitors")


class RoundROIScan(ScanBase):
    scan_name = "round_roi_scan"
    scan_report_hint = "table"
    required_kwargs = ["dr", "nth", "relative"]
    arg_input = [ScanArgType.DEVICE, ScanArgType.FLOAT]
    arg_bundle_size = len(arg_input)

    def __init__(
        self,
        *args,
        dr=1,
        nth=5,
        exp_time=0,
        relative=False,
        burst_at_each_point=1,
        **kwargs,
    ):
        """
        A scan following a round-roi-like pattern.

        Args:
            *args: motor1, width for motor1, motor2, width for motor2,
            dr (float): shell width. Default is 1.
            nth (int): number of points in the first shell. Default is 5.
            exp_time (float): exposure time in seconds. Default is 0.
            relative (bool): Start from an absolute or relative position. Default is False.
            burst_at_each_point (int): number of acquisition per point. Default is 1.

        Returns:
            ScanReport

        Examples:
            >>> scans.round_roi_scan(dev.motor1, 20, dev.motor2, 20, dr=2, nth=3, exp_time=0.1, relative=True)

        """
        super().__init__(**kwargs)
        self.axis = []
        self.dr = dr
        self.nth = nth
        self.exp_time = exp_time
        self.relative = relative
        self.burst_at_each_point = burst_at_each_point

    def _calculate_positions(self) -> None:
        params = list(self.caller_args.values())
        self.positions = get_round_roi_scan_positions(
            lx=params[0][0], ly=params[1][0], dr=self.dr, nth=self.nth
        )


class ListScan(ScanBase):
    scan_name = "list_scan"
    scan_report_hint = "table"
    required_kwargs = ["relative"]
    arg_input = [ScanArgType.DEVICE, ScanArgType.LIST]
    arg_bundle_size = len(arg_input)

    def __init__(self, *args, parameter=None, **kwargs):
        """
        A scan following the positions specified in a list.
        Please note that all lists must be of equal length.

        Args:
            *args: pairs of motors and position lists
            relative: Start from an absolute or relative position
            burst: number of acquisition per point

        Returns:
            ScanReport

        Examples:
            >>> scans.list_scan(dev.motor1, [0,1,2,3,4], dev.motor2, [4,3,2,1,0], exp_time=0.1, relative=True)

        """
        super().__init__(parameter=parameter, **kwargs)
        self.axis = []
        if len(set(len(entry[0]) for entry in self.caller_args.values())) != 1:
            raise ValueError("All position lists must be of equal length.")

    def _calculate_positions(self):
        self.positions = np.vstack(tuple(self.caller_args.values())).T.tolist()


class TimeScan(ScanBase):
    scan_name = "time_scan"
    scan_report_hint = "table"
    required_kwargs = ["points", "interval"]
    arg_input = []
    arg_bundle_size = len(arg_input)

    def __init__(self, points, interval, *args, **kwargs):
        """
        Trigger and readout devices at a fixed interval.
        Note that the interval time cannot be less than the exposure time.
        The effective "sleep" time between points is
            sleep_time = interval - exp_time

        Args:
            points: number of points
            interval: time interval between points
            exp_time: exposure time in s
            burst: number of acquisition per point

        Returns:
            ScanReport

        Examples:
            >>> scans.time_scan(points=10, interval=1.5, exp_time=0.1, relative=True)

        """
        super().__init__(**kwargs)
        self.axis = []
        self.points = points
        self.interval = interval
        self.interval -= self.exp_time

    def _calculate_positions(self) -> None:
        pass

    def prepare_positions(self):
        self.num_pos = self.points
        yield None

    def _at_each_point(self, ind=None, pos=None):
        if ind > 0:
            yield from self.stubs.wait(
                wait_type="read", group="primary", wait_group="readout_primary"
            )
        yield from self.stubs.trigger(group="trigger", pointID=self.pointID)
        yield from self.stubs.wait(wait_type="trigger", group="trigger", wait_time=self.exp_time)
        yield from self.stubs.read(
            group="primary", wait_group="readout_primary", pointID=self.pointID
        )
        yield from self.stubs.wait(wait_type="trigger", group="trigger", wait_time=self.interval)
        self.pointID += 1

    def scan_core(self):
        for ind in range(self.num_pos):
            yield from self._at_each_point(ind)


class MonitorScan(ScanBase):
    scan_name = "monitor_scan"
    scan_report_hint = "table"
    required_kwargs = ["relative"]
    arg_input = [ScanArgType.DEVICE, ScanArgType.FLOAT, ScanArgType.FLOAT]
    arg_bundle_size = len(arg_input)
    scan_type = "fly"

    def __init__(self, device, start: float, stop: float, *args, relative=False, **kwargs):
        """
        Readout all primary devices at each update of the monitored device.

        Args:
            device (Device): monitored device
            start (float): start position of the monitored device
            stop (float): stop position of the monitored device

        Returns:
            ScanReport

        Examples:
            >>> scans.monitor_scan(dev.motor1, -5, 5, exp_time=0.1, relative=True)

        """
        self.device = device
        self.start = start
        self.stop = stop
        super().__init__(**kwargs)
        self.axis = []
        self.relative = relative

    def _get_scan_motors(self):
        self.scan_motors = [self.device]
        self.flyer = self.device

    def _calculate_positions(self) -> None:
        self.positions = np.vstack(tuple(self.caller_args.values())).T.tolist()

    def prepare_positions(self):
        self._calculate_positions()
        self.num_pos = 0
        yield from self._set_position_offset()
        self._check_limits()

    def _get_flyer_status(self) -> List:
        producer = self.device_manager.producer

        pipe = producer.pipeline()
        producer.lrange(MessageEndpoints.device_req_status(self.metadata["RID"]), 0, -1, pipe)
        producer.get(MessageEndpoints.device_readback(self.flyer), pipe)
        return pipe.execute()

    def scan_core(self):
        yield from self.stubs.set(
            device=self.flyer, value=self.positions[0][0], wait_group="scan_motor"
        )
        yield from self.stubs.wait(wait_type="move", device=self.flyer, wait_group="scan_motor")

        # send the slow motor on its way
        yield from self.stubs.set(
            device=self.flyer,
            value=self.positions[1][0],
            wait_group="scan_motor",
            metadata={"response": True},
        )

        while True:
            move_completed, readback = self._get_flyer_status()

            if move_completed:
                break

            if not readback:
                continue
            readback = BECMessage.DeviceMessage.loads(readback).content["signals"]
            yield from self.stubs.publish_data_as_read(
                device=self.flyer, data=readback, pointID=self.pointID
            )
            self.pointID += 1
            self.num_pos += 1


class Acquire(ScanBase):
    scan_name = "acquire"
    scan_report_hint = "table"
    required_kwargs = []
    arg_input = []
    arg_bundle_size = len(arg_input)

    def __init__(self, *args, exp_time=0, burst_at_each_point=1, **kwargs):
        """
        A simple acquisition at the current position.

        Args:
            burst: number of acquisition per point

        Returns:
            ScanReport

        Examples:
            >>> scans.acquire(exp_time=0.1, relative=True)

        """
        super().__init__(**kwargs)
        self.exp_time = exp_time
        self.burst_at_each_point = burst_at_each_point
        self.axis = []

    def _calculate_positions(self) -> None:
        self.num_pos = self.burst_at_each_point

    def prepare_positions(self):
        self._calculate_positions()

    def _at_each_point(self, ind=None, pos=None):
        if ind > 0:
            yield from self.stubs.wait(
                wait_type="read", group="primary", wait_group="readout_primary"
            )
        yield from self.stubs.trigger(group="trigger", pointID=self.pointID)
        yield from self.stubs.wait(wait_type="trigger", group="trigger", wait_time=self.exp_time)
        yield from self.stubs.read(
            group="primary", wait_group="readout_primary", pointID=self.pointID
        )
        self.pointID += 1

    def scan_core(self):
        for self.burst_index in range(self.burst_at_each_point):
            yield from self._at_each_point(self.burst_index)
        self.burst_index = 0

    def run(self):
        self.initialize()
        self.prepare_positions()
        yield from self.open_scan()
        yield from self.stage()
        yield from self.run_baseline_reading()
        yield from self.pre_scan()
        yield from self.scan_core()
        yield from self.finalize()
        yield from self.unstage()
        yield from self.cleanup()


class LineScan(ScanBase):
    scan_name = "line_scan"
    scan_report_hint = "table"
    required_kwargs = ["steps", "relative"]
    arg_input = [ScanArgType.DEVICE, ScanArgType.FLOAT, ScanArgType.FLOAT]
    arg_bundle_size = len(arg_input)

    def __init__(
        self,
        *args,
        exp_time: float = 0,
        steps: int = None,
        relative: bool = False,
        burst_at_each_point: int = 1,
        **kwargs,
    ):
        """
        A line scan for one or more motors.

        Args:
            *args (Device, float, float): pairs of device / start position / end position
            exp_time (float): exposure time in s. Default: 0
            steps (int): number of steps. Default: 10
            relative (bool): if True, the start and end positions are relative to the current position. Default: False
            burst_at_each_point (int): number of acquisition per point. Default: 1

        Returns:
            ScanReport

        Examples:
            >>> scans.line_scan(dev.motor1, -5, 5, dev.motor2, -5, 5, steps=10, exp_time=0.1, relative=True)

        """
        super().__init__(**kwargs)
        self.exp_time = exp_time
        self.steps = steps
        self.relative = relative
        self.burst_at_each_point = burst_at_each_point
        self.axis = []

    def _calculate_positions(self) -> None:
        for _, val in self.caller_args.items():
            ax_pos = np.linspace(val[0], val[1], self.steps)
            self.axis.append(ax_pos)
        self.positions = np.array(list(zip(*self.axis)))


class OpenInteractiveScan(ScanBase):
    scan_name = "open_interactive_scan"
    scan_report_hint = ""
    required_kwargs = []
    arg_input = [ScanArgType.DEVICE]
    arg_bundle_size = len(arg_input)

    def __init__(self, *args, **kwargs):
        """
        An interactive scan for one or more motors.

        Args:
            *args: devices
            exp_time: exposure time in s
            steps: number of steps (please note: 5 steps == 6 positions)
            relative: Start from an absolute or relative position
            burst: number of acquisition per point

        Returns:
            ScanReport

        Examples:
            >>> scans.open_interactive_scan(dev.motor1, dev.motor2, exp_time=0.1)

        """
        super().__init__(**kwargs)
        self.axis = []

    def _calculate_positions(self):
        pass

    def _get_scan_motors(self):
        caller_args = list(self.caller_args.keys())
        self.scan_motors = caller_args

    def run(self):
        yield from self.stubs.open_scan_def()
        self.initialize()
        yield from self.read_scan_motors()
        yield from self.open_scan()
        yield from self.stage()
        yield from self.run_baseline_reading()


class AddInteractiveScanPoint(ScanBase):
    scan_name = "interactive_scan_trigger"
    scan_report_hint = ""
    arg_input = [ScanArgType.DEVICE]
    arg_bundle_size = len(arg_input)

    def __init__(self, *args, **kwargs):
        """
        An interactive scan for one or more motors.

        Args:
            *args: devices
            exp_time: exposure time in s
            steps: number of steps (please note: 5 steps == 6 positions)
            relative: Start from an absolute or relative position
            burst: number of acquisition per point

        Returns:
            ScanReport

        Examples:
            >>> scans.interactive_scan_trigger()

        """
        super().__init__(**kwargs)
        self.axis = []

    def _calculate_positions(self):
        pass

    def _get_scan_motors(self):
        self.scan_motors = list(self.caller_args.keys())

    def _at_each_point(self, ind=None, pos=None):
        yield from self.stubs.trigger(group="trigger", pointID=self.pointID)
        yield from self.stubs.wait(wait_type="trigger", group="trigger", wait_time=self.exp_time)
        yield from self.stubs.read_and_wait(
            group="primary", wait_group="readout_primary", pointID=self.pointID
        )
        self.pointID += 1

    def run(self):
        yield from self.open_scan()
        yield from self._at_each_point()
        yield from self.close_scan()


class CloseInteractiveScan(ScanBase):
    scan_name = "close_interactive_scan"
    scan_report_hint = ""
    arg_input = []
    arg_bundle_size = len(arg_input)

    def __init__(self, *args, **kwargs):
        """
        An interactive scan for one or more motors.

        Args:
            *args: devices
            exp_time: exposure time in s
            steps: number of steps (please note: 5 steps == 6 positions)
            relative: Start from an absolute or relative position
            burst: number of acquisition per point

        Returns:
            ScanReport

        Examples:
            >>> scans.close_interactive_scan(dev.motor1, dev.motor2, exp_time=0.1)

        """
        super().__init__(**kwargs)
        self.axis = []

    def _calculate_positions(self):
        pass

    def run(self):
        yield from self.finalize()
        yield from self.unstage()
        yield from self.cleanup()
        yield from self.stubs.close_scan_def()
