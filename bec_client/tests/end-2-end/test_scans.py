import _thread
import threading
import time

import numpy as np
import pytest
from bec_lib import BECClient
from bec_lib.alarm_handler import AlarmBase
from bec_lib.core import (
    BECMessage,
    MessageEndpoints,
    RedisConnector,
    ServiceConfig,
    bec_logger,
)
from bec_lib.core.bec_errors import ScanAbortion, ScanInterruption
from bec_lib.core.tests.utils import wait_for_empty_queue

from bec_client import BECIPythonClient
from bec_client.callbacks.utils import ScanRequestError

logger = bec_logger.logger

CONFIG_PATH = "../ci/test_config.yaml"
# CONFIG_PATH = "../bec_config_dev.yaml"
# pylint: disable=no-member
# pylint: disable=missing-function-docstring
# pylint: disable=redefined-outer-name
# pylint: disable=protected-access
# pylint: disable=undefined-variable


@pytest.fixture(scope="function")
def client():
    config = ServiceConfig(CONFIG_PATH)
    bec = BECIPythonClient(forced=True)
    bec.initialize(
        config,
        RedisConnector,
    )
    bec.start()
    bec.queue.request_queue_reset()
    bec.queue.request_scan_continuation()
    time.sleep(5)
    yield bec
    bec.shutdown()


@pytest.mark.timeout(100)
def test_grid_scan(capsys, client):
    bec = client
    scans = bec.scans
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_grid_scan"})
    dev = bec.device_manager.devices
    scans.umv(dev.samx, 0, dev.samy, 0, relative=False)
    status = scans.grid_scan(dev.samx, -5, 5, 10, dev.samy, -5, 5, 10, exp_time=0.01, relative=True)
    assert len(status.scan.data) == 100
    assert status.scan.num_points == 100
    captured = capsys.readouterr()
    assert "finished. Scan ID" in captured.out


@pytest.mark.timeout(100)
def test_fermat_scan(capsys, client):
    bec = client
    scans = bec.scans
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_fermat_scan"})
    dev = bec.device_manager.devices
    status = scans.fermat_scan(
        dev.samx,
        -5,
        5,
        dev.samy,
        -5,
        5,
        step=0.5,
        exp_time=0.01,
        relative=True,
        optim_trajectory="corridor",
    )
    assert len(status.scan.data) == 393
    assert status.scan.num_points == 393
    captured = capsys.readouterr()
    assert "finished. Scan ID" in captured.out


@pytest.mark.timeout(100)
def test_line_scan(capsys, client):
    bec = client
    scans = bec.scans
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_line_scan"})
    dev = bec.device_manager.devices
    status = scans.line_scan(dev.samx, -5, 5, steps=10, exp_time=0.01, relative=True)
    assert len(status.scan.data) == 10
    assert status.scan.num_points == 10
    captured = capsys.readouterr()
    assert "finished. Scan ID" in captured.out


@pytest.mark.timeout(100)
def test_mv_scan(capsys, client):
    bec = client
    scans = bec.scans
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_mv_scan"})
    dev = bec.device_manager.devices
    scans.mv(dev.samx, 10, dev.samy, 20, relative=False).wait()
    current_pos_samx = dev.samx.read()["samx"]["value"]
    current_pos_samy = dev.samy.read()["samy"]["value"]
    assert np.isclose(
        current_pos_samx, 10, atol=dev.samx._config["deviceConfig"].get("tolerance", 0.05)
    )
    assert np.isclose(
        current_pos_samy, 20, atol=dev.samy._config["deviceConfig"].get("tolerance", 0.05)
    )
    scans.umv(dev.samx, 10, dev.samy, 20, relative=False)
    current_pos_samx = dev.samx.read()["samx"]["value"]
    current_pos_samy = dev.samy.read()["samy"]["value"]
    captured = capsys.readouterr()
    ref_out_samx = f" ━━━━━━━━━━━━━━━ {current_pos_samx:10.2f} /      10.00 / 100 % 0:00:00 0:00:00"
    ref_out_samy = f" ━━━━━━━━━━━━━━━ {current_pos_samy:10.2f} /      20.00 / 100 % 0:00:00 0:00:00"
    assert ref_out_samx in captured.out
    assert ref_out_samy in captured.out


@pytest.mark.timeout(100)
def test_mv_scan_mv(client):
    bec = client
    scans = bec.scans
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_mv_scan_mv"})
    scan_number_start = bec.queue.next_scan_number
    dev = bec.device_manager.devices

    dev.samx.limits = [-50, 50]
    dev.samy.limits = [-50, 50]
    scans.umv(dev.samx, 10, dev.samy, 20, relative=False)
    tolerance_samx = dev.samx._config["deviceConfig"].get("tolerance", 0.05)
    tolerance_samy = dev.samy._config["deviceConfig"].get("tolerance", 0.05)
    current_pos_samx = dev.samx.read()["samx"]["value"]
    current_pos_samy = dev.samy.read()["samy"]["value"]

    # make sure the current position after mv is within the tolerance
    assert np.isclose(current_pos_samx, 10, atol=tolerance_samx)
    assert np.isclose(current_pos_samy, 20, atol=tolerance_samy)

    status = scans.grid_scan(dev.samx, -5, 5, 10, dev.samy, -5, 5, 10, exp_time=0.01, relative=True)

    # make sure the scan completed the expected number of positions
    assert len(status.scan.data) == 100
    assert status.scan.num_points == 100

    # make sure the scan is relative to the starting position
    assert np.isclose(
        current_pos_samx - 5,
        status.scan.data[0].content["data"]["samx"]["samx"]["value"],
        atol=tolerance_samx,
    )

    current_pos_samx = dev.samx.read()["samx"]["value"]
    current_pos_samy = dev.samy.read()["samy"]["value"]

    # make sure the new position is within 2x the tolerance (two movements)
    assert np.isclose(current_pos_samx, 10, atol=tolerance_samx * 2)
    assert np.isclose(current_pos_samy, 20, atol=tolerance_samy * 2)

    scans.umv(dev.samx, 20, dev.samy, -20, relative=False)
    current_pos_samx = dev.samx.read()["samx"]["value"]
    current_pos_samy = dev.samy.read()["samy"]["value"]

    # make sure the umv movement is within the tolerance
    assert np.isclose(current_pos_samx, 20, atol=tolerance_samx)
    assert np.isclose(current_pos_samy, -20, atol=tolerance_samy)

    status = scans.grid_scan(
        dev.samx, -5, 5, 10, dev.samy, -5, 5, 10, exp_time=0.01, relative=False
    )

    # make sure the scan completed the expected number of points
    assert len(status.scan.data) == 100
    assert status.scan.num_points == 100

    # make sure the scan was absolute, not relative
    assert np.isclose(
        -5, status.scan.data[0].content["data"]["samx"]["samx"]["value"], atol=tolerance_samx
    )
    scan_number_end = bec.queue.next_scan_number
    assert scan_number_end == scan_number_start + 2


@pytest.mark.timeout(100)
def test_scan_abort(client):
    def send_abort(bec):
        while True:
            current_scan_info = bec.queue.scan_storage.current_scan_info
            if not current_scan_info:
                continue
            status = current_scan_info.get("status").lower()
            if status not in ["running", "deferred_pause"]:
                continue
            if bec.queue.scan_storage.current_scan is None:
                continue
            if len(bec.queue.scan_storage.current_scan.data) > 10:
                _thread.interrupt_main()
                break
        while True:
            queue = bec.queue.queue_storage.current_scan_queue
            if queue["primary"]["info"][0]["status"] == "DEFERRED_PAUSE":
                break
            time.sleep(0.5)
        _thread.interrupt_main()

    bec = client
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_scan_abort"})
    scan_number_start = bec.queue.next_scan_number
    scans = bec.scans
    dev = bec.device_manager.devices
    aborted_scan = False
    try:
        threading.Thread(target=send_abort, args=(bec,), daemon=True).start()
        scans.line_scan(dev.samx, -5, 5, steps=200, exp_time=0.1, relative=True)
    except ScanInterruption:
        logger.info("Raised ScanInterruption")
        time.sleep(2)
        bec.queue.request_scan_abortion()
        aborted_scan = True
    assert aborted_scan is True
    while bec.queue.scan_storage.storage[0].status == "open":
        time.sleep(0.5)

    current_queue = bec.queue.queue_storage.current_scan_queue["primary"]
    while current_queue["info"] or current_queue["status"] != "RUNNING":
        time.sleep(0.5)
        current_queue = bec.queue.queue_storage.current_scan_queue["primary"]

    assert len(bec.queue.scan_storage.storage[-1].data) < 200

    scans.line_scan(dev.samx, -5, 5, steps=10, exp_time=0.1, relative=True)
    scan_number_end = bec.queue.next_scan_number
    assert scan_number_end == scan_number_start + 2


@pytest.mark.timeout(100)
def test_limit_error(client):
    bec = client
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_limit_error"})
    scan_number_start = bec.queue.next_scan_number
    scans = bec.scans
    dev = bec.device_manager.devices
    aborted_scan = False
    dev.samx.limits = [-50, 50]
    try:
        scans.line_scan(dev.samx, -520, 5, steps=200, exp_time=0.1, relative=False)
    except AlarmBase as alarm:
        assert alarm.alarm_type == "LimitError"
        aborted_scan = True

    assert aborted_scan is True

    aborted_scan = False
    dev.samx.limits = [-50, 50]
    time.sleep(10)
    try:
        scans.umv(dev.samx, 500, relative=False)
    except AlarmBase as alarm:
        assert alarm.alarm_type == "LimitError"
        aborted_scan = True

    assert aborted_scan is True
    scan_number_end = bec.queue.next_scan_number
    assert scan_number_end == scan_number_start + 1


@pytest.mark.timeout(100)
def test_queued_scan(client):
    bec = client
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_queued_scan"})
    scan_number_start = bec.queue.next_scan_number
    scans = bec.scans
    dev = bec.device_manager.devices
    scan1 = scans.line_scan(
        dev.samx, -5, 5, steps=100, exp_time=0.1, hide_report=True, relative=True
    )
    scan2 = scans.line_scan(
        dev.samx, -5, 5, steps=50, exp_time=0.1, hide_report=True, relative=True
    )

    while True:
        if not scan1.scan or not scan2.scan:
            continue
        if scan1.scan.status != "open":
            continue
        assert scan1.scan.queue.queue_position == 0
        assert scan2.scan.queue.queue_position == 1
        break
    while len(scan2.scan.data) != 50:
        time.sleep(0.5)
    current_queue = bec.queue.queue_storage.current_scan_queue["primary"]
    while current_queue["info"] or current_queue["status"] != "RUNNING":
        time.sleep(0.5)
        current_queue = bec.queue.queue_storage.current_scan_queue["primary"]
    scan_number_end = bec.queue.next_scan_number
    assert scan_number_end == scan_number_start + 2


@pytest.mark.timeout(100)
def test_fly_scan(client):
    bec = client
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_fly_scan"})
    scans = bec.scans
    dev = bec.device_manager.devices
    status = scans.round_scan_fly(dev.flyer_sim, 0, 50, 20, 3, exp_time=0.1, relative=True)
    assert len(status.scan.data) == 693
    assert status.scan.num_points == 693


@pytest.mark.timeout(100)
def test_scan_restart(client):
    bec = client
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_scan_restart"})
    scans = bec.scans
    dev = bec.device_manager.devices

    def send_repeat(bec):
        while True:
            if not bec.queue.scan_storage.current_scan:
                continue
            if len(bec.queue.scan_storage.current_scan.data) > 0:
                time.sleep(2)
                bec.queue.request_scan_restart()
                bec.queue.request_scan_continuation()
                break

    scan_number_start = bec.queue.next_scan_number
    # start repeat thread
    threading.Thread(target=send_repeat, args=(bec,), daemon=True).start()
    # start scan
    scan1 = scans.line_scan(
        dev.samx, -5, 5, steps=50, exp_time=0.1, hide_report=True, relative=True
    )
    scan2 = scans.line_scan(
        dev.samx, -5, 5, steps=50, exp_time=0.1, hide_report=True, relative=True
    )

    scan2.wait()

    current_queue = bec.queue.queue_storage.current_scan_queue["primary"]
    while current_queue["info"] or current_queue["status"] != "RUNNING":
        time.sleep(0.5)
        current_queue = bec.queue.queue_storage.current_scan_queue["primary"]
    scan_number_end = bec.queue.next_scan_number
    assert scan_number_end == scan_number_start + 3


@pytest.mark.timeout(100)
def test_scan_observer_repeat_queued(client):
    bec = client
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_scan_observer_repeat_queued"})
    scans = bec.scans
    dev = bec.device_manager.devices

    def send_repeat(bec):
        while True:
            if not bec.queue.scan_storage.current_scan:
                continue
            if len(bec.queue.scan_storage.current_scan.data) > 0:
                time.sleep(2)
                bec.queue.request_scan_interruption(deferred_pause=False)
                time.sleep(5)
                bec.queue.request_scan_restart()
                bec.queue.request_scan_continuation()
                break

    scan_number_start = bec.queue.next_scan_number
    # start repeat thread
    threading.Thread(target=send_repeat, args=(bec,), daemon=True).start()
    # start scan
    scan1 = scans.line_scan(
        dev.samx, -5, 5, steps=100, exp_time=0.1, hide_report=True, relative=True
    )
    scan2 = scans.line_scan(
        dev.samx, -5, 5, steps=100, exp_time=0.1, hide_report=True, relative=True
    )

    scan2.wait()

    current_queue = bec.queue.queue_storage.current_scan_queue["primary"]
    while current_queue["info"] or current_queue["status"] != "RUNNING":
        time.sleep(0.5)
        current_queue = bec.queue.queue_storage.current_scan_queue["primary"]
    scan_number_end = bec.queue.next_scan_number
    assert scan_number_end == scan_number_start + 3


@pytest.mark.timeout(100)
def test_scan_observer_repeat(client):
    bec = client
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_scan_observer_repeat"})
    scans = bec.scans
    dev = bec.device_manager.devices

    def send_repeat(bec):
        while True:
            if not bec.queue.scan_storage.current_scan:
                continue
            if len(bec.queue.scan_storage.current_scan.data) > 0:
                time.sleep(2)
                bec.queue.request_scan_interruption(deferred_pause=False)
                time.sleep(5)
                bec.queue.request_scan_restart()
                bec.queue.request_scan_continuation()
                break

    scan_number_start = bec.queue.next_scan_number
    # start repeat thread
    threading.Thread(target=send_repeat, args=(bec,), daemon=True).start()
    # start scan
    with pytest.raises(ScanAbortion):
        scan1 = scans.line_scan(
            dev.samx, -5, 5, steps=50, exp_time=0.1, hide_report=True, relative=True
        )
        scan1.wait()

    current_queue = bec.queue.queue_storage.current_scan_queue["primary"]
    while current_queue["info"] or current_queue["status"] != "RUNNING":
        time.sleep(0.5)
        current_queue = bec.queue.queue_storage.current_scan_queue["primary"]
    while True:
        if bec.queue.next_scan_number == scan_number_start + 2:
            break


@pytest.mark.timeout(100)
def test_file_writer(client):
    bec = client
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_file_writer"})
    scans = bec.scans
    dev = bec.device_manager.devices

    scan = scans.grid_scan(
        dev.samx,
        -5,
        5,
        10,
        dev.samy,
        -5,
        5,
        10,
        exp_time=0.01,
        relative=True,
        md={"datasetID": 325},
    )
    assert len(scan.scan.data) == 100
    msg = bec.device_manager.producer.get(MessageEndpoints.public_file(scan.scan.scanID, "master"))
    while True:
        if msg:
            break
        msg = bec.device_manager.producer.get(
            MessageEndpoints.public_file(scan.scan.scanID, "master")
        )

    file_msg = BECMessage.FileMessage.loads(msg)
    assert file_msg.content["successful"]

    # currently not working due to access restrictions in docker:

    # with h5py.File(file_msg.content["file_path"], "r") as file:
    #     assert file["entry"]["collection"]["bec"]["metadata"]["datasetID"][()] == 325

    #     file_data = file["entry"]["collection"]["bec"]["samx"][()]
    #     stream_data = [
    #         msg.content["data"]["samx"]["samx"]["value"] for msg in scan.scan.data.values()
    #     ]
    #     assert all(file_data == stream_data)


@pytest.mark.timeout(100)
def test_scan_def_callback(capsys, client):
    bec = client
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_scan_def_callback"})
    scans = bec.scans
    dev = bec.device_manager.devices
    scan_number = bec.queue.next_scan_number
    with scans.scan_def:
        scans.line_scan(dev.samx, -5, 5, steps=10, exp_time=0.1, relative=False)
        scans.umv(dev.samy, 5, relative=False)
        current_pos_samy = dev.samy.read(cached=True)["value"]
        captured = capsys.readouterr()
        assert f"Starting scan {scan_number}" in captured.out
        ref_out_samy = (
            f" ━━━━━━━━━━━━━━━ {current_pos_samy:10.2f} /       5.00 / 100 % 0:00:00 0:00:00"
        )
        assert ref_out_samy in captured.out
        scans.line_scan(dev.samx, -5, 5, steps=10, exp_time=0.1, relative=False)
    captured = capsys.readouterr()
    assert f"Scan {scan_number} finished." in captured.out


@pytest.mark.timeout(100)
def test_scan_def(client):
    bec = client
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_scan_def"})
    scans = bec.scans
    dev = bec.device_manager.devices
    scan_number = bec.queue.next_scan_number
    with scans.scan_def:
        scans.line_scan(dev.samx, -5, 5, steps=10, exp_time=0.1, relative=False)
        scans.umv(dev.samy, 5, relative=False)
        scans.line_scan(dev.samx, -5, 5, steps=10, exp_time=0.1, relative=False)

    assert scan_number == bec.queue.next_scan_number - 1

    scan_number = bec.queue.next_scan_number

    @scans.scan_def
    def scan_def_with_decorator():
        scans.line_scan(dev.samx, -5, 5, steps=10, exp_time=0.1, relative=False)
        scans.umv(dev.samy, 5, relative=False)
        scans.line_scan(dev.samx, -5, 5, steps=10, exp_time=0.1, relative=False)

    scan_def_with_decorator()
    assert scan_number == bec.queue.next_scan_number - 1


@pytest.mark.timeout(100)
def test_group_def(client):
    bec = client
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_scan_def"})
    scans = bec.scans
    dev = bec.device_manager.devices
    scan_number = bec.queue.next_scan_number
    with scans.scan_group:
        scans.line_scan(dev.samx, -5, 5, steps=10, exp_time=0.1, relative=False)
        scans.umv(dev.samy, 5, relative=False)
        scans.line_scan(dev.samx, -5, 5, steps=10, exp_time=0.1, relative=False)

    assert scan_number == bec.queue.next_scan_number - 2


@pytest.mark.timeout(100)
def test_list_scan(client):
    bec = client
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_list_scan"})
    scans = bec.scans
    dev = bec.device_manager.devices

    status = scans.list_scan(
        dev.samx, [0, 1, 2, 3, 4], dev.samy, [0, 1, 2, 3, 4], exp_time=0.1, relative=False
    )
    assert len(status.scan.data) == 5

    status = scans.list_scan(dev.samx, [0, 1, 2, 3, 4, 5], exp_time=0.1, relative=False)
    assert len(status.scan.data) == 6

    status = scans.list_scan(
        dev.samx,
        [0, 1, 2, 3],
        dev.samy,
        [0, 1, 2, 3],
        dev.samz,
        [0, 1, 2, 3],
        exp_time=0.1,
        relative=False,
    )
    assert len(status.scan.data) == 4


@pytest.mark.timeout(100)
def test_time_scan(client):
    bec = client
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_time_scan"})
    scans = bec.scans
    status = scans.time_scan(points=5, interval=0.5, exp_time=0.1, relative=False)
    assert len(status.scan.data) == 5


@pytest.mark.timeout(100)
def test_monitor_scan(client):
    bec = client
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_monitor_scan"})
    scans = bec.scans
    dev = bec.device_manager.devices
    dev.samx.limits = [-1100, 1100]
    time.sleep(5)
    status = scans.monitor_scan(dev.samx, -100, 100, relative=False)
    assert len(status.scan.data) > 100


@pytest.mark.timeout(100)
def test_rpc_calls(client):
    bec = client
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_rpc_calls"})
    dev = bec.device_manager.devices
    assert dev.samx.dummy_controller._func_with_args(2, 3) == [2, 3]
    assert dev.samx.dummy_controller._func_with_kwargs(kwinput1=2, kwinput2=3) == {
        "kwinput1": 2,
        "kwinput2": 3,
    }
    assert dev.samx.dummy_controller._func_with_args_and_kwargs(2, 3, kwinput1=2, kwinput2=3) == [
        [2, 3],
        {"kwinput1": 2, "kwinput2": 3},
    ]

    assert dev.samx.dummy_controller._func_without_args_kwargs() is None


@pytest.mark.timeout(100)
def test_burst_scan(client):
    bec = client
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_burst_scan"})
    dev = bec.device_manager.devices
    s = scans.line_scan(dev.samx, 0, 1, burst_at_each_point=2, steps=10, relative=False)
    assert len(s.scan.data) == 20


@pytest.mark.timeout(100)
def test_callback_data_matches_scan_data(client):
    bec = client
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_callback_data_matches_scan_data"})
    dev = bec.device_manager.devices
    reference_container = {"data": [], "metadata": {}}

    def dummy_callback(data, metadata):
        reference_container["metadata"] = metadata
        reference_container["data"].append(data)

    s = scans.line_scan(dev.samx, 0, 1, steps=10, relative=False, callback=dummy_callback)
    while len(reference_container["data"]) < 10:
        time.sleep(0.1)
    assert len(s.scan.data) == 10
    assert len(reference_container["data"]) == 10

    for ii, msg in enumerate(s.scan.data.values()):
        assert msg.content == reference_container["data"][ii]


@pytest.mark.timeout(100)
def test_async_callback_data_matches_scan_data(client):
    bec = client
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_async_callback_data_matches_scan_data"})
    dev = bec.device_manager.devices
    reference_container = {"data": [], "metadata": {}}

    def dummy_callback(data, metadata):
        reference_container["metadata"] = metadata
        reference_container["data"].append(data)

    s = scans.line_scan(dev.samx, 0, 1, steps=10, relative=False, async_callback=dummy_callback)

    while len(reference_container["data"]) < 10:
        time.sleep(0.1)

    assert len(s.scan.data) == 10
    assert len(reference_container["data"]) == 10

    for ii, msg in enumerate(s.scan.data.values()):
        assert msg.content == reference_container["data"][ii]


@pytest.mark.timeout(100)
def test_disabled_device_raises_scan_request_error(client):
    bec = client
    wait_for_empty_queue(bec)
    bec.metadata.update({"unit_test": "test_disabled_device_raises_scan_rejection"})
    dev = bec.device_manager.devices
    dev.samx.enabled = False
    time.sleep(1)
    with pytest.raises((AlarmBase, ScanRequestError)):
        scans.line_scan(dev.samx, 0, 1, steps=10, relative=False)
    dev.samx.enabled = True
    time.sleep(1)
    scans.line_scan(dev.samx, 0, 1, steps=10, relative=False)
