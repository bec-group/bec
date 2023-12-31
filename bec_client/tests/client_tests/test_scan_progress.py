import asyncio
from unittest import mock

import pytest
from bec_lib.core import BECMessage, MessageEndpoints

from bec_client.callbacks.scan_progress import LiveUpdatesScanProgress


@pytest.mark.asyncio
async def test_update_progressbar_continues_without_device_data():
    bec = mock.MagicMock()
    request = mock.MagicMock()
    live_update = LiveUpdatesScanProgress(bec=bec, report_instruction={}, request=request)
    progressbar = mock.MagicMock()

    bec.producer.get.return_value = None
    res = await live_update._update_progressbar(progressbar, "async_dev1")
    assert res is False


@pytest.mark.asyncio
async def test_update_progressbar_continues_when_scanID_doesnt_match():
    bec = mock.MagicMock()
    request = mock.MagicMock()
    live_update = LiveUpdatesScanProgress(bec=bec, report_instruction={}, request=request)
    progressbar = mock.MagicMock()
    live_update.scan_item = mock.MagicMock()
    live_update.scan_item.scanID = "scanID2"

    bec.producer.get.return_value = BECMessage.DeviceStatusMessage(
        device="async_dev1", status={"value": 1}, metadata={"scanID": "scanID"}
    ).dumps()
    res = await live_update._update_progressbar(progressbar, "async_dev1")
    assert res is False


@pytest.mark.asyncio
async def test_update_progressbar_continues_when_msg_specifies_no_value():
    bec = mock.MagicMock()
    request = mock.MagicMock()
    live_update = LiveUpdatesScanProgress(bec=bec, report_instruction={}, request=request)
    progressbar = mock.MagicMock()
    live_update.scan_item = mock.MagicMock()
    live_update.scan_item.scanID = "scanID"

    bec.producer.get.return_value = BECMessage.DeviceStatusMessage(
        device="async_dev1", status={}, metadata={"scanID": "scanID"}
    ).dumps()
    res = await live_update._update_progressbar(progressbar, "async_dev1")
    assert res is False


@pytest.mark.asyncio
async def test_update_progressbar_updates_max_value():
    bec = mock.MagicMock()
    request = mock.MagicMock()
    live_update = LiveUpdatesScanProgress(bec=bec, report_instruction={}, request=request)
    progressbar = mock.MagicMock()
    live_update.scan_item = mock.MagicMock()
    live_update.scan_item.scanID = "scanID"

    bec.producer.get.return_value = BECMessage.DeviceStatusMessage(
        device="async_dev1", status={"value": 10, "max_value": 20}, metadata={"scanID": "scanID"}
    ).dumps()
    res = await live_update._update_progressbar(progressbar, "async_dev1")
    assert res is False
    assert progressbar.max_points == 20
    progressbar.update.assert_called_once_with(10)


@pytest.mark.asyncio
async def test_update_progressbar_returns_true_when_max_value_is_reached():
    bec = mock.MagicMock()
    request = mock.MagicMock()
    live_update = LiveUpdatesScanProgress(bec=bec, report_instruction={}, request=request)
    progressbar = mock.MagicMock()
    live_update.scan_item = mock.MagicMock()
    live_update.scan_item.scanID = "scanID"

    bec.producer.get.return_value = BECMessage.DeviceStatusMessage(
        device="async_dev1", status={"value": 20, "max_value": 20}, metadata={"scanID": "scanID"}
    ).dumps()
    res = await live_update._update_progressbar(progressbar, "async_dev1")
    assert res is True
