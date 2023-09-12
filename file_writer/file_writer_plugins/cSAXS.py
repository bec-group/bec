from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from bec_lib.core import DeviceManagerBase

    from file_writer.file_writer import HDF5Storage


def get_entry(data: dict, name: str, default=None) -> Any:
    """
    Get an entry from the scan data assuming a <device>.<device>.value structure.

    Args:
        data (dict): Scan data
        name (str): Entry name
        default (Any, optional): Default value. Defaults to None.
    """
    if isinstance(data.get(name), list) and isinstance(data.get(name)[0], dict):
        return [sub_data.get(name, {}).get("value", default) for sub_data in data.get(name)]

    return data.get(name, {}).get(name, {}).get("value", default)


def NeXus_format(
    storage: HDF5Storage,
    data: dict,
    file_references: dict,
    device_manager: DeviceManagerBase,
) -> HDF5Storage:
    """
    Prepare the NeXus file format.

    Args:
        storage (HDF5Storage): HDF5 storage. Pseudo hdf5 file container that will be written to disk later.
        data (dict): scan data
        file_references (dict): File references. Can be used to add external files to the HDF5 file. The path is given relative to the HDF5 file.
        device_manager (DeviceManagerBase): Device manager. Can be used to check if devices are available.

    Returns:
        HDF5Storage: Updated HDF5 storage
    """
    # /entry
    entry = storage.create_group("entry")
    entry.attrs["NX_class"] = "NXentry"
    entry.attrs["definition"] = "NXsas"
    entry.attrs["start_time"] = data.get("start_time")
    entry.attrs["end_time"] = data.get("end_time")
    entry.attrs["version"] = 1.0

    # /entry/collection
    collection = entry.create_group("collection")
    collection.attrs["NX_class"] = "NXcollection"
    bec_collection = collection.create_group("bec")

    # /entry/control
    control = entry.create_group("control")
    control.attrs["NX_class"] = "NXmonitor"
    control.create_dataset(name="mode", data="monitor")
    control.create_dataset(name="integral", data=get_entry(data, "bpm4i"))

    # /entry/data
    main_data = entry.create_group("data")
    main_data.attrs["NX_class"] = "NXdata"
    if "eiger_4" in device_manager.devices:
        main_data.create_soft_link(name="data", target="/entry/instrument/eiger_4/data")
    elif "eiger9m" in device_manager.devices:
        main_data.create_soft_link(name="data", target="/entry/instrument/eiger9m/data")
    elif "pilatus_2" in device_manager.devices:
        main_data.create_soft_link(name="data", target="/entry/instrument/pilatus_2/data")

    # /entry/sample
    control = entry.create_group("sample")
    control.attrs["NX_class"] = "NXsample"
    control.create_dataset(name="name", data=get_entry(data, "samplename"))
    control.create_dataset(name="description", data=data.get("sample_description"))
    x_translation = control.create_dataset(name="x_translation", data=get_entry(data, "samx"))
    x_translation.attrs["units"] = "mm"
    y_translation = control.create_dataset(name="y_translation", data=get_entry(data, "samy"))
    y_translation.attrs["units"] = "mm"
    temperature_log = control.create_dataset(name="temperature_log", data=get_entry(data, "temp"))
    temperature_log.attrs["units"] = "K"

    # /entry/instrument
    instrument = entry.create_group("instrument")
    instrument.attrs["NX_class"] = "NXinstrument"
    instrument.create_dataset(name="name", data="cSAXS beamline")

    source = instrument.create_group("source")
    source.attrs["NX_class"] = "NXsource"
    source.create_dataset(name="type", data="Synchrotron X-ray Source")
    source.create_dataset(name="name", data="Swiss Light Source")
    source.create_dataset(name="probe", data="x-ray")
    distance = source.create_dataset(
        name="distance", data=-33800 - np.asarray(get_entry(data, "samz", 0))
    )
    distance.attrs["units"] = "mm"
    sigma_x = source.create_dataset(name="sigma_x", data=0.202)
    sigma_x.attrs["units"] = "mm"
    sigma_y = source.create_dataset(name="sigma_y", data=0.018)
    sigma_y.attrs["units"] = "mm"
    divergence_x = source.create_dataset(name="divergence_x", data=0.000135)
    divergence_x.attrs["units"] = "radians"
    divergence_y = source.create_dataset(name="divergence_y", data=0.000025)
    divergence_y.attrs["units"] = "radians"
    current = source.create_dataset(name="current", data=get_entry(data, "curr"))
    current.attrs["units"] = "mA"

    insertion_device = instrument.create_group("insertion_device")
    insertion_device.attrs["NX_class"] = "NXinsertion_device"
    source.create_dataset(name="type", data="undulator")
    gap = source.create_dataset(name="gap", data=get_entry(data, "idgap"))
    gap.attrs["units"] = "mm"
    k = source.create_dataset(name="k", data=2.46)
    k.attrs["units"] = "NX_DIMENSIONLESS"
    length = source.create_dataset(name="length", data=1820)
    length.attrs["units"] = "mm"

    slit_0 = instrument.create_group("slit_0")
    slit_0.attrs["NX_class"] = "NXslit"
    source.create_dataset(name="material", data="OFHC Cu")
    source.create_dataset(name="description", data="Horizontal secondary source slit")
    x_gap = source.create_dataset(name="x_gap", data=get_entry(data, "sl0wh"))
    x_gap.attrs["units"] = "mm"
    x_translation = source.create_dataset(name="x_translation", data=get_entry(data, "sl0ch"))
    x_translation.attrs["units"] = "mm"
    distance = source.create_dataset(
        name="distance", data=-21700 - np.asarray(get_entry(data, "samz", 0))
    )
    distance.attrs["units"] = "mm"

    slit_1 = instrument.create_group("slit_1")
    slit_1.attrs["NX_class"] = "NXslit"
    source.create_dataset(name="material", data="OFHC Cu")
    source.create_dataset(name="description", data="Horizontal secondary source slit")
    x_gap = source.create_dataset(name="x_gap", data=get_entry(data, "sl1wh"))
    x_gap.attrs["units"] = "mm"
    y_gap = source.create_dataset(name="y_gap", data=get_entry(data, "sl1wv"))
    y_gap.attrs["units"] = "mm"
    x_translation = source.create_dataset(name="x_translation", data=get_entry(data, "sl1ch"))
    x_translation.attrs["units"] = "mm"
    height = source.create_dataset(name="x_translation", data=get_entry(data, "sl1ch"))
    height.attrs["units"] = "mm"
    distance = source.create_dataset(
        name="distance", data=-7800 - np.asarray(get_entry(data, "samz", 0))
    )
    distance.attrs["units"] = "mm"

    mono = instrument.create_group("monochromator")
    mono.attrs["NX_class"] = "NXmonochromator"
    mokev = data.get("mokev", {})
    if mokev:
        if isinstance(mokev, list):
            mokev = mokev[0]
        wavelength = mono.create_dataset(
            name="wavelength",
            data=12.3984193 / (mokev.get("mokev").get("value") + 1e-9),
        )
        wavelength.attrs["units"] = "Angstrom"
        energy = mono.create_dataset(name="energy", data=mokev.get("mokev").get("value"))
        energy.attrs["units"] = "keV"
    mono.create_dataset(name="type", data="Double crystal fixed exit monochromator.")
    distance = mono.create_dataset(
        name="distance", data=-5220 - np.asarray(get_entry(data, "samz", 0))
    )
    distance.attrs["units"] = "mm"

    crystal_1 = mono.create_group("crystal_1")
    crystal_1.attrs["NX_class"] = "NXcrystal"
    crystal_1.create_dataset(name="usage", data="Bragg")
    crystal_1.create_dataset(name="order_no", data="1")
    crystal_1.create_dataset(name="reflection", data="[1 1 1]")
    bragg_angle = crystal_1.create_dataset(name="bragg_angle", data=get_entry(data, "moth1"))
    bragg_angle.attrs["units"] = "degrees"

    crystal_2 = mono.create_group("crystal_2")
    crystal_2.attrs["NX_class"] = "NXcrystal"
    crystal_2.create_dataset(name="usage", data="Bragg")
    crystal_2.create_dataset(name="order_no", data="2")
    crystal_2.create_dataset(name="reflection", data="[1 1 1]")
    bragg_angle = crystal_2.create_dataset(name="bragg_angle", data=get_entry(data, "moth1"))
    bragg_angle.attrs["units"] = "degrees"
    bend_x = crystal_2.create_dataset(name="bend_x", data=get_entry(data, "mobd"))
    bend_x.attrs["units"] = "degrees"

    xbpm4 = instrument.create_group("XBPM4")
    xbpm4.attrs["NX_class"] = "NXdetector"
    xbpm4_sum = xbpm4.create_group("XBPM4_sum")
    xbpm4_sum_data = xbpm4_sum.create_dataset(name="data", data=get_entry(data, "bpm4s"))
    xbpm4_sum_data.attrs["units"] = "NX_DIMENSIONLESS"
    xbpm4_sum.create_dataset(name="description", data="Sum of counts for the four quadrants.")
    xbpm4_x = xbpm4.create_group("XBPM4_x")
    xbpm4_x_data = xbpm4_x.create_dataset(name="data", data=get_entry(data, "bpm4x"))
    xbpm4_x_data.attrs["units"] = "NX_DIMENSIONLESS"
    xbpm4_x.create_dataset(
        name="description",
        data="Normalized difference of counts between left and right quadrants.",
    )
    xbpm4_y = xbpm4.create_group("XBPM4_y")
    xbpm4_y_data = xbpm4_y.create_dataset(name="data", data=get_entry(data, "bpm4y"))
    xbpm4_y_data.attrs["units"] = "NX_DIMENSIONLESS"
    xbpm4_y.create_dataset(
        name="description",
        data="Normalized difference of counts between high and low quadrants.",
    )
    xbpm4_skew = xbpm4.create_group("XBPM4_skew")
    xbpm4_skew_data = xbpm4_skew.create_dataset(name="data", data=get_entry(data, "bpm4z"))
    xbpm4_skew_data.attrs["units"] = "NX_DIMENSIONLESS"
    xbpm4_skew.create_dataset(
        name="description",
        data="Normalized difference of counts between diagonal quadrants.",
    )

    mirror = instrument.create_group("mirror")
    mirror.attrs["NX_class"] = "NXmirror"
    mirror.create_dataset(name="type", data="single")
    mirror.create_dataset(
        name="description",
        data="Grazing incidence mirror to reject high-harmonic wavelengths from the monochromator. There are three coating options available that are used depending on the X-ray energy, no coating (SiO2), rhodium (Rh) or platinum (Pt).",
    )
    incident_angle = mirror.create_dataset(name="incident_angle", data=get_entry(data, "mith"))
    incident_angle.attrs["units"] = "degrees"
    substrate_material = mirror.create_dataset(name="substrate_material", data="SiO2")
    substrate_material.attrs["units"] = "NX_CHAR"
    coating_material = mirror.create_dataset(name="coating_material", data="SiO2")
    coating_material.attrs["units"] = "NX_CHAR"
    bend_y = mirror.create_dataset(name="bend_y", data="mibd")
    bend_y.attrs["units"] = "NX_DIMENSIONLESS"
    distance = mirror.create_dataset(
        name="distance", data=-4370 - np.asarray(get_entry(data, "samz", 0))
    )
    distance.attrs["units"] = "mm"

    xbpm5 = instrument.create_group("XBPM5")
    xbpm5.attrs["NX_class"] = "NXdetector"
    xbpm5_sum = xbpm5.create_group("XBPM5_sum")
    xbpm5_sum_data = xbpm5_sum.create_dataset(name="data", data=get_entry(data, "bpm5s"))
    xbpm5_sum_data.attrs["units"] = "NX_DIMENSIONLESS"
    xbpm5_sum.create_dataset(name="description", data="Sum of counts for the four quadrants.")
    xbpm5_x = xbpm5.create_group("XBPM5_x")
    xbpm5_x_data = xbpm5_x.create_dataset(name="data", data=get_entry(data, "bpm5x"))
    xbpm5_x_data.attrs["units"] = "NX_DIMENSIONLESS"
    xbpm5_x.create_dataset(
        name="description",
        data="Normalized difference of counts between left and right quadrants.",
    )
    xbpm5_y = xbpm5.create_group("XBPM5_y")
    xbpm5_y_data = xbpm5_y.create_dataset(name="data", data=get_entry(data, "bpm5y"))
    xbpm5_y_data.attrs["units"] = "NX_DIMENSIONLESS"
    xbpm5_y.create_dataset(
        name="description",
        data="Normalized difference of counts between high and low quadrants.",
    )
    xbpm5_skew = xbpm5.create_group("XBPM5_skew")
    xbpm5_skew_data = xbpm5_skew.create_dataset(name="data", data=get_entry(data, "bpm5z"))
    xbpm5_skew_data.attrs["units"] = "NX_DIMENSIONLESS"
    xbpm5_skew.create_dataset(
        name="description",
        data="Normalized difference of counts between diagonal quadrants.",
    )

    slit_2 = instrument.create_group("slit_2")
    slit_2.attrs["NX_class"] = "NXslit"
    source.create_dataset(name="material", data="Ag")
    source.create_dataset(name="description", data="Slit 2, optics hutch")
    x_gap = source.create_dataset(name="x_gap", data=get_entry(data, "sl2wh"))
    x_gap.attrs["units"] = "mm"
    y_gap = source.create_dataset(name="y_gap", data=get_entry(data, "sl2wv"))
    y_gap.attrs["units"] = "mm"
    x_translation = source.create_dataset(name="x_translation", data=get_entry(data, "sl2ch"))
    x_translation.attrs["units"] = "mm"
    height = source.create_dataset(name="x_translation", data=get_entry(data, "sl2cv"))
    height.attrs["units"] = "mm"
    distance = source.create_dataset(
        name="distance", data=-3140 - np.asarray(get_entry(data, "samz", 0))
    )
    distance.attrs["units"] = "mm"

    slit_3 = instrument.create_group("slit_3")
    slit_3.attrs["NX_class"] = "NXslit"
    source.create_dataset(name="material", data="Si")
    source.create_dataset(name="description", data="Slit 3, experimental hutch, exposure box")
    x_gap = source.create_dataset(name="x_gap", data=get_entry(data, "sl3wh"))
    x_gap.attrs["units"] = "mm"
    y_gap = source.create_dataset(name="y_gap", data=get_entry(data, "sl3wv"))
    y_gap.attrs["units"] = "mm"
    x_translation = source.create_dataset(name="x_translation", data=get_entry(data, "sl3ch"))
    x_translation.attrs["units"] = "mm"
    height = source.create_dataset(name="x_translation", data=get_entry(data, "sl3cv"))
    height.attrs["units"] = "mm"
    # distance = source.create_dataset(name="distance", data=-3140 - get_entry(data, "samz", 0))
    # distance.attrs["units"] = "mm"

    filter_set = instrument.create_group("filter_set")
    filter_set.attrs["NX_class"] = "NXattenuator"
    filter_set.create_dataset(name="material", data="Si")
    filter_set.create_dataset(
        name="description",
        data="The filter set consists of 4 linear stages, each with five filter positions. Additionally, each one allows for an out position to allow 'no filtering'.",
    )
    attenuator_transmission = filter_set.create_dataset(
        name="attenuator_transmission", data=10 ** get_entry(data, "ftrans", 0)
    )
    attenuator_transmission.attrs["units"] = "NX_DIMENSIONLESS"

    slit_4 = instrument.create_group("slit_4")
    slit_4.attrs["NX_class"] = "NXslit"
    source.create_dataset(name="material", data="Si")
    source.create_dataset(name="description", data="Slit 4, experimental hutch, exposure box")
    x_gap = source.create_dataset(name="x_gap", data=get_entry(data, "sl4wh"))
    x_gap.attrs["units"] = "mm"
    y_gap = source.create_dataset(name="y_gap", data=get_entry(data, "sl4wv"))
    y_gap.attrs["units"] = "mm"
    x_translation = source.create_dataset(name="x_translation", data=get_entry(data, "sl4ch"))
    x_translation.attrs["units"] = "mm"
    height = source.create_dataset(name="x_translation", data=get_entry(data, "sl4cv"))
    height.attrs["units"] = "mm"
    # distance = source.create_dataset(name="distance", data=-3140 - get_entry(data, "samz", 0))
    # distance.attrs["units"] = "mm"

    slit_5 = instrument.create_group("slit_5")
    slit_5.attrs["NX_class"] = "NXslit"
    source.create_dataset(name="material", data="Si")
    source.create_dataset(name="description", data="Slit 5, experimental hutch, exposure box")
    x_gap = source.create_dataset(name="x_gap", data=get_entry(data, "sl5wh"))
    x_gap.attrs["units"] = "mm"
    y_gap = source.create_dataset(name="y_gap", data=get_entry(data, "sl5wv"))
    y_gap.attrs["units"] = "mm"
    x_translation = source.create_dataset(name="x_translation", data=get_entry(data, "sl5ch"))
    x_translation.attrs["units"] = "mm"
    height = source.create_dataset(name="x_translation", data=get_entry(data, "sl5cv"))
    height.attrs["units"] = "mm"
    # distance = source.create_dataset(name="distance", data=-3140 - get_entry(data, "samz", 0))
    # distance.attrs["units"] = "mm"

    beam_stop_1 = instrument.create_group("beam_stop_1")
    beam_stop_1.attrs["NX_class"] = "NX_beamstop"
    beam_stop_1.create_dataset(name="description", data="circular")
    bms1_size = beam_stop_1.create_dataset(name="size", data=3)
    bms1_size.attrs["units"] = "mm"
    bms1_x = beam_stop_1.create_dataset(name="size", data=get_entry(data, "bs1x"))
    bms1_x.attrs["units"] = "mm"
    bms1_y = beam_stop_1.create_dataset(name="size", data=get_entry(data, "bs1y"))
    bms1_y.attrs["units"] = "mm"

    beam_stop_2 = instrument.create_group("beam_stop_2")
    beam_stop_2.attrs["NX_class"] = "NX_beamstop"
    beam_stop_2.create_dataset(name="description", data="rectangular")
    bms2_size_x = beam_stop_2.create_dataset(name="size_x", data=5)
    bms2_size_x.attrs["units"] = "mm"
    bms2_size_y = beam_stop_2.create_dataset(name="size_y", data=2.25)
    bms2_size_y.attrs["units"] = "mm"
    bms2_x = beam_stop_2.create_dataset(name="size", data=get_entry(data, "bs2x"))
    bms2_x.attrs["units"] = "mm"
    bms2_y = beam_stop_2.create_dataset(name="size", data=get_entry(data, "bs2y"))
    bms2_y.attrs["units"] = "mm"
    bms2_data = beam_stop_2.create_dataset(name="data", data=get_entry(data, "diode"))
    bms2_data.attrs["units"] = "NX_DIMENSIONLESS"

    if "eiger1p5m" in device_manager.devices and device_manager.devices.eiger1p5m.enabled:
        eiger_4 = instrument.create_group("eiger_4")
        eiger_4.attrs["NX_class"] = "NXdetector"
        x_pixel_size = eiger_4.create_dataset(name="x_pixel_size", data=75)
        x_pixel_size.attrs["units"] = "um"
        y_pixel_size = eiger_4.create_dataset(name="y_pixel_size", data=75)
        y_pixel_size.attrs["units"] = "um"
        polar_angle = eiger_4.create_dataset(name="polar_angle", data=0)
        polar_angle.attrs["units"] = "degrees"
        azimuthal_angle = eiger_4.create_dataset(name="azimuthal_angle", data=0)
        azimuthal_angle.attrs["units"] = "degrees"
        rotation_angle = eiger_4.create_dataset(name="rotation_angle", data=0)
        rotation_angle.attrs["units"] = "degrees"
        description = eiger_4.create_dataset(
            name="description",
            data="Single-photon counting detector, 320 micron-thick Si chip",
        )
        orientation = eiger_4.create_group("orientation")
        orientation.attrs[
            "description"
        ] = "Orientation defines the number of counterclockwise rotations by 90 deg followed by a transposition to reach the 'cameraman orientation', that is looking towards the beam."
        orientation.create_dataset(name="transpose", data=1)
        orientation.create_dataset(name="rot90", data=3)

    if (
        "eiger9m" in device_manager.devices
        and device_manager.devices.eiger9m.enabled
        and "eiger9m" in file_references
    ):
        eiger9m = instrument.create_group("eiger9m")
        eiger9m.attrs["NX_class"] = "NXdetector"
        x_pixel_size = eiger9m.create_dataset(name="x_pixel_size", data=75)
        x_pixel_size.attrs["units"] = "um"
        y_pixel_size = eiger9m.create_dataset(name="y_pixel_size", data=75)
        y_pixel_size.attrs["units"] = "um"
        polar_angle = eiger9m.create_dataset(name="polar_angle", data=0)
        polar_angle.attrs["units"] = "degrees"
        azimuthal_angle = eiger9m.create_dataset(name="azimuthal_angle", data=0)
        azimuthal_angle.attrs["units"] = "degrees"
        rotation_angle = eiger9m.create_dataset(name="rotation_angle", data=0)
        rotation_angle.attrs["units"] = "degrees"
        description = eiger9m.create_dataset(
            name="description",
            data="Eiger9M detector, in-house developed, Paul Scherrer Institute",
        )
        orientation = eiger9m.create_group("orientation")
        orientation.attrs[
            "description"
        ] = "Orientation defines the number of counterclockwise rotations by 90 deg followed by a transposition to reach the 'cameraman orientation', that is looking towards the beam."
        orientation.create_dataset(name="transpose", data=1)
        orientation.create_dataset(name="rot90", data=3)
        data = eiger9m.create_ext_link("data", file_references["eiger9m"]["path"], "EG9M/data")
        status = eiger9m.create_ext_link(
            "status", file_references["eiger9m"]["path"], "EG9M/status"
        )

    if (
        "pilatus_2" in device_manager.devices
        and device_manager.devices.pilatus_2.enabled
        and "pilatus_2" in file_references
    ):
        pilatus_2 = instrument.create_group("pilatus_2")
        pilatus_2.attrs["NX_class"] = "NXdetector"
        x_pixel_size = pilatus_2.create_dataset(name="x_pixel_size", data=172)
        x_pixel_size.attrs["units"] = "um"
        y_pixel_size = pilatus_2.create_dataset(name="y_pixel_size", data=172)
        y_pixel_size.attrs["units"] = "um"
        polar_angle = pilatus_2.create_dataset(name="polar_angle", data=0)
        polar_angle.attrs["units"] = "degrees"
        azimuthal_angle = pilatus_2.create_dataset(name="azimuthal_angle", data=0)
        azimuthal_angle.attrs["units"] = "degrees"
        rotation_angle = pilatus_2.create_dataset(name="rotation_angle", data=0)
        rotation_angle.attrs["units"] = "degrees"
        description = pilatus_2.create_dataset(
            name="description", data="Pilatus 300K detector, Dectris, Switzerland"
        )
        orientation = pilatus_2.create_group("orientation")
        orientation.attrs[
            "description"
        ] = "Orientation defines the number of counterclockwise rotations by 90 deg followed by a transposition to reach the 'cameraman orientation', that is looking towards the beam."
        orientation.create_dataset(name="transpose", data=1)
        orientation.create_dataset(name="rot90", data=2)
        data = pilatus_2.create_ext_link(
            "data",
            file_references["pilatus_2"]["path"],
            "entry/instrument/pilatus_2/data",
        )

    if (
        "falcon" in device_manager.devices
        and device_manager.devices.falcon.enabled
        and "falcon" in file_references
    ):
        falcon = instrument.create_ext_link(
            "falcon",
            file_references["falcon"]["path"],
            "entry/instrument/FalconX1",
        )

    return storage
