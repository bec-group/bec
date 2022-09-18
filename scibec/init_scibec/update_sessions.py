import datetime
import json

import requests
import urllib3
import yaml


def get_sessions():
    headers = {"Content-type": "application/json"}
    res = requests.get(scibec_url + "/sessions", headers=headers, verify=False)
    resJSON = json.loads(res.content)
    return resJSON


def delete_all_session():
    headers = {"Content-type": "application/json"}
    resJSON = get_sessions()
    sessionIDs = [session["id"] for session in resJSON]
    for id in sessionIDs:
        res = requests.delete(scibec_url + "/sessions/" + id, headers=headers, verify=False)


def add_session():
    headers = {"Content-type": "application/json"}
    obj = {"ownerGroup": "test", "accessGroups": ["customer"], "name": "test"}
    url = scibec_url + "/sessions"
    res = requests.post(url, json=obj, headers=headers, verify=False)
    resJSON = json.loads(res.content)
    return resJSON


def make_filter(
    where: dict = None,
    limit: int = 0,
    skip: int = 0,
    fields: dict = None,
    include: dict = None,
    order: list = None,
):
    filt = dict()
    if where is not None:
        items = [where.copy()]
        filt["where"] = {"and": items}
    if limit > 0:
        filt["limit"] = limit
    if skip > 0:
        filt["skip"] = skip
    if fields is not None:
        filt["fields"] = include
    if order is not None:
        filt["order"] = order
    filt = json.dumps(filt)
    return {"filter": filt}


def add_device(
    *, name, enabled, deviceClass, deviceGroup, deviceConfig, acquisitionConfig, session
):
    headers = {"Content-type": "application/json"}
    obj = {
        "ownerGroup": "test",
        "accessGroups": ["customer"],
        "name": name,
        "sessionId": session,
        "enabled": enabled,
        "deviceClass": deviceClass,
        "deviceGroup": deviceGroup,
        "deviceConfig": deviceConfig,
        "acquisitionConfig": acquisitionConfig,
    }
    url = scibec_url + "/devices"
    res = requests.post(url, json=obj, headers=headers, verify=False)
    resJSON = json.loads(res.content)
    return resJSON


if __name__ == "__main__":
    import argparse

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    scibec_url = "http://localhost:3030"
    # config_path = "./dummy_config.yaml"
    config_path = "./init_scibec/demo_config.yaml"

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config",
        default="./init_scibec/demo_config.yaml",
        help="path to the config file",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:3030",
        help="scibec url",
    )
    clargs = parser.parse_args()
    config_path = clargs.config
    scibec_url = clargs.url

    delete_all_session()
    print(get_sessions())
    session = add_session()
    sessionId = session["id"]
    print(session)
    data = []
    with open(config_path, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as er:
            print(f"Error while loading config from disk: {repr(er)}")

    for key, val in data.items():
        add_device(
            name=key,
            enabled=val["status"]["enabled"],
            deviceClass=val["type"],
            deviceGroup=val["deviceGroup"],
            deviceConfig=val["config"],
            acquisitionConfig=val["acquisition"],
            session=sessionId,
        )

    dump_to_disk = True
    if dump_to_disk:
        filter = make_filter(where={"sessionId": sessionId})
        session = requests.get(
            scibec_url + "/devices",
            params=filter,
            verify=False,
        )
        with open("test_session.yaml", "w+") as fstream:
            yaml.dump(json.loads(session.content), fstream)
