"""Read-only clock evidence from correlated G1 GET_FSM responses.

Publishes only API 7001 requests. No clock adjustment, velocity or FSM setter.
Response writer time must lie between request send and response receive after
clock conversion; these samples bound an offset, not one-way network latency.
"""
import argparse
import json
from pathlib import Path
import secrets
import sys
import time
from xml.sax.saxutils import quoteattr

from g1_dds_diagnostic import _progress, _run_supervised


def offset_interval(send_ns, receive_ns, remote_ns):
    if any(type(value) is not int or value <= 0 for value in (send_ns, receive_ns, remote_ns)) or receive_ns < send_ns:
        raise ValueError("invalid clock sample")
    return {"round_trip_s": (receive_ns - send_ns) / 1e9,
            "local_minus_remote_min_s": (send_ns - remote_ns) / 1e9,
            "local_minus_remote_max_s": (receive_ns - remote_ns) / 1e9}


def collect(interface):
    from cyclonedds.domain import Domain, DomainParticipant
    from cyclonedds.topic import Topic
    from cyclonedds.pub import DataWriter
    from cyclonedds.sub import DataReader
    from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicDcpsPublication
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, BmsState_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
    from unitree_sdk2py.idl.unitree_api.msg.dds_ import (
        Request_, RequestHeader_, RequestIdentity_, RequestLease_, RequestPolicy_, Response_)
    config = ('<CycloneDDS><Domain Id="0"><General><Interfaces><NetworkInterface name=' +
              quoteattr(interface) + '/></Interfaces></General></Domain></CycloneDDS>')
    domain = Domain(0, config)
    participant = DomainParticipant(0)
    request_topic = Topic(participant, "rt/api/sport/request", Request_)
    response_topic = Topic(participant, "rt/api/sport/response", Response_)
    writer, reader = DataWriter(participant, request_topic), DataReader(participant, response_topic)
    discovery = BuiltinDataReader(participant, BuiltinTopicDcpsPublication)
    telemetry_specs = [("rt/lowstate", LowState_), ("rt/odommodestate", SportModeState_),
                       ("rt/lf/bmsstate", BmsState_)]
    telemetry_topics = [Topic(participant, name, kind) for name, kind in telemetry_specs]
    telemetry_readers = [DataReader(participant, topic) for topic in telemetry_topics]
    time.sleep(1.)
    samples = []
    for _ in range(3):
        request_id = secrets.randbits(62)
        request = Request_(RequestHeader_(RequestIdentity_(request_id, 7001),
                                         RequestLease_(0), RequestPolicy_(0, False)), "{}", [])
        sent_ns, mono = time.time_ns(), time.monotonic()
        writer.write(request)
        matched = False
        while time.monotonic() - mono < 2:
            for response in reader.take(32):
                received_ns = time.time_ns()
                if not isinstance(response, Response_) or response.header.identity.id != request_id:
                    continue
                if response.header.identity.api_id != 7001:
                    raise ValueError("response API mismatch")
                sample = offset_interval(sent_ns, received_ns, response.sample_info.source_timestamp)
                sample.update(request_id=request_id, code=response.header.status.code,
                              data=response.data, publication_handle=response.sample_info.publication_handle)
                samples.append(sample)
                _progress("GET_FSM 时钟样本：" + json.dumps(sample))
                matched = True
                break
            if matched:
                break
            time.sleep(.005)
        if not matched:
            samples.append({"request_id": request_id, "error": "GET_FSM response timeout"})
        time.sleep(.1)
    endpoints = {}
    for endpoint in discovery.take(256):
        if endpoint.sample_info.valid_data:
            endpoints[endpoint.sample_info.instance_handle] = {
                "writer_guid": str(endpoint.key), "participant_guid": str(endpoint.participant_key),
                "topic": endpoint.topic_name}
    telemetry = {}
    for (name, kind), telemetry_reader in zip(telemetry_specs, telemetry_readers):
        for sample in telemetry_reader.take(64):
            if isinstance(sample, kind):
                telemetry[name] = {"publication_handle": sample.sample_info.publication_handle,
                                   "writer": endpoints.get(sample.sample_info.publication_handle),
                                   "source_timestamp_ns": sample.sample_info.source_timestamp}
    for sample in samples:
        sample["writer"] = endpoints.get(sample.get("publication_handle"))
    print(json.dumps({"interface": interface, "samples": samples, "telemetry_writers": telemetry,
                      "motion_commands_sent": 0, "clocks_modified": False,
                      "limitations": ["Offsets apply to the response writer only; telemetry writers may have different clocks.",
                                      "No automatic time correction or freshness-check bypass."]}, indent=2), flush=True)
    return 0 if all("round_trip_s" in row and row["code"] == 0 for row in samples) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker:
        return _run_supervised([sys.executable, "-u", str(Path(__file__).resolve()),
                               "--worker", "--interface", args.interface], 15)
    return collect(args.interface)


if __name__ == "__main__":
    raise SystemExit(main())
