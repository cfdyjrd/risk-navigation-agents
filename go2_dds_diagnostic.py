"""Read-only DDS diagnostic. Never creates command publishers or sport clients."""
import argparse
import ipaddress
import socket
import time
import traceback
from xml.sax.saxutils import quoteattr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interface', default='en0')
    parser.add_argument('--duration', type=int, default=15)
    parser.add_argument('--peer', type=ipaddress.IPv4Address,
                        help='Optional robot IPv4 address for DDS unicast discovery')
    args = parser.parse_args()
    if args.interface not in dict(socket.if_nameindex()).values():
        parser.error('network interface does not exist')
    if not 1 <= args.duration <= 60:
        parser.error('duration must be 1–60 seconds')

    from cyclonedds.domain import Domain, DomainParticipant
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_, LowState_

    discovery = (f'<Discovery><ParticipantIndex>auto</ParticipantIndex>'
                 f'<Peers><Peer Address={quoteattr(str(args.peer))}/></Peers>'
                 '</Discovery>' if args.peer else '')
    config = ('<CycloneDDS><Domain Id="0"><General><Interfaces>'
              f'<NetworkInterface name={quoteattr(args.interface)}/>'
              f'</Interfaces></General>{discovery}</Domain></CycloneDDS>')
    domain = Domain(0, config)
    participant = DomainParticipant(0)
    topics = [Topic(participant, name, kind) for name, kind in (
        ('rt/sportmodestate', SportModeState_),
        ('rt/lowstate', LowState_),
    )]
    readers = [DataReader(participant, topic) for topic in topics]
    counts = [0, 0]
    last_print = [0.0, 0.0]
    print(f'只读 DDS 检查：interface={args.interface}, domain=0', flush=True)
    if args.peer:
        print(f'附加发现地址：{args.peer}（不限制数据来源）', flush=True)
    end = time.monotonic() + args.duration
    while time.monotonic() < end:
        for index, reader in enumerate(readers):
            # Nonblocking take avoids the SDK wrapper that hides exceptions.
            samples = reader.take(32)
            for sample in samples:
                if not isinstance(sample, (SportModeState_, LowState_)):
                    continue
                counts[index] += 1
                now = time.monotonic()
                if now - last_print[index] >= 1:
                    last_print[index] = now
                    details = (f'position={list(sample.position)}, velocity={list(sample.velocity)}'
                               if index == 0 else f'battery_soc={sample.bms_state.soc}')
                    print(f'{topics[index].name}: {details}', flush=True)
        time.sleep(0.05)
    print(f'收到有效样本：sport={counts[0]}, low={counts[1]}', flush=True)
    print('接收数据不验证设备身份或控制权限；请现场核对数据来源。')
    return 0 if any(counts) else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\n已结束只读检查。')
    except Exception:
        traceback.print_exc()
        raise SystemExit(2)
