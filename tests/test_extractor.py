"""Validate the PCAP extractor on a synthetic capture with known ground truth.

Builds packets by hand so we know exactly what flag/count/window values should
come out, then asserts the extractor recovers them. This proves the fiddly
window and flag logic without needing Docker or a real capture.
"""

import struct
import sys
from pathlib import Path

import dpkt
import pytest

LAB = Path(__file__).resolve().parent.parent / "lab" / "extract"
sys.path.insert(0, str(LAB))

from pcap_to_nslkdd import compute_features, parse_connections  # noqa: E402

SYN, ACK, FIN, RST = 0x02, 0x10, 0x01, 0x04
ATT = "192.168.90.20"
VIC = "192.168.90.10"
NOISE = "192.168.90.30"


def _pkt(src, dst, sport, dport, flags, payload=b""):
    import socket
    tcp = dpkt.tcp.TCP(sport=sport, dport=dport, flags=flags, data=payload)
    tcp.seq, tcp.ack = 1, 1
    ip = dpkt.ip.IP(src=socket.inet_aton(src), dst=socket.inet_aton(dst),
                    p=dpkt.ip.IP_PROTO_TCP, data=tcp)
    ip.len = len(ip)
    eth = dpkt.ethernet.Ethernet(
        src=b"\x00\x00\x00\x00\x00\x01", dst=b"\x00\x00\x00\x00\x00\x02",
        type=dpkt.ethernet.ETH_TYPE_IP, data=ip)
    return bytes(eth)


def _write(path, packets):
    with open(path, "wb") as f:
        w = dpkt.pcap.Writer(f)
        for ts, p in packets:
            w.writepkt(p, ts=ts)


@pytest.fixture
def sf_rej_s0_pcap(tmp_path):
    """One SF (open port), one REJ (closed port), one S0 (no reply)."""
    p = tmp_path / "basic.pcap"
    pk = []
    # SF: full handshake to :80 with payload, then FIN both ways.
    pk += [(0.0, _pkt(ATT, VIC, 5000, 80, SYN)),
           (0.01, _pkt(VIC, ATT, 80, 5000, SYN | ACK)),
           (0.02, _pkt(ATT, VIC, 5000, 80, ACK, b"GET / HTTP/1.0\r\n\r\n")),
           (0.03, _pkt(VIC, ATT, 80, 5000, ACK, b"HTTP/1.0 200 OK\r\n")),
           (0.04, _pkt(ATT, VIC, 5000, 80, FIN | ACK)),
           (0.05, _pkt(VIC, ATT, 80, 5000, FIN | ACK))]
    # REJ: SYN to a closed port answered with RST-ACK.
    pk += [(0.1, _pkt(ATT, VIC, 5001, 999, SYN)),
           (0.11, _pkt(VIC, ATT, 999, 5001, RST | ACK))]
    # S0: SYN with no reply at all.
    pk += [(0.2, _pkt(ATT, VIC, 5002, 8888, SYN))]
    _write(p, pk)
    return p


def test_flags_are_recovered(sf_rej_s0_pcap):
    conns = parse_connections(sf_rej_s0_pcap)
    rows = compute_features(conns)
    flags = {(r["service"], r["dst_host_count"]): r["flag"] for r in rows}
    got = sorted(r["flag"] for r in rows)
    assert got == sorted(["SF", "REJ", "S0"]), got


def test_src_bytes_and_service(sf_rej_s0_pcap):
    rows = compute_features(parse_connections(sf_rej_s0_pcap))
    sf = [r for r in rows if r["flag"] == "SF"][0]
    assert sf["service"] == "http"
    assert sf["src_bytes"] == len(b"GET / HTTP/1.0\r\n\r\n")
    assert sf["dst_bytes"] == len(b"HTTP/1.0 200 OK\r\n")


def test_land_flag(tmp_path):
    p = tmp_path / "land.pcap"
    _write(p, [(0.0, _pkt(VIC, VIC, 80, 80, SYN))])
    rows = compute_features(parse_connections(p))
    assert rows[0]["land"] == 1


@pytest.fixture
def burst_pcap(tmp_path):
    """40 rapid SYNs to :80 within the 2s window -> count should climb."""
    p = tmp_path / "burst.pcap"
    pk = []
    for i in range(40):
        t = i * 0.01                       # all within 0.4s -> same 2s window
        pk.append((t, _pkt(ATT, VIC, 6000 + i, 80, SYN)))
        pk.append((t + 0.001, _pkt(VIC, ATT, 80, 6000 + i, SYN | ACK)))
        pk.append((t + 0.002, _pkt(ATT, VIC, 6000 + i, 80, FIN | ACK)))
    _write(p, pk)
    return p


def test_count_and_srv_count_climb(burst_pcap):
    rows = compute_features(parse_connections(burst_pcap))
    last = rows[-1]
    # 39 prior same-host connections in the window before the 40th.
    assert last["count"] == 39, last["count"]
    assert last["srv_count"] == 39
    assert last["same_srv_rate"] == 1.0            # all to :80
    assert last["dst_host_count"] == 39


def test_host_window_caps_at_100(tmp_path):
    p = tmp_path / "many.pcap"
    pk = []
    for i in range(130):
        t = i * 0.001
        pk.append((t, _pkt(ATT, VIC, 7000 + i, 80, SYN)))
        pk.append((t + 0.0001, _pkt(VIC, ATT, 80, 7000 + i, RST | ACK)))
    _write(p, pk)
    rows = compute_features(parse_connections(p))
    assert rows[-1]["dst_host_count"] == 100, rows[-1]["dst_host_count"]


def test_third_party_moves_dst_host_features(tmp_path):
    """The Test-2 thesis in miniature: NOISE connections raise dst_host_count for
    an attacker connection that comes after them, though the attacker sent them
    nothing. dst_host_count aggregates over ALL sources to the victim."""
    p = tmp_path / "thirdparty.pcap"
    pk = []
    for i in range(20):                    # 20 noise conns to victim:80
        t = i * 0.001
        pk.append((t, _pkt(NOISE, VIC, 8000 + i, 80, SYN)))
        pk.append((t + 0.0001, _pkt(VIC, NOISE, 80, 8000 + i, SYN | ACK)))
    pk.append((0.05, _pkt(ATT, VIC, 9000, 80, SYN)))   # one attacker conn after
    pk.append((0.0501, _pkt(VIC, ATT, 80, 9000, SYN | ACK)))
    _write(p, pk)
    rows = compute_features(parse_connections(p))
    att = [r for r in rows if r["_src_ip"] == ATT][0]
    assert att["dst_host_count"] == 20, att["dst_host_count"]
    assert att["count"] >= 20                       # 2s window also sees them
