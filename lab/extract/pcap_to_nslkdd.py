"""PCAP -> NSL-KDD feature vectors.

Turns a real capture into rows in the same 41-feature space the model consumes,
so the taxonomy stops being an assumption and becomes a measurement. The window
definitions below follow the KDD Cup 1999 derivation (Lee and Stolfo), which
NSL-KDD inherits:

  time-based (23-31): connections in the PAST 2 SECONDS relative to this
                      connection's start.
  host-based (32-41): the LAST 100 connections to this connection's destination
                      host (a count-based window, not a time window).

What is NOT recoverable from a black-box PCAP: the content features (10-22).
Those are victim/host state (logged_in, num_root, ...) -- by the taxonomy they
are Class C, victim-controlled, precisely because they never appear on the wire.
They are emitted as 0 and flagged, which is itself evidence for the mask.

Scope: TCP (the SYN/RST state machine gives the interesting `flag` values).
Deterministic, single pass over connections sorted by start time.
"""

from __future__ import annotations

import argparse
import csv
import socket
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import dpkt

# ip.off carries the fragment flags/offset; dpkt warns but the value is correct.
warnings.filterwarnings("ignore", message="IP.off is deprecated")

# dst port -> NSL-KDD service name, spelled exactly as in the dataset (imap4,
# not imap) or the encoder treats it as unseen. Unmapped ports -> private.
SERVICE = {80: "http", 22: "ssh", 21: "ftp", 143: "imap4", 23: "telnet",
           25: "smtp", 53: "domain", 443: "http_443", 110: "pop_3"}

# NSL-KDD "content" features -- not on the wire. Emitted 0, see module docstring.
CONTENT_ZERO = ["hot", "num_failed_logins", "logged_in", "num_compromised",
                "root_shell", "su_attempted", "num_root", "num_file_creations",
                "num_shells", "num_access_files", "num_outbound_cmds",
                "is_host_login", "is_guest_login"]

SYN, ACK, FIN, RST, URG = 0x02, 0x10, 0x01, 0x04, 0x20


@dataclass
class Conn:
    """One TCP connection, keyed by (src, sport, dst, dport)."""
    src: str
    sport: int
    dst: str
    dport: int
    proto: str = "tcp"
    start: float = 0.0
    end: float = 0.0
    src_bytes: int = 0
    dst_bytes: int = 0
    urgent: int = 0
    wrong_fragment: int = 0
    src_flags: int = 0          # OR of all TCP flags src->dst
    dst_flags: int = 0          # OR of all TCP flags dst->src
    saw_synack: bool = False    # did responder complete the handshake?

    def service(self) -> str:
        return SERVICE.get(self.dport, "private")

    def land(self) -> int:
        return int(self.src == self.dst and self.sport == self.dport)

    def flag(self) -> str:
        """Derive the NSL-KDD connection-status flag from observed TCP flags.

        Victim-coupled by construction (it depends on how the responder
        answered) -- which is why the taxonomy treats `flag` as not freely
        attacker-settable."""
        s, d = self.src_flags, self.dst_flags
        syn_only = (s & SYN) and not self.saw_synack
        if self.saw_synack:                       # handshake completed
            if s & RST:
                return "RSTO"                      # originator reset it
            if d & RST:
                return "RSTR"                      # responder reset it
            if (s & FIN) or (d & FIN):
                return "SF"                        # normal establish + close
            return "S1"                            # established, not closed
        if d & RST:
            return "REJ"                           # SYN answered with RST
        if syn_only and (s & FIN):
            return "SH"                            # SYN then FIN, no reply
        if syn_only:
            return "S0"                            # SYN sent, no reply
        return "OTH"                               # midstream / no SYN seen

    def is_serror(self) -> bool:                   # SYN-error family
        return self.flag() in ("S0", "S1", "S2", "S3")

    def is_rerror(self) -> bool:                   # REJ / reset family
        return self.flag() in ("REJ", "RSTO", "RSTR", "RSTOS0")


def _ip(b: bytes) -> str:
    return socket.inet_ntoa(b)


def parse_connections(pcap_path: Path) -> list[Conn]:
    conns: dict[tuple, Conn] = {}
    order: list[Conn] = []
    with open(pcap_path, "rb") as f:
        try:
            reader = dpkt.pcap.Reader(f)
        except ValueError:
            f.seek(0)
            reader = dpkt.pcapng.Reader(f)
        for ts, buf in reader:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
            except dpkt.dpkt.UnpackError:
                continue
            if not isinstance(eth.data, dpkt.ip.IP):
                continue
            ip = eth.data
            frag = bool(ip.off & (dpkt.ip.IP_MF | dpkt.ip.IP_OFFMASK))
            if not isinstance(ip.data, dpkt.tcp.TCP):
                continue                          # TCP-only scope
            tcp = ip.data
            src, dst = _ip(ip.src), _ip(ip.dst)
            key = (src, tcp.sport, dst, tcp.dport)
            rev = (dst, tcp.dport, src, tcp.sport)

            # A fresh SYN (no live entry) starts a new connection.
            if (tcp.flags & SYN) and not (tcp.flags & ACK) and key not in conns:
                c = Conn(src=src, sport=tcp.sport, dst=dst, dport=tcp.dport,
                         start=ts, end=ts)
                conns[key] = c
                order.append(c)

            c = conns.get(key) or conns.get(rev)
            if c is None:                          # midstream, no SYN captured
                c = Conn(src=src, sport=tcp.sport, dst=dst, dport=tcp.dport,
                         start=ts, end=ts)
                conns[key] = c
                order.append(c)

            c.end = ts
            forward = (src == c.src and tcp.sport == c.sport)
            payload = len(tcp.data)
            if forward:
                c.src_flags |= tcp.flags
                c.src_bytes += payload
            else:
                c.dst_flags |= tcp.flags
                c.dst_bytes += payload
                if (tcp.flags & SYN) and (tcp.flags & ACK):
                    c.saw_synack = True
            if tcp.flags & URG:
                c.urgent += 1
            if frag:
                c.wrong_fragment += 1
    return sorted(order, key=lambda x: x.start)


def _rate(num: int, den: int) -> float:
    return round(num / den, 4) if den else 0.0


def compute_features(conns: list[Conn]) -> list[dict]:
    rows = []
    for i, c in enumerate(conns):
        # --- time-based: connections started within 2s before this one --------
        win = [p for p in conns[:i] if 0 <= c.start - p.start <= 2.0]
        same_host = [p for p in win if p.dst == c.dst]
        count = len(same_host)
        srv = [p for p in win if p.dport == c.dport]
        srv_count = len(srv)
        # --- host-based: last 100 connections to this destination host --------
        prior_host = [p for p in conns[:i] if p.dst == c.dst][-100:]
        dhc = len(prior_host)
        prior_srv = [p for p in prior_host if p.dport == c.dport]
        dhsc = len(prior_srv)

        row = {
            "duration": int(c.end - c.start),
            "protocol_type": c.proto,
            "service": c.service(),
            "flag": c.flag(),
            "src_bytes": c.src_bytes,
            "dst_bytes": c.dst_bytes,
            "land": c.land(),
            "wrong_fragment": c.wrong_fragment,
            "urgent": c.urgent,
        }
        row.update({k: 0 for k in CONTENT_ZERO})
        row.update({
            "count": count,
            "srv_count": srv_count,
            "serror_rate": _rate(sum(p.is_serror() for p in same_host), count),
            "srv_serror_rate": _rate(sum(p.is_serror() for p in srv), srv_count),
            "rerror_rate": _rate(sum(p.is_rerror() for p in same_host), count),
            "srv_rerror_rate": _rate(sum(p.is_rerror() for p in srv), srv_count),
            "same_srv_rate": _rate(sum(p.dport == c.dport for p in same_host), count),
            "diff_srv_rate": _rate(sum(p.dport != c.dport for p in same_host), count),
            "srv_diff_host_rate": _rate(sum(p.dst != c.dst for p in srv), srv_count),
            "dst_host_count": dhc,
            "dst_host_srv_count": dhsc,
            "dst_host_same_srv_rate": _rate(dhsc, dhc),
            "dst_host_diff_srv_rate": _rate(dhc - dhsc, dhc),
            "dst_host_same_src_port_rate":
                _rate(sum(p.sport == c.sport for p in prior_host), dhc),
            "dst_host_srv_diff_host_rate":
                _rate(sum(p.dst != c.dst for p in prior_srv), dhsc),
            "dst_host_serror_rate":
                _rate(sum(p.is_serror() for p in prior_host), dhc),
            "dst_host_srv_serror_rate":
                _rate(sum(p.is_serror() for p in prior_srv), dhsc),
            "dst_host_rerror_rate":
                _rate(sum(p.is_rerror() for p in prior_host), dhc),
            "dst_host_srv_rerror_rate":
                _rate(sum(p.is_rerror() for p in prior_srv), dhsc),
        })
        # provenance for the realizability comparison (dropped before ML use)
        row["_src_ip"] = c.src
        rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pcap", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--src", default=None,
                    help="keep only connections from this source IP")
    args = ap.parse_args()

    conns = parse_connections(args.pcap)
    rows = compute_features(conns)
    if args.src:
        rows = [r for r in rows if r["_src_ip"] == args.src]
    print(f"parsed {len(conns)} connections; {len(rows)} rows after filter")

    if rows:
        flags = Counter(r["flag"] for r in rows)
        svc = Counter(r["service"] for r in rows)
        print(f"  flags   : {dict(flags)}")
        print(f"  services: {dict(svc)}")
        print(f"  src_bytes range: {min(r['src_bytes'] for r in rows)}.."
              f"{max(r['src_bytes'] for r in rows)}")
        print(f"  count    range: {min(r['count'] for r in rows)}.."
              f"{max(r['count'] for r in rows)}")
        print(f"  dst_host_count range: {min(r['dst_host_count'] for r in rows)}.."
              f"{max(r['dst_host_count'] for r in rows)}")

    out = args.out or args.pcap.with_suffix(".csv")
    cols = list(rows[0].keys()) if rows else []
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
