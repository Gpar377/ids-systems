"""Minimal multi-port TCP victim.

Not a real ssh/http/ftp stack -- it only needs to complete or refuse handshakes
so that captured connections resolve to genuine NSL-KDD `flag` values. Each open
port sends a short banner and echoes one line, giving a normal-looking SF
connection. Everything else on the host is closed -> RST -> REJ.
"""
import socket
import threading

# port -> banner. Chosen to map onto NSL-KDD `service` names later.
PORTS = {
    21: b"220 ftp ready\r\n",     # ftp
    22: b"SSH-2.0-OpenSSH_9\r\n",  # ssh
    80: b"HTTP/1.0 200 OK\r\n\r\n",  # http
    143: b"* OK IMAP\r\n",         # imap4
}


def serve(port: int, banner: bytes) -> None:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", port))
    s.listen(64)
    while True:
        try:
            conn, _ = s.accept()
        except OSError:
            continue
        threading.Thread(target=_handle, args=(conn, banner), daemon=True).start()


def _handle(conn: socket.socket, banner: bytes) -> None:
    try:
        conn.sendall(banner)
        conn.settimeout(2.0)
        data = conn.recv(1024)      # read one request line if any
        if data:
            conn.sendall(b"ok\r\n")  # dst_bytes > 0 on interactive connections
    except OSError:
        pass
    finally:
        conn.close()               # clean close -> SF, not S0/S1


def main() -> None:
    for port, banner in PORTS.items():
        threading.Thread(target=serve, args=(port, banner), daemon=True).start()
    print(f"victim up on {sorted(PORTS)} (all others closed -> REJ)", flush=True)
    threading.Event().wait()


if __name__ == "__main__":
    main()
