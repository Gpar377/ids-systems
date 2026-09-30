FROM debian:bookworm-slim
# Traffic generation toolkit for the isolated lab. nmap for recon-style scans,
# hping3 for crafted-packet timing/volume control, curl/netcat for benign
# sessions, python for scripted scenarios.
RUN apt-get update && apt-get install -y --no-install-recommends \
      nmap hping3 curl netcat-openbsd iproute2 python3 python3-minimal \
    && rm -rf /var/lib/apt/lists/*
CMD ["sleep", "infinity"]
