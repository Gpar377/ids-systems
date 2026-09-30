FROM python:3.12-slim
COPY server.py /server.py
# Open ports -> completed handshakes (NSL-KDD flag SF). Ports left unlisted are
# closed, so the kernel answers SYN with RST -> flag REJ. Both are needed so the
# capture contains real, differentiable connection-resolution states.
EXPOSE 21 22 80 143
CMD ["python", "/server.py"]
