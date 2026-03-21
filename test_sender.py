"""
test_sender.py — Send test UDP frames to a running SNN agent.

The agent must already be running:
  python server.py

Usage:
  python test_sender.py               # random 16-bit values at 20 fps
  python test_sender.py 42            # fixed value  42  at 20 fps
  python test_sender.py --sweep       # sweep 0 → 65535 repeatedly
  python test_sender.py --fps 100     # override frame rate (random mode)

Frame format:  struct.pack('!HI', 0xABCD, value)
  magic  uint16  0xABCD  (validation sentinel)
  value  uint32  integer to send  (input_bit_width LSBs used by agent)
"""

import socket
import struct
import time
import sys
import random

HOST  = "127.0.0.1"
PORT  = 9000          # must match udp_port in config.py
MAGIC = 0xABCD
FMT   = "!HI"
FPS   = 20

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def send(value: int) -> None:
    sock.sendto(struct.pack(FMT, MAGIC, value & 0xFFFFFFFF), (HOST, PORT))


def main() -> None:
    args  = sys.argv[1:]
    fps   = FPS

    # Parse --fps override
    if "--fps" in args:
        idx = args.index("--fps")
        fps = int(args[idx + 1])
        args = [a for i, a in enumerate(args) if i not in (idx, idx + 1)]

    delay = 1.0 / fps
    mode  = args[0] if args else "random"

    if mode == "--sweep":
        print(f"Sweeping 0→65535 → {HOST}:{PORT}  ({fps} fps)  Ctrl+C to stop.")
        v = 0
        while True:
            send(v)
            v = (v + 1) % 65536
            time.sleep(delay)

    elif mode.lstrip("-").isdigit():
        value = int(mode)
        print(f"Sending fixed value {value} → {HOST}:{PORT}  ({fps} fps)  Ctrl+C to stop.")
        while True:
            send(value)
            time.sleep(delay)

    else:  # random
        print(f"Sending random 16-bit values → {HOST}:{PORT}  ({fps} fps)  Ctrl+C to stop.")
        while True:
            send(random.randint(0, 0xFFFF))
            time.sleep(delay)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSender stopped.")
