import redis
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import typer


def main(filename: str, redis_ip: str = "localhost"):
    c = redis.Redis(redis_ip)
    res = c.xrevrange("box_tracking", "+", "-", count=1)
    assert res, "Please make sure tracking is running"
    init_time, _ = res[0]  # type: ignore
    init_time: str
    print("Recording!")
    wait_for_env_reset(c)


def wait_for_env_reset(c: redis.Redis):
    while 1:
        last_id = "$"
        messages = r.xread({"cart_cmd": last_id}, block=100, count=1)
        if messages:
            message_id, payload = messages[0][1][-1]
            last_id = message_id
            if payload["cmd"] == "RESET":
                break


if __name__ == "__main__":
    typer.run(main)
