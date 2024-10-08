import redis
import numpy as np


def main(wait: bool = True, redis_ip: str = "localhost"):
    c = redis.Redis(redis_ip)
    if wait:
        wait_for_env_reset()
    res = c.xrevrange("cart_cmd", "+", "-", count=1)
    assert res, "Please make sure tracking is running"
    init_time, _ = res[0]  # type: ignore
    init_time: str
    print(init_time)


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
