from relative import main
import redis
import json
import typer


def mocap(redis_ip: str = "127.0.0.1"):
    redis_connection = redis.Redis(redis_ip)

    for frame in main():
        marker_id, transform = frame
        marker_id = int(marker_id)
        x, y = transform[:2, 3].flatten()
        transform = list(transform.flatten())
        payload = {"transform": json.dumps(transform)}
        redis_connection.xadd("box_tracking", payload)
        print(f"x={x:.2f}\ty={y:.2f}")


if __name__ == "__main__":
    typer.run(mocap)
