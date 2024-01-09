from relative import main
import redis
import json

redis_connection = redis.Redis()

for frame in main():
    marker_id, transform = frame
    marker_id = int(marker_id)
    x, y = transform[:2, 3].flatten()
    transform = list(transform.flatten())
    payload = {"transform": json.dumps(transform)}
    redis_connection.xadd("box_tracking", payload)
    print(f"x={x:.2f}\ty={y:.2f}")
