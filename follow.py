from relative import main
import redis

redis_connection = redis.Redis()

for frame in main():
    marker_id, transform = frame
    # if marker_id != "2":
    #     continue
    marker_id = int(marker_id)
    x, y = transform[:2, 3].flatten()
    payload = {"x": x, "y": y, "cmd": "GOTO"}
    redis_connection.xadd("cart_cmd", payload)
    print(f"x={x:.2f}\ty={y:.2f}")
