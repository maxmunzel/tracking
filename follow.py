from relative import main
import redis

redis_connection = redis.Redis()

for frame in main():
    marker_id, transform = frame
    # if marker_id != "2":
    #     continue
    marker_id = int(marker_id)
    x, y, _ = transform[:3, 3].flatten()
    x += 0.13
    payload = {"x": x, "y": y}
    redis_connection.xadd("cart_cmd", payload)
    print(f"x={x:.2f}\ty={y:.2f}")
