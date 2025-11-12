#!/usr/bin/env python3
"""Diagnostic script to check Redis connection and RQ worker registration."""

import redis
import sys

REDIS_URL = "redis://default:iZpSAwxbAtgOibJIFfNXyoTiGDrCempA@trolley.proxy.rlwy.net:42826"


def main() -> int:
    print(f"Connecting to Redis at: {REDIS_URL.replace('iZpSAwxbAtgOibJIFfNXyoTiGDrCempA', '***')}")
    print("-" * 80)

    try:
        # Connect with decode_responses=True for easier reading
        client = redis.from_url(REDIS_URL, decode_responses=True)

        # Test connection
        print("\n1. Testing Redis connection...")
        pong = client.ping()
        print(f"   [OK] Redis PING: {pong}")

        # Check rq:workers set
        print("\n2. Checking rq:workers set...")
        workers_count = client.scard("rq:workers")
        print(f"   Workers count: {workers_count}")

        if workers_count > 0:
            workers = client.smembers("rq:workers")
            print(f"   Registered workers:")
            for w in workers:
                print(f"     - {w}")
        else:
            print("   [WARNING] No workers found in rq:workers set!")

        # List all RQ-related keys
        print("\n3. All RQ-related keys in Redis...")
        rq_keys = client.keys("rq:*")
        print(f"   Found {len(rq_keys)} RQ keys:")
        for key in sorted(rq_keys):
            key_type = client.type(key)
            if key_type == "set":
                count = client.scard(key)
                print(f"     - {key} (set, {count} members)")
            elif key_type == "list":
                count = client.llen(key)
                print(f"     - {key} (list, {count} items)")
            elif key_type == "zset":
                count = client.zcard(key)
                print(f"     - {key} (zset, {count} items)")
            elif key_type == "hash":
                count = client.hlen(key)
                print(f"     - {key} (hash, {count} fields)")
            else:
                print(f"     - {key} ({key_type})")

        # Check training queue specifically
        print("\n4. Checking training queue...")
        training_queue_len = client.llen("rq:queue:training")
        print(f"   Training queue length: {training_queue_len}")

        # Redis info
        print("\n5. Redis Server Info...")
        info = client.info("server")
        print(f"   Redis version: {info.get('redis_version', 'unknown')}")
        print(f"   Uptime (seconds): {info.get('uptime_in_seconds', 'unknown')}")

        print("\n" + "=" * 80)
        print("Diagnosis Summary:")
        print("=" * 80)

        if workers_count > 0:
            print("[OK] Redis connection: OK")
            print(f"[OK] Workers registered: {workers_count}")
            print("\nYour setup looks healthy!")
        else:
            print("[OK] Redis connection: OK")
            print("[FAIL] Workers registered: 0")
            print("\n[WARNING] ISSUE IDENTIFIED:")
            print("  The worker is not registering itself in the rq:workers set.")
            print("\n  Possible causes:")
            print("  1. Worker service doesn't have REDIS_URL environment variable set")
            print("  2. Worker is connecting to a different Redis instance")
            print("  3. Worker failed to start (check worker deployment logs)")
            print("  4. Network connectivity issue between worker and Redis")

        return 0

    except redis.exceptions.ConnectionError as e:
        print(f"\n[FAIL] Failed to connect to Redis: {e}")
        return 1
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
