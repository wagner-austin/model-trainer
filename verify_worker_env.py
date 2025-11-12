#!/usr/bin/env python3
"""Quick script to verify worker environment and Redis connectivity."""

import os
import sys


def main() -> int:
    print("=" * 80)
    print("Worker Environment Verification")
    print("=" * 80)

    # Check REDIS_URL
    redis_url = os.getenv("REDIS_URL", "")
    if redis_url:
        # Mask password
        if "@" in redis_url:
            parts = redis_url.split("@")
            user_pass = parts[0].split("//")[1]
            if ":" in user_pass:
                masked = redis_url.replace(user_pass.split(":")[1], "***")
            else:
                masked = redis_url
        else:
            masked = redis_url
        print(f"\n[OK] REDIS_URL is set: {masked}")
    else:
        print("\n[FAIL] REDIS_URL is NOT set!")
        print("  This is likely why the worker isn't registering in Redis.")
        return 1

    # Try to connect
    print("\nAttempting to connect to Redis...")
    try:
        import redis

        client = redis.from_url(redis_url, decode_responses=False)
        pong = client.ping()
        if pong:
            print("[OK] Successfully connected to Redis!")
            print(f"     Redis server version: {client.info('server').get('redis_version', 'unknown')}")

            # Try to see if we can see other keys
            all_keys = client.keys("*")
            print(f"     Total keys in Redis: {len(all_keys)}")

            rq_keys = client.keys("rq:*")
            print(f"     RQ-related keys: {len(rq_keys)}")

            return 0
        else:
            print("[FAIL] Redis ping returned False")
            return 1

    except ImportError:
        print("[WARNING] redis module not available (expected if running outside venv)")
        return 0
    except Exception as e:
        print(f"[FAIL] Failed to connect: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
