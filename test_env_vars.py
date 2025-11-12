#!/usr/bin/env python3
"""Test which environment variable Pydantic Settings reads for redis.url"""

import os
import sys
sys.path.insert(0, "server")

# Test with REDIS_URL
os.environ["REDIS_URL"] = "redis://test-single-underscore:6379"
from model_trainer.core.config.settings import Settings

settings1 = Settings()
print(f"With REDIS_URL set: {settings1.redis.url}")

# Test with REDIS__URL
os.environ["REDIS__URL"] = "redis://test-double-underscore:6379"
settings2 = Settings()
print(f"With REDIS__URL set: {settings2.redis.url}")

# Clean up and test which one wins
del os.environ["REDIS_URL"]
settings3 = Settings()
print(f"With only REDIS__URL: {settings3.redis.url}")
