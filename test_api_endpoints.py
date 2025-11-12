#!/usr/bin/env python3
"""Comprehensive test script for Model Trainer API endpoints.

This script tests all API endpoints against a deployed instance to ensure
the system is fully functional end-to-end.
"""

import json
import sys
import time
from typing import Any

import requests

# Configuration
API_BASE_URL = ""  # Will be set from command line
API_KEY = ""  # Optional, set if API requires authentication
HEADERS = {"Content-Type": "application/json"}


def set_api_key(key: str) -> None:
    """Set API key for authenticated requests."""
    global HEADERS
    if key:
        HEADERS["X-API-Key"] = key


def print_test(name: str) -> None:
    """Print test header."""
    print(f"\n{'='*80}")
    print(f"TEST: {name}")
    print('='*80)


def print_result(success: bool, message: str, details: dict[str, Any] | None = None) -> None:
    """Print test result."""
    status = "[PASS]" if success else "[FAIL]"
    print(f"{status} {message}")
    if details:
        print(f"  Details: {json.dumps(details, indent=2)}")


def test_health_endpoints() -> tuple[bool, bool]:
    """Test health check endpoints."""
    print_test("Health Endpoints")

    # Test /healthz
    try:
        resp = requests.get(f"{API_BASE_URL}/healthz", timeout=10)
        healthz_ok = resp.status_code == 200
        data = resp.json()
        print_result(healthz_ok, "GET /healthz", {"status_code": resp.status_code, "response": data})
    except Exception as e:
        print_result(False, f"GET /healthz - Exception: {e}")
        healthz_ok = False

    # Test /readyz
    try:
        resp = requests.get(f"{API_BASE_URL}/readyz", timeout=10)
        readyz_ok = resp.status_code == 200
        data = resp.json()
        print_result(readyz_ok, "GET /readyz", {"status_code": resp.status_code, "response": data})
    except Exception as e:
        print_result(False, f"GET /readyz - Exception: {e}")
        readyz_ok = False

    return healthz_ok, readyz_ok


def test_tokenizer_training() -> str | None:
    """Test tokenizer training endpoint."""
    print_test("Tokenizer Training")

    payload = {
        "text": "Hello world! This is a test tokenizer training dataset. "
               "The quick brown fox jumps over the lazy dog. "
               "We need enough text to train a small tokenizer vocabulary.",
        "method": "bpe",
        "vocab_size": 100
    }

    try:
        resp = requests.post(
            f"{API_BASE_URL}/tokenizers/train",
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        success = resp.status_code == 200
        data = resp.json()
        tokenizer_id = data.get("tokenizer_id") if success else None
        print_result(
            success,
            "POST /tokenizers/train",
            {"status_code": resp.status_code, "tokenizer_id": tokenizer_id}
        )
        return tokenizer_id
    except Exception as e:
        print_result(False, f"POST /tokenizers/train - Exception: {e}")
        return None


def test_tokenizer_status(tokenizer_id: str, max_wait: int = 120) -> bool:
    """Test tokenizer status endpoint and wait for completion."""
    print_test(f"Tokenizer Status (ID: {tokenizer_id})")

    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            resp = requests.get(
                f"{API_BASE_URL}/tokenizers/{tokenizer_id}",
                headers=HEADERS,
                timeout=10
            )
            success = resp.status_code == 200
            data = resp.json()
            status = data.get("status", "unknown")

            print_result(
                success,
                f"GET /tokenizers/{tokenizer_id}",
                {"status_code": resp.status_code, "status": status, "artifact_path": data.get("artifact_path")}
            )

            if status == "completed":
                return True
            elif status in ["failed", "unknown"]:
                return False

            # Still processing, wait a bit
            time.sleep(5)
        except Exception as e:
            print_result(False, f"GET /tokenizers/{tokenizer_id} - Exception: {e}")
            return False

    print_result(False, f"Tokenizer training timed out after {max_wait}s")
    return False


def test_model_training() -> str | None:
    """Test model training endpoint."""
    print_test("Model Training")

    payload = {
        "model_family": "gpt2",
        "model_size": "test",
        "text": "Once upon a time in a land far away, there lived a brave knight. "
               "The knight embarked on many adventures and fought dragons. "
               "This is a simple training dataset for testing purposes. "
               * 10,  # Repeat to have enough tokens
        "tokenizer_id": "default",
        "steps": 5,
        "batch_size": 1
    }

    try:
        resp = requests.post(
            f"{API_BASE_URL}/runs/train",
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        success = resp.status_code == 200
        data = resp.json()
        run_id = data.get("run_id") if success else None
        print_result(
            success,
            "POST /runs/train",
            {"status_code": resp.status_code, "run_id": run_id, "job_id": data.get("job_id")}
        )
        return run_id
    except Exception as e:
        print_result(False, f"POST /runs/train - Exception: {e}")
        return None


def test_run_status(run_id: str, max_wait: int = 300) -> bool:
    """Test run status endpoint and wait for completion."""
    print_test(f"Run Status (ID: {run_id})")

    start_time = time.time()
    last_status = None

    while time.time() - start_time < max_wait:
        try:
            resp = requests.get(
                f"{API_BASE_URL}/runs/{run_id}",
                headers=HEADERS,
                timeout=10
            )
            success = resp.status_code == 200
            data = resp.json()
            status = data.get("status", "unknown")

            # Only print if status changed
            if status != last_status:
                print_result(
                    success,
                    f"GET /runs/{run_id}",
                    {
                        "status_code": resp.status_code,
                        "status": status,
                        "progress": data.get("progress"),
                        "artifact_path": data.get("artifact_path")
                    }
                )
                last_status = status

            if status == "completed":
                return True
            elif status in ["failed", "cancelled"]:
                return False

            # Still processing, wait a bit
            time.sleep(10)
        except Exception as e:
            print_result(False, f"GET /runs/{run_id} - Exception: {e}")
            return False

    print_result(False, f"Training run timed out after {max_wait}s")
    return False


def test_run_logs(run_id: str) -> bool:
    """Test run logs endpoint."""
    print_test(f"Run Logs (ID: {run_id})")

    try:
        resp = requests.get(
            f"{API_BASE_URL}/runs/{run_id}/logs",
            headers=HEADERS,
            params={"tail": 50},
            timeout=10
        )
        success = resp.status_code == 200
        log_lines = len(resp.text.split("\n")) if success else 0
        print_result(
            success,
            f"GET /runs/{run_id}/logs",
            {"status_code": resp.status_code, "log_lines": log_lines}
        )
        if success and log_lines > 0:
            print(f"  Sample logs (first 3 lines):")
            for line in resp.text.split("\n")[:3]:
                print(f"    {line}")
        return success
    except Exception as e:
        print_result(False, f"GET /runs/{run_id}/logs - Exception: {e}")
        return False


def test_run_evaluation(run_id: str) -> bool:
    """Test run evaluation endpoint."""
    print_test(f"Run Evaluation (ID: {run_id})")

    payload = {"split": "val"}

    try:
        resp = requests.post(
            f"{API_BASE_URL}/runs/{run_id}/evaluate",
            headers=HEADERS,
            json=payload,
            timeout=30
        )
        success = resp.status_code == 200
        data = resp.json() if success else {}
        print_result(
            success,
            f"POST /runs/{run_id}/evaluate",
            {"status_code": resp.status_code, "job_id": data.get("job_id")}
        )
        return success
    except Exception as e:
        print_result(False, f"POST /runs/{run_id}/evaluate - Exception: {e}")
        return False


def test_evaluation_results(run_id: str, max_wait: int = 120) -> bool:
    """Test evaluation results endpoint."""
    print_test(f"Evaluation Results (ID: {run_id})")

    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            resp = requests.get(
                f"{API_BASE_URL}/runs/{run_id}/eval",
                headers=HEADERS,
                timeout=10
            )
            success = resp.status_code == 200
            data = resp.json() if success else {}
            status = data.get("status", "unknown")

            print_result(
                success,
                f"GET /runs/{run_id}/eval",
                {
                    "status_code": resp.status_code,
                    "status": status,
                    "loss": data.get("loss"),
                    "perplexity": data.get("perplexity")
                }
            )

            if status == "completed":
                return True
            elif status in ["failed", "not_found"]:
                return False

            time.sleep(5)
        except Exception as e:
            print_result(False, f"GET /runs/{run_id}/eval - Exception: {e}")
            return False

    print_result(False, f"Evaluation timed out after {max_wait}s")
    return False


def test_artifacts(kind: str, item_id: str) -> bool:
    """Test artifact listing and download endpoints."""
    print_test(f"Artifacts ({kind}/{item_id})")

    # Test listing
    try:
        resp = requests.get(
            f"{API_BASE_URL}/artifacts/{kind}/{item_id}",
            headers=HEADERS,
            timeout=10
        )
        success = resp.status_code == 200
        data = resp.json() if success else {}
        files = data.get("files", [])
        print_result(
            success,
            f"GET /artifacts/{kind}/{item_id}",
            {"status_code": resp.status_code, "file_count": len(files)}
        )

        if not success or not files:
            return False

        # Test downloading first file
        first_file = files[0]
        resp2 = requests.get(
            f"{API_BASE_URL}/artifacts/{kind}/{item_id}/download",
            headers=HEADERS,
            params={"path": first_file},
            timeout=10
        )
        download_ok = resp2.status_code == 200
        size = len(resp2.content) if download_ok else 0
        print_result(
            download_ok,
            f"GET /artifacts/{kind}/{item_id}/download?path={first_file}",
            {"status_code": resp2.status_code, "size_bytes": size}
        )

        return success and download_ok
    except Exception as e:
        print_result(False, f"Artifact tests - Exception: {e}")
        return False


def main() -> int:
    """Run all tests."""
    global API_BASE_URL, API_KEY

    if len(sys.argv) < 2:
        print("Usage: python test_api_endpoints.py <API_BASE_URL> [API_KEY]")
        print("Example: python test_api_endpoints.py https://model-trainer.railway.app my-secret-key")
        return 1

    API_BASE_URL = sys.argv[1].rstrip("/")
    if len(sys.argv) >= 3:
        API_KEY = sys.argv[2]
        set_api_key(API_KEY)

    print(f"\n{'='*80}")
    print(f"Model Trainer API - Comprehensive Test Suite")
    print(f"{'='*80}")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"API Key: {'(set)' if API_KEY else '(not set)'}")
    print(f"{'='*80}\n")

    results: dict[str, bool] = {}

    # 1. Health checks
    healthz_ok, readyz_ok = test_health_endpoints()
    results["healthz"] = healthz_ok
    results["readyz"] = readyz_ok

    if not (healthz_ok and readyz_ok):
        print("\n[ERROR] Health checks failed. Cannot proceed with further tests.")
        return 1

    # 2. Tokenizer workflow
    tokenizer_id = test_tokenizer_training()
    results["tokenizer_training"] = tokenizer_id is not None

    if tokenizer_id:
        tokenizer_ok = test_tokenizer_status(tokenizer_id)
        results["tokenizer_status"] = tokenizer_ok

        if tokenizer_ok:
            results["tokenizer_artifacts"] = test_artifacts("tokenizers", tokenizer_id)

    # 3. Model training workflow
    run_id = test_model_training()
    results["model_training"] = run_id is not None

    if run_id:
        training_ok = test_run_status(run_id)
        results["run_status"] = training_ok

        logs_ok = test_run_logs(run_id)
        results["run_logs"] = logs_ok

        if training_ok:
            results["model_artifacts"] = test_artifacts("models", run_id)

            eval_ok = test_run_evaluation(run_id)
            results["run_evaluation"] = eval_ok

            if eval_ok:
                eval_results_ok = test_evaluation_results(run_id)
                results["evaluation_results"] = eval_results_ok

    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{status} {test_name}")

    print(f"\n{passed}/{total} tests passed ({100*passed//total}%)")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
