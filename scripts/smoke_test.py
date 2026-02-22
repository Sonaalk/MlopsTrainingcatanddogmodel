import requests
import sys

BASE_URL = "http://localhost:30007"

def fail(msg):
    print(f"SMOKE TEST FAILED: {msg}")
    sys.exit(1)

# Health check
try:
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    if r.status_code != 200:
        fail("Health endpoint returned non-200")
    print("Health check passed")
except Exception as e:
    fail(str(e))

# Prediction check (no real image, just endpoint availability)
try:
    r = requests.post(f"{BASE_URL}/predict")
    if r.status_code not in [200, 422]:
        fail("Predict endpoint not reachable")
    print("Predict endpoint reachable")
except Exception as e:
    fail(str(e))

print("ALL SMOKE TESTS PASSED")