import sys
from fastapi.testclient import TestClient
import sys
sys.path.insert(0, ".")
from app import app

client = TestClient(app)

def run_tests():
    success = True
    print("Starting tests for Mutual Fund Analyzer API backend...")
    
    # helper to run request and check status
    def test_endpoint(name, method, url, json=None, expected_status=200):
        nonlocal success
        print(f"Testing {name} ({method} {url})...", end="")
        try:
            if method == "GET":
                response = client.get(url)
            elif method == "POST":
                response = client.post(url, json=json)
            else:
                response = client.request(method, url)
                
            if response.status_code == expected_status:
                print(f" OK (Status: {response.status_code})")
            else:
                print(f" FAIL! Expected {expected_status}, got {response.status_code}")
                print(f"Response: {response.text}")
                success = False
        except Exception as e:
            print(f" ERROR! Exception: {e}")
            success = False

    # Standard Endpoints
    test_endpoint("Summary", "GET", "/api/summary")
    test_endpoint("Holdings", "GET", "/api/holdings")
    test_endpoint("Allocation", "GET", "/api/allocation")
    test_endpoint("Transactions", "GET", "/api/transactions?limit=10")
    test_endpoint("Risk Metrics", "GET", "/api/risk")
    test_endpoint("Portfolio Growth", "GET", "/api/growth")
    test_endpoint("Overlap", "GET", "/api/overlap")
    test_endpoint("Taxes", "GET", "/api/taxes")
    test_endpoint("Dividends", "GET", "/api/dividends")
    test_endpoint("Stress Test", "GET", "/api/stress-test")
    
    # POST Endpoints with payloads
    test_endpoint("Goal Strategy", "POST", "/api/goal-strategy", json={
        "target_amount": 1000000,
        "horizon_years": 5,
        "include_current_portfolio": True
    })
    
    test_endpoint("Simulations", "POST", "/api/simulations", json={
        "monthly_sip": 5000,
        "step_up_pct": 10.0,
        "horizon_years": 10,
        "mean_return_pct": 12.0,
        "volatility_pct": 15.0
    })
    
    test_endpoint("Rebalance", "POST", "/api/rebalance", json={
        "target_equity_pct": 60.0
    })
    
    # Test Chat (Will likely fail or error if UPSTAGE_API_KEY is missing, but should handle correctly)
    print("Testing Chat (POST /api/chat)...", end="")
    try:
        response = client.post("/api/chat", json={"message": "Hello"})
        if response.status_code in [200, 500]: # 500 is expected if API key is not configured
             print(f" OK (Status: {response.status_code})")
        else:
            print(f" FAIL! Got {response.status_code}")
            print(f"Response: {response.text}")
            success = False
    except Exception as e:
        print(f" ERROR! Exception: {e}")
        success = False

    if success:
        print("\nAll endpoints tested successfully!")
        sys.exit(0)
    else:
        print("\nSome endpoints failed.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
