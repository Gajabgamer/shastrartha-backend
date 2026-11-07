#!/usr/bin/env python3
"""Test deployed backend health and endpoints."""

import sys
from urllib.request import urlopen, Request
from urllib.error import URLError
import json

def test_endpoint(url: str, name: str) -> bool:
    """Test a single endpoint."""
    try:
        req = Request(url, headers={"User-Agent": "ShastrarthaTest/1.0"})
        with urlopen(req, timeout=10) as response:
            data = json.loads(response.read())
            print(f"  ✅ {name}")
            return True
    except URLError as e:
        print(f"  ❌ {name}: {e}")
        return False
    except Exception as e:
        print(f"  ⚠️  {name}: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_deployment.py <backend-url>")
        print("Example: python test_deployment.py https://shastrartha-backend.onrender.com")
        sys.exit(1)
    
    base_url = sys.argv[1].rstrip("/")
    print(f"🧪 Testing deployment: {base_url}\n")
    
    tests = [
        (f"{base_url}/api/health", "Health check"),
        (f"{base_url}/api/personas", "List personas"),
        (f"{base_url}/api/topics", "List topics"),
        (f"{base_url}/api/regions", "List regions"),
        (f"{base_url}/api/glossary", "Glossary"),
        (f"{base_url}/api/today?scope=india", "Today's desk"),
    ]
    
    passed = 0
    failed = 0
    
    for url, name in tests:
        if test_endpoint(url, name):
            passed += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    if failed == 0:
        print(f"✅ All {passed} tests passed!")
        print("\n🎉 Backend is live and healthy!")
        print(f"\nFrontend setup:")
        print(f"  cd frontend")
        print(f"  npx vercel env add NEXT_PUBLIC_API_URL production")
        print(f"  # Enter: {base_url}")
        print(f"  npx vercel --prod")
    else:
        print(f"⚠️  {passed} passed, {failed} failed")
        print("\nTroubleshooting:")
        print("  1. Wait 30-90s for cold start wake-up")
        print("  2. Check Render logs for errors")
        print("  3. Verify environment variables are set")
        print("  4. Ensure build completed successfully")
        sys.exit(1)


if __name__ == "__main__":
    main()
