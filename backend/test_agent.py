import urllib.request
import json
import time

def test_query(message, thread_id):
    print("=" * 60)
    print(f"Query: {message}")
    print("=" * 60)
    
    payload = {
        "userId": "test-user",
        "threadId": thread_id,
        "message": message
    }
    
    req = urllib.request.Request(
        "http://localhost:8000/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}
    )
    
    start = time.time()
    try:
        response = urllib.request.urlopen(req, timeout=90)
        elapsed = time.time() - start
        data = json.loads(response.read().decode())
        
        print(f"Response Time: {elapsed:.2f}s")
        
        # Stylist response
        stylist = data.get("stylist_response", "")
        if stylist:
            print(f"\nStylist Response:\n{stylist[:500]}")
        
        # Products
        products = data.get("products", [])
        print(f"\nProducts Found: {len(products)}")
        for i, p in enumerate(products[:4]):
            title = p.get("title", "?")[:45]
            price = p.get("price", {})
            if isinstance(price, dict):
                price_val = price.get("current", "?")
            else:
                price_val = price
            brand = p.get("brand", "?")
            img = "Yes" if p.get("image_url") else "No"
            print(f"  {i+1}. {brand} - {title} | Rs {price_val} | Image: {img}")
        
        # Weather
        weather = data.get("user_profile", {})
        if data.get("clarification"):
            print(f"\nClarification: {data['clarification']}")
        
        return data
        
    except Exception as e:
        print(f"Error: {e}")
        return None

# Run tests
print("\n" + "="*60)
print("SORTME AI AGENT QUALITY TEST")
print("="*60 + "\n")

# Test 1: Specific product query
test_query("men's linen shirts", "test-1")

print("\n" + "-"*60 + "\n")

# Test 2: Broad travel query (should trigger weather)
test_query("what to wear for a beach trip to Goa next week", "test-2")

print("\n" + "-"*60 + "\n")

# Test 3: Occasion-based query
test_query("party outfit for women", "test-3")

print("\n\nALL TESTS COMPLETE")
