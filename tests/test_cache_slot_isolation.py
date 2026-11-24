def test_semantic_cache_returns_stale_dynamic_arguments():
    """
    Simulates a semantic cache hit where the prompt structure is identical,
    but the dynamic ID changes.
    
    In a semantic cache, 'Refund charge ch_111' and 'Refund charge ch_222'
    score ~0.97 cosine similarity. Without slot isolation, the cache returns
    the stale arguments for ch_111.
    """
    
    # 1. Simulate the cached response for a previous request
    cached_tool_call = {
        "tool": "stripe_refund",
        "arguments": {"amount_cents": 4500, "charge_id": "ch_99102000112200"}
    }
    
    # 2. The current request has a new dynamic ID, but hits the cache
    current_request_expected = {
        "tool": "stripe_refund",
        "arguments": {"amount_cents": 4500, "charge_id": "ch_3MtwBwLkdI9999"}
    }
    
    # 3. The cache returns the old arguments verbatim
    retrieved_call = cached_tool_call 
    
    # 4. This assertion proves the data leak/stale argument bug
    assert retrieved_call["arguments"]["charge_id"] == current_request_expected["arguments"]["charge_id"], \
        f"Stale arguments returned! Expected {current_request_expected['arguments']['charge_id']} but got {retrieved_call['arguments']['charge_id']}"
