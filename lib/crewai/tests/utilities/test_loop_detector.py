"""Tests for LoopDetector middleware."""
import sys
import sys; sys.path.insert(0, 'lib/crewai/src')

from loop_detector import LoopDetector, ActionRecord


def test_no_loop_with_few_actions():
    """Should not detect a loop with fewer actions than max_repetitions."""
    detector = LoopDetector(max_repetitions=3)
    detector.record_action("search", "find files")
    detector.record_action("search", "find files")
    assert not detector.is_looping(), "Should not loop with only 2 actions"
    print("PASS: test_no_loop_with_few_actions")


def test_exact_repetition_detected():
    """Should detect when the same action is repeated N times."""
    detector = LoopDetector(max_repetitions=3)
    for _ in range(3):
        detector.record_action("search", "find /tmp -name '*.py'")
    assert detector.is_looping(), "Should detect exact repetition"
    print("PASS: test_exact_repetition_detected")


def test_similarity_loop_detected():
    """Should detect when actions are suspiciously similar."""
    detector = LoopDetector(max_repetitions=3, similarity_threshold=0.8)
    detector.record_action("search", "find /tmp/crewAI -name '*.py' -type f")
    detector.record_action("search", "find /tmp/crewAI -name '*.py' -type d")
    detector.record_action("search", "find /tmp/crewAI -name '*.py' -type l")
    assert detector.is_looping(), "Should detect similarity loop"
    print("PASS: test_similarity_loop_detected")


def test_no_false_positive_different_actions():
    """Should not trigger on genuinely different actions."""
    detector = LoopDetector(max_repetitions=3)
    detector.record_action("search", "find files in directory")
    detector.record_action("write", "create new configuration file")
    detector.record_action("deploy", "push changes to production")
    assert not detector.is_looping(), "Different actions should not trigger loop"
    print("PASS: test_no_false_positive_different_actions")


def test_cyclic_pattern_detected():
    """Should detect A->B->A->B cyclic patterns."""
    detector = LoopDetector(window_size=10)
    actions = [("clone", "git clone repo"), ("explore", "ls -la"), 
               ("clone", "git clone repo"), ("explore", "ls -la")]
    for action_type, content in actions:
        detector.record_action(action_type, content)
    assert detector.is_looping(), "Should detect cyclic A->B->A->B pattern"
    print("PASS: test_cyclic_pattern_detected")


def test_exit_strategy_exact():
    """Should suggest force_different_action for exact repetitions."""
    detector = LoopDetector(max_repetitions=3)
    for _ in range(3):
        detector.record_action("search", "same thing over and over")
    detector.is_looping()  # trigger detection
    strategy = detector.suggest_exit_strategy()
    assert strategy["strategy"] in ("force_different_action", "escalate"), \
        f"Expected force_different_action, got {strategy['strategy']}"
    print("PASS: test_exit_strategy_exact")


def test_exit_strategy_cyclic():
    """Should suggest break_cycle for cyclic patterns."""
    detector = LoopDetector(window_size=10, similarity_threshold=0.99)
    # Use very different content so similarity doesn't trigger
    actions = [("clone", "aaaa"), ("explore", "zzzz"),
               ("clone", "aaaa"), ("explore", "zzzz")]
    for action_type, content in actions:
        detector.record_action(action_type, content)
    detector.is_looping()
    strategy = detector.suggest_exit_strategy()
    assert strategy["strategy"] in ("break_cycle", "escalate"), \
        f"Expected break_cycle or escalate, got {strategy['strategy']}"
    print("PASS: test_exit_strategy_cyclic")


def test_stats():
    """Should track statistics correctly."""
    detector = LoopDetector(max_repetitions=2)
    detector.record_action("a", "content1")
    detector.record_action("a", "content1")
    detector.is_looping()
    stats = detector.get_stats()
    assert stats["total_actions"] == 2
    assert stats["loops_detected"] >= 1
    print("PASS: test_stats")


def test_reset():
    """Should clear all state on reset."""
    detector = LoopDetector(max_repetitions=2)
    detector.record_action("a", "same")
    detector.record_action("a", "same")
    detector.is_looping()
    detector.reset()
    stats = detector.get_stats()
    assert stats["total_actions"] == 0
    assert stats["loops_detected"] == 0
    print("PASS: test_reset")


def test_callback_on_loop():
    """Should call the callback when a loop is detected."""
    events = []
    detector = LoopDetector(max_repetitions=2, on_loop_detected=lambda e: events.append(e))
    detector.record_action("search", "same query")
    detector.record_action("search", "same query")
    detector.is_looping()
    assert len(events) == 1, f"Expected 1 callback, got {len(events)}"
    assert "exact" in events[0]["pattern_type"]
    print("PASS: test_callback_on_loop")


if __name__ == "__main__":
    tests = [
        test_no_loop_with_few_actions,
        test_exact_repetition_detected,
        test_similarity_loop_detected,
        test_no_false_positive_different_actions,
        test_cyclic_pattern_detected,
        test_exit_strategy_exact,
        test_exit_strategy_cyclic,
        test_stats,
        test_reset,
        test_callback_on_loop,
    ]
    
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {test.__name__}: {e}")
            failed += 1
    
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
        sys.exit(1)
