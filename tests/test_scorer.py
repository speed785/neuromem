from neuromem.scorer import MessageScorer


def test_score_messages_empty_input():
    scorer = MessageScorer()
    assert scorer.score_messages([]) == []


def test_system_message_gets_full_score():
    scorer = MessageScorer()
    scored = scorer.score_messages([
        {"role": "system", "content": "Always follow policy."},
        {"role": "user", "content": "hello"},
    ])
    assert scored[0].score == 1.0
    assert "system-override" in scored[0].reasons


def test_keyword_boost_increases_score():
    scorer = MessageScorer(recency_decay=0.0)
    scored = scorer.score_messages([
        {"role": "user", "content": "hello world"},
        {"role": "user", "content": "critical requirement must never fail"},
    ])
    assert scored[1].score > scored[0].score


def test_very_long_message_has_bounded_score_and_large_token_count():
    scorer = MessageScorer(recency_decay=0.0)
    long_text = "word " * 12000
    scored = scorer.score_messages([
        {"role": "user", "content": "short"},
        {"role": "assistant", "content": long_text},
    ])
    assert scored[1].token_count > 1000
    assert 0.0 <= scored[1].score <= 1.0


def test_relevance_prefers_text_similar_to_reference():
    scorer = MessageScorer(recency_decay=0.0)
    scored = scorer.score_messages(
        [
            {"role": "assistant", "content": "quantum state and qubits"},
            {"role": "assistant", "content": "cooking recipe and tomatoes"},
        ],
        reference_text="quantum qubits superposition",
    )
    assert scored[0].score > scored[1].score
