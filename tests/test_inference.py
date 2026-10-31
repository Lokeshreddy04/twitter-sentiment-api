import pytest

from inference import normalize_tweet, tokenize_tweet, SentimentPredictor


@pytest.fixture(scope="module")
def predictor():
    return SentimentPredictor(artifact_dir="artifacts")


def test_normalize_tweet_lowercases_text():
    assert normalize_tweet("I LOVE THIS") == "i love this"


def test_normalize_tweet_replaces_url():
    result = normalize_tweet("Visit https://example.com now")
    assert "URL" in result


def test_normalize_tweet_replaces_mention():
    result = normalize_tweet("Hello @someone")
    assert "USER" in result


def test_normalize_tweet_preserves_hashtag_word():
    result = normalize_tweet("This is #Amazing")
    assert "amazing" in result
    assert "#" not in result


def test_tokenizer_preserves_punctuation():
    tokens = tokenize_tweet("Amazing!!!")
    assert "!!!" in tokens


def test_encoding_returns_valid_ids(predictor):
    ids = predictor.encode("I love this product!")

    assert len(ids) > 0
    assert len(ids) <= predictor.max_length
    assert all(0 <= token_id < len(predictor.vocab) for token_id in ids)


def test_unknown_words_use_unk_id(predictor):
    # Use a deliberately improbable token
    ids = predictor.encode("qzxvplmokn123456")
    assert predictor.unk_id in ids


def test_empty_input_is_rejected(predictor):
    with pytest.raises(ValueError):
        predictor.predict("   ")


def test_batch_predictions_match_single_predictions(predictor):
    texts = [
        "I love this!",
        "This is awful.",
        "It is okay.",
    ]

    batch_results = predictor.predict_batch(texts)
    single_results = [predictor.predict(text) for text in texts]

    assert len(batch_results) == len(single_results)

    for batch_result, single_result in zip(batch_results, single_results):
        assert batch_result["sentiment"] == single_result["sentiment"]
        assert abs(
            batch_result["positive_score"]
            - single_result["positive_score"]
        ) < 1e-5