from fastapi.testclient import TestClient

from main import app


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")

        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


def test_model_info_endpoint():
    with TestClient(app) as client:
        response = client.get("/model-info")

        assert response.status_code == 200

        data = response.json()
        assert data["model"] == "TwitterGRU"
        assert data["framework"] == "PyTorch"
        assert data["vocabulary_size"] > 0
        assert data["max_sequence_length"] == 40


def test_predict_endpoint():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "I absolutely love this product!"},
        )

        assert response.status_code == 200

        data = response.json()
        assert data["sentiment"] in {"positive", "negative"}
        assert data["text"] == "I absolutely love this product!"
        assert 0 <= data["positive_score"] <= 1
        assert 0 <= data["negative_score"] <= 1


def test_predict_rejects_empty_text():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "   "},
        )

        assert response.status_code == 422


def test_predict_rejects_missing_text():
    with TestClient(app) as client:
        response = client.post("/predict", json={})

        assert response.status_code == 422


def test_batch_prediction():
    tweets = [
        "I love this!",
        "This is awful.",
        "It is okay.",
    ]

    with TestClient(app) as client:
        response = client.post(
            "/predict/batch",
            json={"texts": tweets},
        )

        assert response.status_code == 200

        data = response.json()
        assert data["count"] == len(tweets)
        assert len(data["predictions"]) == len(tweets)


def test_batch_rejects_whitespace_tweet():
    with TestClient(app) as client:
        response = client.post(
            "/predict/batch",
            json={"texts": ["I like this", "   "]},
        )

        assert response.status_code == 422