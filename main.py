
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from inference import SentimentPredictor


# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the service starts."""
    logger.info("Starting Twitter Sentiment Analysis API")
    logger.info("Loading sentiment model from artifacts directory")

    try:
        app.state.predictor = SentimentPredictor(
            artifact_dir="artifacts"
        )
        logger.info("Sentiment model loaded successfully")
    except Exception:
        logger.exception("Failed to load sentiment model")
        raise

    yield

    logger.info("Shutting down Twitter Sentiment Analysis API")


app = FastAPI(
    title="Twitter Sentiment Analysis API",
    description="Sentiment classification using a PyTorch GRU.",
    version="1.0.0",
    lifespan=lifespan,
)


class TweetRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=5000,
        description="The tweet text to classify.",
    )


class BatchTweetRequest(BaseModel):
    texts: list[str] = Field(
        min_length=1,
        max_length=100,
        description="Between 1 and 100 tweets.",
    )


def get_predictor(request: Request) -> SentimentPredictor:
    return request.app.state.predictor


@app.get("/health")
def health(request: Request):
    predictor = get_predictor(request)

    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
    }


@app.get("/model-info")
def model_info(request: Request):
    predictor = get_predictor(request)

    return {
        "model": "TwitterGRU",
        "framework": "PyTorch",
        "task": "binary sentiment classification",
        "labels": ["negative", "positive"],
        "vocabulary_size": len(predictor.vocab),
        "max_sequence_length": predictor.max_length,
    }



@app.post("/predict")
def predict(payload: TweetRequest, request: Request):
    if not payload.text.strip():
        logger.warning("Rejected prediction request: empty tweet")
        raise HTTPException(
            status_code=422,
            detail="Tweet text cannot contain only whitespace.",
        )

    predictor = get_predictor(request)
    start_time = time.perf_counter()

    try:
        result = predictor.predict(payload.text)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Prediction completed | latency_ms=%.2f | sentiment=%s",
            elapsed_ms,
            result["sentiment"],
        )

        return result

    except ValueError as error:
        logger.warning("Prediction rejected: %s", error)
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during prediction.",
        ) from None
    except Exception:
        logger.exception("Unexpected error during batch prediction")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred during batch prediction.",
        ) from None


@app.post("/predict/batch")
def predict_batch(
    payload: BatchTweetRequest,
    request: Request,
):
    if any(not text.strip() for text in payload.texts):
        logger.warning(
            "Rejected batch prediction: one or more empty tweets"
        )
        raise HTTPException(
            status_code=422,
            detail="Every tweet must contain non-whitespace text.",
        )

    predictor = get_predictor(request)
    start_time = time.perf_counter()

    try:
        results = predictor.predict_batch(payload.texts)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Batch prediction completed | batch_size=%d | latency_ms=%.2f",
            len(payload.texts),
            elapsed_ms,
        )

        return {
            "count": len(results),
            "predictions": results,
        }

    except ValueError as error:
        logger.warning("Batch prediction rejected: %s", error)
        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error
