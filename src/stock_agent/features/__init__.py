"""Feature engineering: price-derived (point-in-time) and news-derived (Claude sentiment)."""

from stock_agent.features.assembler import (
    THRESHOLDS,
    build_training_matrix,
    current_features_vector,
)
from stock_agent.features.news_features import NEWS_FEATURE_COLS, build_news_features
from stock_agent.features.price_features import (
    PRICE_FEATURE_COLS,
    build_price_feature_matrix,
    current_price_features,
)

__all__ = [
    "PRICE_FEATURE_COLS",
    "NEWS_FEATURE_COLS",
    "THRESHOLDS",
    "build_price_feature_matrix",
    "current_price_features",
    "build_news_features",
    "build_training_matrix",
    "current_features_vector",
]
