import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import re
import gc
import orjson
import logging
from sklearn.preprocessing import MinMaxScaler
from typing import Set, Optional
from pathlib import Path

import config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class YelpRestaurantPipeline:
    """
    Pipeline for processing Yelp business and review data to rank restaurants.
    """

    def __init__(self):
        self.business_df: Optional[pd.DataFrame] = None
        self.reviews_df: Optional[pd.DataFrame] = None
        self.final_df: Optional[pd.DataFrame] = None
        self.top_reviews_df: Optional[pd.DataFrame] = None

        self.rest_id_set: Set[str] = set()

    @staticmethod
    def _clean_city_names(city: str) -> str:
        """
        Applies corrections to city names based on config.

        Args:
            city (str): The original city name to be cleaned.

        Returns:
            str: The cleaned and lowercased city name.
        """
        if not city:
            return ""

        if city in config.US_CITIES_CORRECTIONS:
            return config.US_CITIES_CORRECTIONS[city].lower()

        for wrong, correct in config.US_CITIES_CORRECTIONS.items():
            if city.startswith(wrong):
                return city.replace(wrong, correct).lower()
        return city.lower()

    @staticmethod
    def _get_dynamic_n(total_restaurants: int) -> int:
        """
        Determines the limit of restaurants based on city density.

        Args:
            total_restaurants (int): The total number of restaurants in a city.

        Returns:
            int: The dynamic limit of top restaurants to return.
        """
        if total_restaurants < 20:
            return 20
        elif total_restaurants < 100:
            return 40
        elif total_restaurants < 200:
            return 80
        elif total_restaurants < 600:
            return 120
        else:
            return config.TOP_RESTAURANTS_PER_CITY

    @staticmethod
    def _get_sentiment(stars: float) -> str:
        """
        Determines the sentiment of a restaurant based on its star rating.

        Args:
            stars (float): The star rating of the restaurant.

        Returns:
            str: The sentiment label ("positive", "neutral", or "negative").
        """
        if stars >= 4:
            return "positive"
        elif stars == 3:
            return "neutral"
        return "negative"

    def _load_and_process_business(self):
        """
        Loads business data, filters for restaurants, and cleans city names.
        """
        logger.info("Processing Businesses...")

        data = []
        path = Path(config.BUSINESS_PATH)

        if not path.exists():
            raise FileNotFoundError(f"Business file not found at {path}")

        n_corrupt = 0
        with open(path, "rb") as f:
            for line in f:
                try:
                    data.append(orjson.loads(line))
                except orjson.JSONDecodeError:
                    n_corrupt += 1

        if n_corrupt:
            logger.warning(f"Skipped {n_corrupt} corrupted/unparseable business records.")

        df = pd.DataFrame(data)

        # Clean categories
        df = df.dropna(subset=["categories"])

        # Filter for "Restaurants" using regex vectorization logic
        rest_pattern = re.compile(r"\bRestaurants\b", re.IGNORECASE)
        df["is_restaurant"] = df["categories"].apply(
            lambda x: bool(rest_pattern.search(x))
        )

        df_restaurants = df[df["is_restaurant"]].copy()

        # Calculate initial score
        df_restaurants["restaurant_score"] = df_restaurants["stars"] * np.log1p(
            df_restaurants["review_count"]
        )

        # Clean City Names
        df_restaurants["city_clean"] = df_restaurants["city"].str.strip().str.title()
        df_restaurants["city"] = df_restaurants["city_clean"].apply(
            self._clean_city_names
        )

        # Dynamic Limits
        city_counts = df_restaurants["city_clean"].value_counts()
        df_restaurants["city_total"] = df_restaurants["city_clean"].map(city_counts)
        df_restaurants["dynamic_limit"] = df_restaurants["city_total"].apply(
            self._get_dynamic_n
        )

        # Cleanup
        self.business_df = df_restaurants.reset_index(drop=True)

        self.rest_id_set = set(self.business_df["business_id"].astype(str).str.strip())

        logger.info(f"Filtered down to {len(self.rest_id_set)} restaurants.")

        del data, df, city_counts
        gc.collect()

    def _stream_and_filter_reviews(self):
        """
        Streams reviews and filters strictly by Date and Restaurant ID
        BEFORE appending to memory to save RAM.
        """
        logger.info("Streaming and Filtering Reviews...")

        path = Path(config.REVIEW_PATH)
        if not path.exists():
            raise FileNotFoundError(f"Review file not found at {path}")

        reviews_dict = []

        min_date_str = f"{config.FILTER_YEAR}-01-01"

        n_corrupt = 0
        with open(path, "rb") as f:

            for line in tqdm(f, desc="Reading Reviews", unit="lines", smoothing=0):
                try:
                    review = orjson.loads(line)

                    # 1. Check ID
                    if review["business_id"] not in self.rest_id_set:
                        continue

                    # 2. Check Date
                    if review["date"] < min_date_str:
                        continue

                    # 3. Check Word Count (Slowest check)
                    if len(review["text"].split()) < config.MIN_REVIEW_WORDS:
                        continue

                    reviews_dict.append(
                        {
                            "business_id": review["business_id"],
                            "date": review["date"],
                            "text": review["text"],
                            "useful": review["useful"],
                            "stars": review["stars"],
                        }
                    )

                except orjson.JSONDecodeError:
                    n_corrupt += 1

        if n_corrupt:
            logger.warning(f"Skipped {n_corrupt} corrupted/unparseable review records.")

        self.reviews_df = pd.DataFrame(reviews_dict)
        logger.info(f"Loaded {len(self.reviews_df)} relevant reviews.")

    def _score_and_rank(self):
        """Scores restaurants based on weighted metrics and ranks them per city."""
        logger.info("Scoring and Ranking Restaurants...")

        # Aggregate review stats
        stat_df = (
            self.reviews_df.groupby("business_id")["stars"]
            .agg(["sum", "mean", "count"])
            .rename(columns={"count": "filtered_review_count"})
            .reset_index()
        )
        stat_df["reviews_score"] = stat_df["mean"] * np.log1p(stat_df["sum"])

        # Merge
        top_merged = pd.merge(self.business_df, stat_df, on="business_id", how="left")
        top_merged["filtered_review_count"] = top_merged["filtered_review_count"].fillna(0)

        # Normalize
        scaler = MinMaxScaler()
        if not top_merged["restaurant_score"].empty:
            top_merged["restaurant_score_norm"] = scaler.fit_transform(
                top_merged[["restaurant_score"]].fillna(0)
            )

        if not top_merged["reviews_score"].empty:
            top_merged["reviews_score_norm"] = scaler.fit_transform(
                top_merged[["reviews_score"]].fillna(0)
            )

        # Weighted Score
        top_merged["weighted_combined_score"] = (
            config.RESTAURANTS_COEFF * top_merged["restaurant_score_norm"]
            + config.REVIEWS_COEFF * top_merged["reviews_score_norm"]
        )

        # Filter by minimum review count. review_count is Yelp's raw, all-time
        # count; filtered_review_count is how many reviews actually survived the
        # date/word-count filters upstream — a restaurant needs enough of both,
        # otherwise it gets selected with no review content to chunk.
        base = top_merged[
            (top_merged["review_count"] > config.REVIEW_COUNTS_PER_RESTAURANT)
            & (top_merged["filtered_review_count"] >= config.MIN_FILTERED_REVIEWS)
        ]

        # Sort
        top_merged_sorted = base.sort_values(
            ["city", "weighted_combined_score"], ascending=[True, False]
        )

        # Top N per city
        def get_top_n(group):
            """
            Extracts the top N restaurants for a given city group.

            Args:
                group (pd.DataFrame): Dataframe partition containing a single city's restaurants.

            Returns:
                pd.DataFrame: Top N performing restaurants for the given city.
            """
            n = int(group["dynamic_limit"].iloc[0])
            top = group.nlargest(n, "weighted_combined_score").copy()
            top["city"] = group.name
            return top

        self.final_df = top_merged_sorted.groupby("city", group_keys=False).apply(
            get_top_n, include_groups=False
        )

        # Add stats back to final df for clarity
        self.final_df = pd.merge(
            self.final_df, stat_df, on="business_id", how="left", suffixes=("", "_stat")
        )
        logger.info(f"Final dataset contains {len(self.final_df)} restaurants.")

    def _select_top_reviews(self):
        """Selects representative reviews for the final restaurant list."""
        logger.info("Selecting Top Reviews...")

        target_ids = set(self.final_df["business_id"])

        # Filter reviews for only the top restaurants
        filtered_reviews = self.reviews_df[
            self.reviews_df["business_id"].isin(target_ids)
        ].copy()

        # Sort by usefulness
        filtered_reviews = filtered_reviews.sort_values(
            ["business_id", "useful"], ascending=[True, False]
        )

        # Cap candidates per business
        candidates = filtered_reviews.groupby("business_id").head(
            config.TOP_RESTAURANTS_PER_CITY
        ).copy()
        candidates["sentiment"] = candidates["stars"].apply(self._get_sentiment)

        # Sampling Logic
        def sample_balanced_reviews(group):
            """
            Samples a balanced subset of reviews across different sentiments.

            Args:
                group (pd.DataFrame): Dataframe partition containing reviews for a single business.

            Returns:
                pd.DataFrame: A sampled dataframe of balanced reviews by sentiment.
            """
            sentiment_counts = group["sentiment"].value_counts(normalize=True)
            sampled = []
            for sentiment in ["positive", "neutral", "negative"]:
                if sentiment in sentiment_counts:
                    n = int(
                        np.ceil(
                            sentiment_counts[sentiment]
                            * config.MAX_REVIEWS_PER_RESTAURANT
                        )
                    )
                    # Safe sample
                    avail = group[group["sentiment"] == sentiment]
                    sampled.append(avail.sample(n=min(n, len(avail)), random_state=42))
            out = pd.concat(sampled) if sampled else pd.DataFrame()
            if not out.empty:
                out["business_id"] = group.name
            return out

        self.top_reviews_df = candidates.groupby("business_id", group_keys=False).apply(
            sample_balanced_reviews, include_groups=False
        )

    def _build_final_output(self):
        """Constructs the dictionary structure and saves to pickle."""
        logger.info("Building Final JSON Structure and Saving...")

        # 1. Group by ID and Sentiment, collect texts
        agg_reviews = (
            self.top_reviews_df.groupby(["business_id", "sentiment"])["text"]
            .apply(list)
            .reset_index()
        )

        # 2. Pivot to make sentiments columns
        pivot_reviews = agg_reviews.pivot(
            index="business_id", columns="sentiment", values="text"
        )

        for col in ["positive", "neutral", "negative"]:
            if col not in pivot_reviews.columns:
                pivot_reviews[col] = np.nan

        # Convert NaNs to empty lists
        pivot_reviews = pivot_reviews.map(
            lambda x: x if isinstance(x, list) else []
        )

        # 3. Sort reviews by length (as in original logic)
        for col in pivot_reviews.columns:
            pivot_reviews[col] = pivot_reviews[col].apply(
                lambda x: sorted(x, key=len) if isinstance(x, list) else []
            )

        pivot_reviews.reset_index(inplace=True)
        final_merged = pd.merge(
            self.final_df, pivot_reviews, on="business_id", how="left"
        )

        config.OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        final_merged.to_pickle(config.OUTPUT_PATH)
        logger.info(f"Pipeline complete. Saved to {config.OUTPUT_PATH}")

    def run(self):
        """Execute the pipeline."""
        logger.info("Starting Yelp Pipeline...")

        try:
            self._load_and_process_business()
            self._stream_and_filter_reviews()
            self._score_and_rank()
            self._select_top_reviews()
            self._build_final_output()
        except Exception:
            logger.error("Pipeline failed.", exc_info=True)
            raise


if __name__ == "__main__":
    pipeline = YelpRestaurantPipeline()
    pipeline.run()
