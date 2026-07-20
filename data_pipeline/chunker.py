import pandas as pd
import numpy as np
import re
import ast
import logging
from tqdm.auto import tqdm
from typing import List, Dict, Any
from transformers import AutoTokenizer

import config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class YelpChunking:
    def __init__(self):
        self.enc = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL_NAME)
        self.preprocessed_path = config.DATA_DIR / "preprocessed.pkl"
        self.output_path = config.DATA_DIR / "chunked_data.pkl"
        self.df = None

    def _clean_text(self, t: str) -> str:
        """Fast regex text cleaning."""
        if not t:
            return ""

        t = re.sub(r"-\s*\n\s*|\r", " ", t)
        t = re.sub(r"\s+", " ", t)
        return t.strip()

    @staticmethod
    def _parse_attribute_dict(attr_str) -> dict:
        if pd.isna(attr_str):
            return {}
        try:
            data = (
                attr_str if isinstance(attr_str, dict) else ast.literal_eval(attr_str)
            )
            return data if isinstance(data, dict) else {}
        except (ValueError, SyntaxError):
            return {}

    @staticmethod
    def _parse_attributes(attrs: dict) -> List[str]:
        """Parses facility attributes from an already-parsed attribute dict."""
        features = []
        for k, v in attrs.items():
            val_str = str(v).strip().lower()

            if k in config.BOOL_ATTRIBUTES and val_str == "true":
                features.append(config.ATTRIBUTE_MAP[k])
            elif k in config.ALCOHOL_KEYS and str(v) not in config.ALCOHOL_SKIP_VALUES:
                features.append("Serves Alcohol")

        return features

    @staticmethod
    def _parse_vibes(attrs: dict) -> List[str]:
        """Parses vibe keywords from an already-parsed attribute dict."""
        vibes = []
        for k, v in attrs.items():
            if isinstance(v, str) and "{" in v:
                try:
                    inner = ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    continue
                if isinstance(inner, dict):
                    for vibe_k, vibe_v in inner.items():
                        if (
                            vibe_k in config.VIBE_KEYWORDS
                            and str(vibe_v).lower() == "true"
                        ):
                            vibes.append(vibe_k)
        return vibes

    def _create_review_batches(
        self, reviews: List[str], header: str, max_single_review_tokens: int = 300
    ) -> List[str]:
        """Batches reviews into chunks that fit within the token limit, tokenized
        with the embedding model's own tokenizer so the budget matches what the
        embedder actually sees."""
        if not reviews or not isinstance(reviews, list):
            return []

        header_tokens = len(self.enc.encode(header, add_special_tokens=False))
        chunks = []
        current_batch = []
        current_cost = header_tokens

        for r in reviews:
            r = self._clean_text(r)
            if not r:
                continue

            token_ids = self.enc.encode(r, add_special_tokens=False)
            if len(token_ids) > max_single_review_tokens:
                token_ids = token_ids[:max_single_review_tokens]
                r = self.enc.decode(token_ids, skip_special_tokens=True)
            token_cost = len(token_ids) + config.OVERHEAD_TOKENS

            if current_cost + token_cost <= config.CHUNK_MAX_TOKENS:
                current_batch.append(r)
                current_cost += token_cost
            else:
                if current_batch:
                    body = "\n\n".join(f"-- {rev}" for rev in current_batch)
                    chunks.append(f"{header}{body}")

                current_batch = [r]
                current_cost = header_tokens + token_cost

        if current_batch and current_cost >= config.CHUNK_MIN_TOKENS:
            body = "\n\n".join(f"-- {rev}" for rev in current_batch)
            chunks.append(f"{header}{body}")

        return chunks

    def process_row(self, row):
        """
        Generates all chunks for a single restaurant row.
        Returns a Series to be expanded into DataFrame columns.
        """
        name = row["name"]
        city = row["city"]
        state = row["state"]

        # 1. Metadata Chunks
        profile_chunks = []

        biz_text = (
            f"passage: {name} is a {row['categories']} restaurant "
            f"located at {row['address']} in {city}, {state}."
        )
        profile_chunks.append(biz_text)

        attrs = self._parse_attribute_dict(row["attributes"])

        features = self._parse_attributes(attrs)
        if features:
            attr_text = (
                f"passage: {name} provides features such as {', '.join(features)}."
            )
            profile_chunks.append(attr_text)

        vibes = self._parse_vibes(attrs)
        if vibes:
            vibe_text = (
                f"passage: The atmosphere at {name} is described as {', '.join(vibes)}."
            )
            profile_chunks.append(vibe_text)

        # 2. Review Chunks
        pos_header = f"passage: positive customer reviews for {name} in {city}, {state} mention:\n "
        neu_header = f"passage: neutral customer reviews for {name} in {city}, {state} mention:\n "
        neg_header = f"passage: negative customer reviews for {name} in {city}, {state} mention:\n "

        chunked_pos = self._create_review_batches(row.get("positive", []), pos_header)
        chunked_neu = self._create_review_batches(row.get("neutral", []), neu_header)
        chunked_neg = self._create_review_batches(row.get("negative", []), neg_header)

        return pd.Series([profile_chunks, chunked_pos, chunked_neu, chunked_neg])

    def load_and_format(self):
        logger.info(f"Loading data from {self.preprocessed_path}...")
        self.df = pd.read_pickle(self.preprocessed_path)

        self.df["categories"] = (
            self.df["categories"]
            .astype(str)
            .str.replace("Restaurants,", "", regex=False)
        )
        logger.info(f"Loaded {len(self.df)} restaurants, columns: {list(self.df.columns)}")

    def run(self):
        self.load_and_format()

        logger.info("Generating Chunks (Single Pass)...")
        tqdm.pandas(desc="Processing Rows")

        cols = ["chunked_profile", "chunked_pos", "chunked_neu", "chunked_neg"]
        self.df[cols] = self.df.progress_apply(self.process_row, axis=1)

        n_empty = (
            self.df[cols].apply(lambda row: sum(len(c) for c in row) == 0, axis=1).sum()
        )
        if n_empty:
            logger.warning(f"{n_empty} restaurants produced zero chunks.")

        logger.info("Saving chunks...")

        keep_cols = [
            "business_id",
            "name",
            "city",
            "state",
            "address",
            "latitude",
            "longitude",
            "chunked_profile",
            "chunked_pos",
            "chunked_neu",
            "chunked_neg",
        ]

        final_df = self.df[keep_cols]
        final_df.to_pickle(self.output_path)
        logger.info(f"Saved to {self.output_path}")


if __name__ == "__main__":
    chunker = YelpChunking()
    chunker.run()
