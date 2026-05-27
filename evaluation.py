import argparse
import time
import requests
from typing import List, Dict, Tuple
from tabulate import tabulate

TEST_QUERIES = [
    {
        "query": "best tacos in Philadelphia",
        "relevant": [
            "El Purepecha",
            "South Philly Barbacoa",
            "Blue Corn",
            "Mission Taqueria",
        ],
    },
    {
        "query": "romantic Italian dinner Philadelphia",
        "relevant": [
            "Bistro Romano",
            "Gran Caffe L'Aquila",
            "L'Angolo Ristorante",
            "Vetri Cucina",
        ],
    },
    {
        "query": "late night bars Philadelphia",
        "relevant": ["Glory Beer Bar & Kitchen", "South", "Barclay Prime"],
    },
    {
        "query": "best cheesesteak Philadelphia",
        "relevant": [
            "Dalessandro's Steaks & Hoagies",
            "Jim's South St",
            "John's Roast Pork",
            "Max's Steaks",
            "Sonny's Famous Steaks",
        ],
    },
    {
        "query": "brunch spots Philadelphia",
        "relevant": [
            "Cafe Lift",
            "Cafe La Maude",
            "On Point Bistro",
            "Sabrina's Café",
            "Honey's Sit N Eat",
        ],
    },
    {
        "query": "sushi Philadelphia",
        "relevant": [
            "Hikari Sushi",
            "Vic Sushi Bar",
            "Bleu Sushi",
            "Royal Sushi & Izakaya",
            "Tomo Sushi & Ramen",
        ],
    },
    {
        "query": "vegan restaurants Philadelphia",
        "relevant": ["Vedge", "Charlie Was a Sinner", "V Street", "HipCityVeg"],
    },
    {
        "query": "rooftop bars Philadelphia",
        "relevant": ["Harp & Crown"],
    },
    {
        "query": "best hot chicken Nashville",
        "relevant": [
            "Hattie B's Hot Chicken - Nashville",
            "Hattie B's Hot Chicken - Lower Broadway",
            "Prince's Hot Chicken Shack",
            "Prince's Hot Chicken South",
            "Music City Chicken",
        ],
    },
    {
        "query": "romantic dinner Nashville",
        "relevant": [
            "The Optimist - Nashville",
            "Merchants",
            "The Standard At The Smith House",
            "Jeff Ruby's Steakhouse- Nashville",
        ],
    },
    {
        "query": "live music bars Nashville",
        "relevant": [
            "Jason Aldean's Kitchen + Rooftop Bar",
            "Bourbon Street Blues & Boogie Bar",
            "Skull's Rainbow Room",
            "Ole Smoky Distillery/Yee-Haw Brewing Co.",
        ],
    },
    {
        "query": "best BBQ Nashville",
        "relevant": [
            "Martin's Bar-B-Que Joint",
            "Charcoal Cowboys BBQ",
            "HoneyFire BBQ",
            "Constant Smoke BBQ",
        ],
    },
    {
        "query": "brunch Nashville",
        "relevant": [
            "The Garden Brunch Cafe",
            "Tavern",
            "Another Broken Egg Cafe",
            "Big Bad Breakfast - Nashville",
        ],
    },
    {
        "query": "best Cuban food Tampa",
        "relevant": ["Cuban Foodies", "Box Of Cubans", "La Teresita Cafe"],
    },
    {
        "query": "seafood restaurants Tampa",
        "relevant": [
            "Shells Seafood Restaurant- Carrollwood",
            "Shells Seafood Restaurant - South Tampa",
            "Eddie V's Prime Seafood",
            "Heights Seafood",
            "Oystercatchers",
        ],
    },
    {
        "query": "best pizza Tampa",
        "relevant": [
            "Fabrica Pizza",
            "Eddie & Sam's NY Pizza",
            "Mirro's Pizzeria",
            "Due Amici",
        ],
    },
    {
        "query": "sushi Tampa",
        "relevant": ["Sushi Cafe", "Soho Sushi", "Matoi Sushi", "Izakaya Tori"],
    },
    {
        "query": "best gumbo New Orleans",
        "relevant": ["Gumbo Shop", "Restaurant Rebirth", "Li'l Dizzy's Cafe"],
    },
    {
        "query": "late night food New Orleans",
        "relevant": ["Daisy Dukes Express", "Olde Nola Cookery"],
    },
    {
        "query": "best beignets New Orleans",
        "relevant": ["Café Du Monde", "Cafe Beignet on Royal Street"],
    },
    {
        "query": "romantic dinner New Orleans",
        "relevant": [
            "Palace Café",
            "Mr. B's Bistro",
            "Coquette",
            "Broussard's",
            "Doris Metropolitan",
        ],
    },
    {
        "query": "cozy coffee shop to study",
        "relevant": [
            "The Broad Street Grind",
            "Chapterhouse Café & Gallery",
            "Felicitous",
            "Picasso's Coffee House",
            "Kaffeine Coffee",
        ],
    },
    {
        "query": "family friendly pizza place",
        "relevant": [
            "Your Pie - Brandon",
            "Little Anthony Pizza",
            "Slimms Pizza and Salads",
            "Uncle Maddios Pizza",
        ],
    },
    {
        "query": "best ramen spots",
        "relevant": [
            "Ramen House",
            "Terakawa Ramen",
            "Ramen Ray",
            "Raijin Ramen",
            "Uncommon Ramen",
        ],
    },
    {
        "query": "spicy food lovers restaurant",
        "relevant": [
            "Spicy Affair",
            "Thai Spice",
            "Spice Indian Cuisine",
            "Spice Kitchen",
            "Thai Basil",
        ],
    },
]


def mrr_at_k(results: List[Dict], relevant: set, k: int = 5) -> float:
    """1 / rank of first relevant result in top-k. 0 if none found."""
    for i, r in enumerate(results[:k]):
        if (r.get("restaurant") or r.get("name") or "").strip() in relevant:
            return 1.0 / (i + 1)
    return 0.0


def hit_at_k(results: List[Dict], relevant: set, k: int) -> float:
    """1 if any relevant result in top-k, else 0."""
    return float(
        any(
            (r.get("restaurant") or r.get("name") or "").strip() in relevant
            for r in results[:k]
        )
    )


def precision_at_k(results: List[Dict], relevant: set, k: int) -> float:
    """Fraction of top-k results that are relevant."""
    hits = sum(
        1
        for r in results[:k]
        if (r.get("restaurant") or r.get("name") or "").strip() in relevant
    )
    return hits / min(k, len(results)) if results else 0.0


def retrieve(
    url: str, query: str, top_k: int = 5, do_rerank: bool = True
) -> Tuple[List[Dict], float]:
    """Returns (results, latency_ms)."""
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{url}/retrieve",
            json={"query": query, "top_k": top_k, "do_rerank": do_rerank},
            timeout=60,
        )
        resp.raise_for_status()
        latency = (time.perf_counter() - t0) * 1000
        data = resp.json()
        results = data if isinstance(data, list) else data.get("results", [])
        return results, latency
    except Exception as e:
        print(f" Failed: {e}")
        return [], (time.perf_counter() - t0) * 1000


def evaluate(url: str, top_k: int = 5):
    strategies = [
        {"name": "Hybrid + Rerank", "do_rerank": True},
        {"name": "Hybrid (no rerank)", "do_rerank": False},
    ]

    all_metrics = {
        s["name"]: {"mrr5": [], "hit3": [], "hit5": [], "p5": [], "latency": []}
        for s in strategies
    }

    total = len(TEST_QUERIES)
    print(f"\n🔍 RezRag Evaluation — {total} queries, human-labeled relevance\n")
    print(f"   URL: {url}\n")

    for i, test in enumerate(TEST_QUERIES):
        query = test["query"]
        relevant = set(test["relevant"])
        print(f"  [{i+1:02d}/{total}] {query}")

        for s in strategies:
            results, latency = retrieve(
                url, query, top_k=top_k, do_rerank=s["do_rerank"]
            )
            if not results:
                continue

            m = all_metrics[s["name"]]
            m["mrr5"].append(mrr_at_k(results, relevant, k=5))
            m["hit3"].append(hit_at_k(results, relevant, k=3))
            m["hit5"].append(hit_at_k(results, relevant, k=5))
            m["p5"].append(precision_at_k(results, relevant, k=5))
            m["latency"].append(latency)

        time.sleep(0.3)

    print("\n" + "=" * 68)
    print("  REZRAG RETRIEVAL EVALUATION ")
    print("=" * 68 + "\n")

    rows = []
    for name, m in all_metrics.items():
        n = len(m["mrr5"])
        if not n:
            continue
        rows.append(
            [
                name,
                f"{sum(m['mrr5'])/n:.3f}",
                f"{sum(m['hit3'])/n:.3f}",
                f"{sum(m['hit5'])/n:.3f}",
                f"{sum(m['p5'])/n:.3f}",
                f"{sum(m['latency'])/n:.0f}ms",
                n,
            ]
        )

    print(
        tabulate(
            rows,
            headers=[
                "Strategy",
                "MRR@5",
                "Hit@3",
                "Hit@5",
                "P@5",
                "Avg Latency",
                "Queries",
            ],
            tablefmt="rounded_outline",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RezRag Human-Labeled Retrieval Evaluator"
    )
    parser.add_argument(
        "--url",
        default="https://megumind6172--food-rag-retriever-serve.modal.run",
        help="Retriever base URL",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Results per query")
    args = parser.parse_args()
    evaluate(url=args.url, top_k=args.top_k)
