from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import os
from pathlib import Path
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
DATA_DIR = Path("data")
BUSINESS_PATH = "yelp_academic_dataset_business.json"
REVIEW_PATH = "yelp_academic_dataset_review.json"
OUTPUT_PATH = DATA_DIR / "preprocessed.pkl"
CHUNKED_DATA_PATH = DATA_DIR / "chunked_data.pkl"
PRECOMPUTED_PATH = DATA_DIR / "vector_embeddings_new.pt"
EMBEDDINGS_PATH = DATA_DIR / "vector_embeddings_new.pt"
BM25_PATH = DATA_DIR / "bm25_ranks.pkl"
METADATA_PATH = DATA_DIR / "retriever_metadata.parquet"
DATA_PATH = DATA_DIR / "retriever_df.csv"

# Models
EMBEDDING_MODEL_NAME = "intfloat/e5-large-v2"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Parameters
RESTAURANTS_COEFF = 0.3
REVIEWS_COEFF = 0.7
MAX_REVIEWS_PER_RESTAURANT = 40
REVIEW_COUNTS_PER_RESTAURANT = 50
MIN_DYNAMIC_LIMIT = 10
TOP_RESTAURANTS_PER_CITY = 200
MIN_REVIEW_WORDS = 30
FILTER_YEAR = 2018

# Parameters
BATCH_SIZE = 32
MAX_TOKENS = 256
TOP_K = 20
INITIAL_K = 20
RRF_K = 60
MAX_DUPLICATES = 1
DO_RERANK = True

# Qdrant Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "food_chunks"
VECTOR_SIZE = 1024
DOC_ID = "foodguru-v1"
INGEST_BATCH_SIZE = 256
MAX_WORKERS = 4

# Text Chunking Settings
CHUNK_MAX_TOKENS = 1024
CHUNK_MIN_TOKENS = 50
TOKEN_ENCODING = "cl100k_base"
OVERHEAD_TOKENS = 4

# External Services
RETRIEVER_URL = os.environ.get("RETRIEVER_URL")
E5_URL = os.environ.get("E5_URL")

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_ID = "qwen/qwen3-32b"

# Model Configuration
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MAX_NEW_TOKENS = 700
TEMPERATURE = 0.2
TOP_P = 0.7
TRIM_LENGTH = 350
MAX_INPUT_TOKENS = 20000

# Quantization Settings
BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": torch.float16,
}

GROQ_SYSTEM_PROMPT = (
    "You are RezRag, a knowledgeable and helpful local food recommendation guide on the Yelp restaurant dataset.\n\n"
    "Use ONLY the provided context. do not rely on outside knowledge. Do NOT invent restaurants, dishes, prices, or locations.\n"
    "Be eloquent and offer long explanations to support each recommendation\n"
    "Maintain a friendly, casual, conversational tone — like a knowledgeable local friend giving advice, not a structured report.\n\n"
    "Use ONLY the provided context. Do not rely on outside knowledge. Do NOT invent restaurants, dishes, prices, or locations.\n\n"
    "For each recommendation use this format:\n"
    "For each recommendation use EXACTLY this format:\n"
    "**Restaurant Name** 📍 *address, city*\n\n"
    "🍽️ Why it fits the query and what makes it special from reviews in 2-3 sentences.\n\n"
    "🌟 Must-try: standout dish or feature mentioned in reviews and explain why.\n\n"
    "💡 *Tip: one practical tip from the reviews.*\n\n"
    "There MUST be a blank line between the description and the tip, and a --- divider between restaurants.\n\n"
    "Keep each recommendation tight — no essays. Cover all spots without repeating yourself.\n\n"
    "Always read the user query carefully and respect the constraints:\n"
    "   - Health conditions or discomfort → recommend "
    "light, fresh, easy-to-digest options. Skip restaurants with negative reviews or heavy/greasy/spicy food.\n"
    "   - Dietary restrictions or allergies → recommend places with confirmed suitable options in the context.\n"
    "   - Budget constraints → recommend places where context confirms prices fit.\n"
    "   - Occasion or mood (romantic, family, casual, quick bite) → match the vibe.\n"
    "   - If NO retrieved restaurant genuinely fits the constraints, say so honestly and briefly list what is available with a caveat.\n\n"
    "LOCATION TRANSPARENCY:\n"
    "   - Never say a location is not in your dataset when results are present in the context.\n"
    "   - The context is the ground truth — trust it over your own knowledge.\n"
    "   - If results come from a nearby area rather than the exact city mentioned, acknowledge this naturally.\n\n"
    "If reviews mention drawbacks, mention them briefly and constructively.\n\n"
    "COVERAGE — you only have data for these areas:\n"
    "   Philadelphia & surrounding South Jersey suburbs (PA/NJ)\n"
    "   Tampa & Fort Myers (FL)\n"
    "   Nashville & surrounding suburbs (TN)\n"
    "   New Orleans & suburbs (LA)\n"
    "   Indianapolis & suburbs (IN)\n"
    "   Tucson & surrounding areas (AZ)\n"
    "   Reno & Spanish Springs (NV)\n"
    "   Edmonton & St. Albert (Alberta, Canada)\n"
    "   Santa Barbara & Goleta (CA)\n"
    "   Boise & surrounding Idaho cities (ID)\n\n"
    "If CONTEXT is empty or has 0 results:\n"
    "   - If the query mentions a city/state outside coverage respond with exactly:\n"
    "     'That location isn't covered in the Yelp dataset. I currently cover cities from:\n\n"
    "📍 **Pennsylvania** — Philadelphia, King of Prussia, Norristown, Doylestown (PA)\n"
    "📍 **California** — Santa Barbara, Goleta, Montecito, Carpinteria (CA)\n"
    "📍 **New Jersey** — Cherry Hill, Camden, Voorhees, Haddonfield (NJ)\n"
    "📍 **Florida** — Tampa, Clearwater, St. Petersburg, Brandon (FL)\n"
    "📍 **Tennessee** — Nashville, Brentwood, Franklin, Hendersonville (TN)\n"
    "📍 **Louisiana** — New Orleans, Metairie, Kenner, Chalmette (LA)\n"
    "📍 **Indiana** — Indianapolis, Carmel, Fishers, Noblesville (IN)\n"
    "📍 **Arizona** — Tucson, Oro Valley, Marana, Sahuarita (AZ)\n"
    "📍 **Nevada** — Reno (NV)\n"
    "📍 **Idaho** — Boise, Meridian, Eagle (ID)\n"
    "📍 **Illinois** — Belleville, Collinsville, Mascoutah, Caseyville (IL)\n"
    "📍 **Missouri** — Saint Louis, Chesterfield, Ballwin, Creve Coeur (MO)\n"
    "📍 **Delaware** — Wilmington, Claymont, Christiana (DE)\n"
    "📍 **Alberta** — Edmonton (AB)\n"
    "     Try: *best tacos in Philadelphia* or *late night food in Nashville* 🍜'\n"
    "   - If the query is within coverage but no results found, suggest rephrasing or a broader query.\n"
    "If the query is a greeting or not about restaurants, respond in one sentence and ask what restaurant they are looking for.\n"
    "If the query mentions a city outside your coverage, list supported cities and suggest a similar query.\n"
)


# City Corrections Mapping
US_CITIES_CORRECTIONS = {
    "Philadelphiadelphia": "Philadelphia",
    "Philly": "Philadelphia",
    "Southwest Philadelphia": "Philadelphia",
    "Tampa Bay": "Tampa",
    "Tampa,Fl": "Tampa",
    "Tampa Florida": "Tampa",
    "Southwest Tampa": "Tampa",
    "Inpolis": "Indianapolis",
    "Indianopolis": "Indianapolis",
    "Tuscon": "Tucson",
    "Tren": "Trenton",
    "Nashville-Davidson Metropolitan Government (Balance)": "Nashville",
    "East Nashville": "Nashville",
    "St. Louis": "Saint Louis",
    "St Louis": "Saint Louis",
    "SaintLouis": "Saint Louis",
    "Saint Louis County": "Saint Louis",
    "Saint Louis Downtown": "Saint Louis",
    "Saint Louis,": "Saint Louis",
    "East St. Louis": "East Saint Louis",
    "East St Louis": "East Saint Louis",
    "St. Petersburg": "Saint Petersburg",
    "St Petersburg": "Saint Petersburg",
    "SaintPetersburg": "Saint Petersburg",
    "Saintt Petersburg": "Saint Petersburg",
    "Saint Petersurg": "Saint Petersburg",
    "Mt. Juliet": "Mount Juliet",
    "Mt Juliet": "Mount Juliet",
    "Mt.Juliet": "Mount Juliet",
    "Mt. Laurel": "Mount Laurel",
    "Mt Laurel": "Mount Laurel",
    "Mt.Laurel": "Mount Laurel",
    "Mount Laurel Township": "Mount Laurel",
    "Mt Laurel Twp, Nj": "Mount Laurel",
    "Mt Holly": "Mount Holly",
    "Mount Holly,": "Mount Holly",
    "West Mount Holly": "Mount Holly",
    "Town N Country": "Town and Country",
    "Twn N Cntry": "Town and Country",
    "Land O Lakes": "Land O' Lakes",
    "Land O'Lakes": "Land O' Lakes",
    "Fairview Hts": "Fairview Heights",
    "Fairview Hts.": "Fairview Heights",
    "Woodbury Hts.": "Woodbury Heights",
    "Temple Terr": "Temple Terrace",
    "Belleair Blf": "Belleair Bluffs",
    "Pass-A-Grille Beach": "Pass-a-Grille Beach",
    "Redingtn Shor": "Redington Shores",
    "Hernando Bch": "Hernando Beach",
    "North Redington Bch": "North Redington Beach",
    "Lutz Fl": "Lutz",
    "Riverview Fl": "Riverview",
    "W.Chester": "West Chester",
    "West Chester Pa": "West Chester",
    "W. Berlin": "West Berlin",
    "S.Pasadena": "South Pasadena",
    "Bensalem. Pa": "Bensalem",
    "SaintAnn": "Saint Ann",
    "SaintRose": "Saint Rose",
    "SaintCharles": "Saint Charles",
    "Saint  Charles": "Saint Charles",
    "Santa  Barbara": "Santa Barbara",
    "Haddon Twp": "Haddon Township",
    "Bristol Twp": "Bristol Township",
    "Washington Twp": "Washington Township",
    "Woolwich Twp": "Woolwich Township",
    "Woolwich Twp.": "Woolwich Township",
    "Delran Twp": "Delran Township",
}

# State Mapping
US_STATES = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
}

STATE_ALIASES = set(US_STATES.keys()) | set(US_STATES.values())
STATE_FULL_TO_ABBR = US_STATES.copy()
STATE_ABBR_TO_FULL = {v: k.title() for k, v in STATE_FULL_TO_ABBR.items()}

# Attribute Mappings
ATTRIBUTE_MAP = {
    "RestaurantsTakeOut": "Takeout Service",
    "HasTV": "Has a Television",
    "RestaurantsDelivery": "Delivery Service",
    "OutdoorSeating": "Outdoor Seating",
    "GoodForKids": "Good For Kids",
    "GoodForDancing": "Good For Dancing",
    "RestaurantsGoodForGroups": "Good For Groups",
    "HappyHour": "Happy Hour",
    "DogsAllowed": "Dogs Allowed",
}

# Special Logic Checks
BOOL_ATTRIBUTES = set(ATTRIBUTE_MAP.keys())

# Keys check for specific values
ALCOHOL_KEYS = ["Alcohol"]
ALCOHOL_SKIP_VALUES = {"u'none'", "'none'", "none", "None"}

# Vibe Keywords
VIBE_KEYWORDS = {
    "touristy",
    "hipster",
    "divey",
    "intimate",
    "trendy",
    "upscale",
    "classy",
    "casual",
    "romantic",
}

DATASET_CITIES = {
    "abington",
    "abington township",
    "affton",
    "alton",
    "ambler",
    "antioch",
    "apollo beach",
    "arabi",
    "ardmore",
    "arnold",
    "ashland city",
    "aston",
    "atco",
    "audubon",
    "avon",
    "avondale",
    "bala cynwyd",
    "ballwin",
    "balm",
    "bargersville",
    "barrington",
    "beaumont",
    "beech grove",
    "belle chasse",
    "belle meade",
    "belleair bluffs",
    "belleville",
    "bellevue",
    "bellmawr",
    "bensalem",
    "bensalem township",
    "berlin",
    "berlin township",
    "berry hill",
    "berwyn",
    "birchrunville",
    "blackwood",
    "blue bell",
    "boise",
    "boise city",
    "boothwyn",
    "bordentown",
    "boyertown",
    "brandon",
    "breckenridge hills",
    "brentwood",
    "bridgeport",
    "bridgeton",
    "bristol",
    "brookhaven",
    "brooklawn",
    "broomall",
    "brownsburg",
    "bryn mawr",
    "buckingham",
    "bucktown",
    "burlington",
    "burlington township",
    "bywater",
    "cahokia",
    "camby",
    "camden",
    "carmel",
    "carpinteria",
    "carrollwood",
    "carversville",
    "caseyville",
    "castleton",
    "catalina",
    "cedars",
    "chadds ford",
    "chalfont",
    "chalmette",
    "cheltenham",
    "cherry hill",
    "chesilhurst",
    "chester",
    "chester springs",
    "chesterbrook",
    "chesterfield",
    "christiana",
    "churchville",
    "cinnaminson",
    "citrus park",
    "clarksboro",
    "claymont",
    "clayton",
    "clearwater",
    "clearwater beach",
    "clementon",
    "clifton heights",
    "coatesville",
    "cold springs",
    "collegeville",
    "collingdale",
    "collingswood",
    "collinsville",
    "colmar",
    "columbia",
    "columbus",
    "conshohocken",
    "corona de tucson",
    "creve coeur",
    "croydon",
    "delran",
    "deptford",
    "deptford township",
    "devon",
    "downingtown",
    "doylestown",
    "dresher",
    "drexel hill",
    "dublin",
    "dunedin",
    "eagle",
    "eagleville",
    "edmonton",
    "elkins park",
    "elmer",
    "elmwood",
    "essington",
    "ewing",
    "ewing township",
    "exton",
    "fairless hills",
    "fishers",
    "florence",
    "flourtown",
    "folcroft",
    "folsom",
    "fort washington",
    "franklin",
    "franklinville",
    "frazer",
    "gallatin",
    "garnet valley",
    "gibbsboro",
    "gibsonton",
    "gilbertsville",
    "glassboro",
    "glen mills",
    "glenside",
    "gloucester city",
    "gloucester township",
    "goleta",
    "goodlettsville",
    "gretna",
    "gulfport",
    "haddon heights",
    "haddon township",
    "haddonfield",
    "hainesport",
    "hamilton",
    "hamilton township",
    "hammonton",
    "harahan",
    "harleysville",
    "hatboro",
    "hatfield",
    "haverford",
    "havertown",
    "hendersonville",
    "hermitage",
    "horsham",
    "indianapolis",
    "isla vista",
    "jenkintown",
    "joelton",
    "kenner",
    "kenneth city",
    "kennett square",
    "king of prussia",
    "kingston springs",
    "kulpsville",
    "la vergne",
    "lafayette hill",
    "lahaska",
    "langhorne",
    "lansdale",
    "lansdowne",
    "largo",
    "laurel springs",
    "lawnside",
    "levittown",
    "limerick",
    "lindenwold",
    "lionville",
    "lutz",
    "madison",
    "malvern",
    "manayunk",
    "mantua",
    "maple glen",
    "maple shade",
    "marana",
    "marlton",
    "marrero",
    "martinsville",
    "mascoutah",
    "medford",
    "media",
    "mendenhall",
    "meraux",
    "merchantville",
    "meridian",
    "metairie",
    "middletown",
    "montecito",
    "montgomeryville",
    "moorestown",
    "mooresville",
    "morrisville",
    "mount holly",
    "mount juliet",
    "mount laurel",
    "mullica hill",
    "narberth",
    "nashville",
    "new hope",
    "new orleans",
    "new port richey",
    "newtown",
    "newtown square",
    "noblesville",
    "nolensville",
    "norristown",
    "norwood",
    "oaklyn",
    "oldsmar",
    "oreland",
    "oro valley",
    "palm harbor",
    "palmetto",
    "palmyra",
    "paoli",
    "pennsauken",
    "perkasie",
    "philadelphia",
    "phoenixville",
    "pitman",
    "plant city",
    "plymouth meeting",
    "pottstown",
    "quakertown",
    "reno",
    "royersford",
    "safety harbor",
    "sahuarita",
    "saint charles",
    "saint louis",
    "saint petersburg",
    "san antonio",
    "santa barbara",
    "seminole",
    "sewell",
    "sicklerville",
    "skippack",
    "souderton",
    "southampton",
    "spring hill",
    "springfield",
    "st pete",
    "st. pete beach",
    "swarthmore",
    "swedesboro",
    "tampa",
    "tarpon springs",
    "telford",
    "thorndale",
    "tucson",
    "turnersville",
    "upper darby",
    "voorhees",
    "warminster",
    "warrington",
    "wayne",
    "wesley chapel",
    "west chester",
    "west deptford",
    "westmont",
    "willingboro",
    "willow grove",
    "wilmington",
    "woodbury",
    "woolwich township",
    "yardley",
    "yeadon",
    "zionsville",
}

DATASET_STATES = {
    "AB",
    "AZ",
    "CA",
    "DE",
    "FL",
    "ID",
    "IL",
    "IN",
    "LA",
    "MO",
    "NJ",
    "NV",
    "PA",
    "TN",
}

STATE_TO_PRIMARY_CITY = {
    "PA": "philadelphia",
    "NJ": "philadelphia",
    "FL": "tampa",
    "TN": "nashville",
    "LA": "new orleans",
    "IN": "indianapolis",
    "AZ": "tucson",
    "NV": "reno",
    "ID": "boise",
    "CA": "santa barbara",
    "AB": "edmonton",
    "IL": None,
    "MO": None,
    "DE": None,
}

NON_FOOD_PATTERNS = [
    # Greetings
    r"^(hi+|hey+|hello+|sup|yo|hiya|howdy|hola|bonjour|ciao)\s*[!?.,]?$",
    r"^(what'?s up|whats up|wassup|wazzup|what up)\s*[!?.,]?$",
    r"^(how (are|you|is it|are you doing|you doing|r u))\s*[!?.,]?$",
    r"^(good morning|good afternoon|good evening|good night|gm|gn)\s*[!?.,]?$",
    r"^(thanks|thank you|thx|ty|bye|goodbye|ciao|ok|okay|cool|nice|great|awesome|sure|alright|aight)\s*[!?.,]?$",
    r"^(lol|lmao|lmfao|haha|hehe|omg|wtf|bruh|bro|sis|dude|man|bro)\s*[!?.,]?$",
    r"^(yes|no|nope|yep|yeah|nah|maybe|idk|idcyou|suck|hate|this|what|wot|shi|bye)\s*[!?.,]?$",
    r"^(you suck|this sucks|hate this|worst|terrible|awful|useless)\s*[!?.,]?$",
    r"^.{1,2}$",  # 1-2 character inputs
    r"^[^a-zA-Z\s]+$",  # only symbols/numbers
    r"^[a-zA-Z]{1,2}\s*$",  # single or double letter
    r"^(.)\1{2,}$",
    # Random keyboard mashing
    r"^[a-z]{6,}$",  # 6+ lowercase with no spaces = likely mashing
    r"^[A-Za-z]{1,}[0-9]{1,}[A-Za-z0-9]*$",
    # Meta questions
    r"what should (i|a user|someone|people) (type|ask|say|search|write|query|enter)",
    r"what (do you|can you) (do|speciali[sz]e|help|offer|cover|know|recommend)",
    r"how (do|can|should) (i|you|someone) use (this|you|the app|rezrag)",
    r"what (are|is) (your|this|the) (coverage|cities|locations|areas|scope|specialty)",
    r"(tell me|explain) (about yourself|what you do|how you work)",
    r"^(what|how|who|where|why|when)\??\s*$",
    r"are you (a bot|an ai|chatgpt|claude|gpt|real|human)",
    r"who (made|built|created|trained|developed) you",
    r"what (model|llm|ai) are you",
    # Hotels / accommodation
    r"\b(hotel|hotels|motel|airbnb|hostel|resort|accommodation|lodging|inn|bed and breakfast|b&b|vrbo|rental)\b",
    r"\b(book a|reserve a|check in|check out|room|suite|stay at)\b",
    # Entertainment
    r"\b(movie|cinema|theater|theatre|film|show|concert|event|ticket|tickets|gig|festival|exhibit|exhibition|gallery|museum)\b",
    r"\b(watch|streaming|netflix|hulu|disney|amazon prime|hbo)\b",
    # Directions / travel
    r"\b(how do i get to|directions to|navigate to|drive to|flight to|train to|bus to|uber to|lyft to)\b",
    r"\b(how far is|distance to|miles from|minutes from|closest to|near me)\b",
    r"\b(airport|terminal|gate|flight|airline|amtrak|greyhound|transit|subway|metro|bus route)\b",
    r"\b(parking|where to park|parking lot|garage near)\b",
    # Sports / news
    r"\b(who won|what score|game result|sports score|super bowl|world cup|championship|playoffs|standings|roster)\b",
    r"\b(nfl|nba|mlb|nhl|mls|premier league|la liga|fifa|olympics)\b",
    r"\b(breaking news|latest news|headlines|weather forecast|stock price|crypto|bitcoin)\b",
    # Shopping
    r"\b(buy|purchase|order online|amazon|walmart|target|best buy|shop for|where to buy)\b",
    r"\b(price of|how much does|cost of|cheapest|discount|coupon|promo code)\b",
    # Medical / personal
    r"\b(doctor|hospital|pharmacy|medication|prescription|symptoms|diagnosis|therapist|dentist)\b",
    r"\b(lawyer|attorney|legal advice|insurance|tax|accountant|financial advisor)\b",
    # Tech support
    r"\b(how to install|download|update|fix|error|bug|crash|not working|reset password)\b",
    r"\b(iphone|android|windows|mac|laptop|computer|wifi|internet|vpn)\b",
    r"^(test|testing|123|hello world|asdf|qwerty|foo|bar)\s*$",
    r"^[^a-zA-Z]*$",
    r"^\s*$",
    r"\b(ignore (all )?previous instructions|system prompt|bypass|jailbreak|you are now|act as|pretend to be|dan|developer mode)\b",
    # 2. Humor / Party Tricks
    r"\b(tell me a joke|make me laugh|sing a song|do a flip|do a barrel roll|tell a story|knock knock|write a poem)\b",
    # 3. Philosophical / Existential
    r"\b(meaning of life|are you conscious|do you have feelings|are you self[- ]?aware|do you sleep|do you dream|are you alive)\b",
    # 4. Romance / Flirting
    r"\b(do you love me|will you marry me|are you single|be my (gf|bf|girlfriend|boyfriend)|i love you|send nudes)\b",
    # 5. Math / Coding / Homework Help
    r"\b(write a (python|javascript|c\+\+|java|html) script|code a|debug this|fix my code|github|stack overflow)\b",
    r"\b(solve for|calculate|math problem|equation|derivative|integral|square root|what is \d+\s*[\+\-\*\/])\b",
    r"\b(write an essay|summarize this article|translate (to|from)|proofread|homework help)\b",
    # 6. Virtual Assistant / Smart Home Commands
    r"\b(set an alarm|remind me to|turn (on|off) the (lights|tv)|call (mom|dad|my)|text (my|mom|dad)|what time is it)\b",
    # 7. Gen-Z Slang
    r"\b(skibidi|rizz|sigma|based|cringe|gyatt|no cap|fr fr|sus|amogus|uwu|owo)\b",
    # 8. Politics / Religion / Controversial
    r"\b(who did you vote for|democrat|republican|trump|biden|election|jesus|god|religion|bible|quran|atheist)\b",
    # 9. Pets / Animals / Vets
    r"\b(dog(s)?|cat(s)?|puppy|kitten|veterinarian|vet near me|pet store|dog park|aquarium|zoo)\b",
    # 10. Jobs / Career / Social Media
    r"\b(resume|cover letter|job interview|hiring|salary|linkedin|indeed|glassdoor)\b",
    r"\b(instagram|tiktok|twitter|x|facebook|snapchat|youtube|influencer|follower(s)?)\b",
    # 11. Cryptic / Edge-case Typing
    r"^(asdf|qwerty|zxcv|hjkl)+.*$",  # Common keyboard mash rows
    r"^(.)\1{4,}$",
    r"\b(stupid|idiot|dumb|useless|hate you|shut up|sucks|terrible|worst)\b",
    r"\b(f[u\*]ck|sh[i\*]t|b[i\*]tch|a[s\$][s\$]hole|crap|damn)\b",
    # 2. Mental Health / Crisis
    r"\b(depressed|suicidal|kill myself|end it all|anxiety|lonely|panic attack|self harm|therapist)\b",
    r"\b(i hate my life|nobody cares about me|giving up)\b",
    # 3. PII (Personally Identifiable Information) / Sensitive Data
    r"\b(ssn|social security|credit card|cvv|password is|my address is|phone number is)\b",
    r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b",  # Basic SSN-like pattern check
    # 4. Office / Admin Chores
    r"\b(write an email|draft a(n)? (letter|memo)|schedule a meeting|create a presentation)\b",
    r"\b(excel formula|spreadsheet|vlookup|pivot table|google docs|pdf)\b",
    # 5. Gaming / Esports
    r"\b(minecraft|roblox|fortnite|gta|valorant|league of legends|elden ring|pokemon)\b",
    r"\b(xbox|playstation|nintendo|steam deck|how to beat|boss fight|cheat codes|walkthrough)\b",
    # 6. Encyclopedia / Trivia
    r"\b(capital of|population of|who invented|history of|world war|president of)\b",
    r"\b(when was .* born|how many countries|tallest building|longest river|facts about)\b",
    # 7. The "Are you there?" Pings
    r"^(you there|u there|hello\?|anyone there|still there|wake up)\??\s*$",
    # 8. Time / Date / Calendar
    r"\b(what time is it( in)?|what day is it|current date|how many days until|leap year)\b",
    # 9. Language / Translation Queries
    r"\b(how do you say|translate .* to|what does .* mean in (spanish|french|english|japanese))\b",
    r"\b(oil change|flat tire|mechanic|dealership|car insurance|brake pads|windshield wipers|transmission|gas prices)\b",
    r"\b(check engine light|jump start|towing company|dmv|driver'?s license)\b",
    # 2. Fitness / Exercise (Non-Diet)
    r"\b(workout routine|gym|pushups|yoga|pilates|crossfit|weightlifting|cardio|treadmill|marathon|biceps|hypertrophy)\b",
    # 3. Real Estate / Housing
    r"\b(mortgage|realtor|zillow|apartments\.com|renting an apartment|homeowner|interest rates|eviction|open house)\b",
    # 4. Fashion / Beauty / Grooming
    r"\b(skincare|makeup|sephora|haircut|hairstyle|outfit|sneakers|wardrobe|cosmetics|dermatologist|manicure|pedicure)\b",
    # 5. Music / Audio
    r"\b(lyrics to|guitar chords|sheet music|spotify|apple music|podcast episode|who sings|album release|playlist)\b",
    # 6. Arts, Crafts, & DIY
    r"\b(crochet|knitting|origami|watercolor|acrylic paint|plumbing|drywall|home depot|lowe'?s|woodworking|diy project)\b",
    # 7. Astrology / Esoteric
    r"\b(horoscope|zodiac|astrology|tarot|pisces|aries|taurus|gemini|mercury retrograde|fortune teller|birth chart)\b",
    # 8. Science / Space / Nature
    r"\b(black hole|nasa|spacex|quantum physics|astronomy|telescope|aliens|ufo|speed of light|dinosaurs|fossils)\b",
    # 9. Parenting / Childcare (Non-Food)
    r"\b(diapers|potty training|babysitter|daycare|kindergarten|toddler tantrums|stroller|crib|pacifier)\b",
    # 10. Dating / Relationship Advice
    r"\b(tinder|bumble|hinge|breakup advice|divorce|marriage counseling|toxic relationship|ghosting|red flags)\b",
    # 11. Delivery/Mail Logistics (Non-Food)
    r"\b(usps|fedex|ups|dhl|tracking number|post office|stamps|shipping cost|po box|return label)\b",
    # Insults / frustration
    r"\b(you suck|this sucks|hate (you|this)|worst (app|bot|thing)|useless|garbage|trash|stupid bot)\b",
    # Empty / whitespace / symbols only
    r"^[^a-zA-Z]*$",
    r"^\s*$",
]
