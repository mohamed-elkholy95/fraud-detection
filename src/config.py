# Fraud Detection - Configuration
import logging
from pathlib import Path

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Cost parameters for threshold optimization
COST_FALSE_POSITIVE = 10.0   # Customer friction cost
COST_FALSE_NEGATIVE = 500.0  # Fraud loss cost

# Dataset
DATASET_URL = "https://datahub.io/machine-learning/creditcard/r/creditcard.csv"
RAW_DATA_PATH = RAW_DIR / "creditcard.csv"

# Model defaults
RANDOM_SEED = 42
TEST_RATIO = 0.2
N_FEATURES = 30  # V1-V28 + Time + Amount

# Autoencoder
AUTOENCODER_ENCODING_DIM = 15
AUTOENCODER_EPOCHS = 50
AUTOENCODER_BATCH_SIZE = 256
ANOMALY_PERCENTILE = 95

# Drift detection
KS_THRESHOLD = 0.05
PSI_THRESHOLD = 0.25
DRIFT_WINDOW_SIZE = 1000

# Feature engineering
AMOUNT_BINS = [0, 10, 50, 200, 500, float("inf")]
AMOUNT_BIN_LABELS = ["0-10", "10-50", "50-200", "200-500", "500+"]

# API
API_HOST = "0.0.0.0"
API_PORT = 8000

# Streamlit theme
STREAMLIT_THEME = {
    "primaryColor": "#1f77b4",
    "backgroundColor": "#0e1117",
    "secondaryBackgroundColor": "#262730",
    "textColor": "#ffffff",
}
