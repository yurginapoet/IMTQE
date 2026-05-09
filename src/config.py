from pathlib import Path

class Config:
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "data"
    PROCESSED_DIR = DATA_DIR / "processed"
    MODELS_DIR = ROOT_DIR / "models"
    
    RANDOM_SEED = 42
    
    SENTENCE_TRAIN = PROCESSED_DIR / "sentence_train.parquet"
    SENTENCE_DEV = PROCESSED_DIR / "sentence_dev.parquet"


Config.MODELS_DIR.mkdir(exist_ok=True)