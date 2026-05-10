from pathlib import Path

class Config:
    ROOT_DIR = Path(__file__).parent.parent
    DATA_DIR = ROOT_DIR / "data"
    PROCESSED_DIR = DATA_DIR / "processed"
    MODELS_DIR = ROOT_DIR / "models"
    
    RANDOM_SEED = 42
    
    SENTENCE_TRAIN = PROCESSED_DIR / "sentence_train.parquet"
    SENTENCE_DEV = PROCESSED_DIR / "sentence_dev.parquet"

    # Артефакты и имена моделей для sentence-level признаков
    LABSE_MODEL_NAME = "sentence-transformers/LaBSE"
    RUGPT_MODEL_NAME = "sberbank-ai/rugpt3small_based_on_gpt2"
    SEMANTIC_ENCODER_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    SEMANTIC_PCA_PATH = MODELS_DIR / "semantic_pca.pkl"

Config.MODELS_DIR.mkdir(exist_ok=True)
