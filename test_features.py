# test_features.py (в корне проекта)
from src.features.extractor import FeatureExtractor

ex = FeatureExtractor()
result = ex.extract(
    src="The government announced new measures on Saturday.",
    mt="Правительство объявило новые меры в субботу."
)
print(result["raw"])
print("vector shape:", result["vector"].shape)

# ex.load_heavy_models()
# result = ex.extract(src, mt)
# print("Признаков:", len(result["vector"])) 