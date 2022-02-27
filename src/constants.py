FOLDERS_TO_PROCESS = [
    "La Comercial SRL",
    "Modadol",
    "Fernando Garcia",
    "Ayala",
    "Antilur",
    "La Banderita",
    "Los Nietitos",
    "Marioni",
    "Masula",
]

RAW_DOCUMENTS_PATH = "assets/json_documents"

CLEANED_DATASET_PATH = "assets/cleaned_dataset"
CLEANED_DATASET_FILE = f"{CLEANED_DATASET_PATH}/dataset.parquet"

SPLITTED_DATASET_PATH = "assets/splitted_dataset"
TRAIN_FEATURES_FILE = f"{SPLITTED_DATASET_PATH}/train_features.parquet"
TEST_FEATURES_FILE = f"{SPLITTED_DATASET_PATH}/test_features.parquet"
TRAIN_TARGET_FILE = f"{SPLITTED_DATASET_PATH}/train_target.parquet"
TEST_TARGET_FILE = f"{SPLITTED_DATASET_PATH}/test_target.parquet"

ENCODED_DATASET_PATH = "assets/encoded_dataset"
TRAIN_ENCODED_FEATURES_FILE = f"{ENCODED_DATASET_PATH}/train_features.npy"
TEST_ENCODED_FEATURES_FILE = f"{ENCODED_DATASET_PATH}/test_features.npy"
TRAIN_ENCODED_TARGET_FILE = f"{ENCODED_DATASET_PATH}/train_target.parquet"
TEST_ENCODED_TARGET_FILE = f"{ENCODED_DATASET_PATH}/test_target.parquet"

MODELS_PATH = "assets/models"
MODEL_FILE = f"{MODELS_PATH}/model.joblib"

# INFERENCE
INFERENCE_PATH = "assets/inference"

INFERENCE_RAW_IMAGES_PATH = f"{INFERENCE_PATH}/raw_images"
INFERENCE_CURRENT_EXEC_PATH = f"{INFERENCE_PATH}/current_execution"
INFERENCE_CURRENT_EXEC_RAW_IMAGES_PATH = f"{INFERENCE_CURRENT_EXEC_PATH}/raw_images"
INFERENCE_CURRENT_EXEC_ROTATED_IMAGES_PATH = (
    f"{INFERENCE_CURRENT_EXEC_PATH}/raw_images_rotated"
)
INFERENCE_CURRENT_EXEC_CROPPED_IMAGES_PATH = (
    f"{INFERENCE_CURRENT_EXEC_PATH}/images_cropped"
)
INFERENCE_CURRENT_EXEC_JSON_PATH = f"{INFERENCE_CURRENT_EXEC_PATH}/json_files"
INFERENCE_CURRENT_EXEC_CLEANED_DATASET = (
    f"{INFERENCE_CURRENT_EXEC_PATH}/dataset.parquet"
)
INFERENCE_TEMPLATES_PATH = f"{INFERENCE_PATH}/templates"
INFERENCE_RAW_DATA_HISTORIC_PATH = f"{INFERENCE_PATH}/raw_data_historic"
