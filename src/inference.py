from preprocess_images import preprocess_images
from cleaning_json import clean_documents_inference
from encode_dataset import encode_inference
from classify_documents import classify_documents
from extract_fields import extract_fields
from inference_utils import save_processing, dvc_push_data

if __name__ == "__main__":
    preprocess_images()
    clean_documents_inference()
    encode_inference()
    classify_documents()
    extract_fields()
    save_processing()
    # dvc_push_data()
