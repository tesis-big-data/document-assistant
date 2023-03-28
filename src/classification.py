from cleaning_json import clean_documents_inference
from encode_dataset import encode_inference
from classify_documents import classify_documents


def classify(uuid_filename):
    clean_documents_inference(uuid_filename)
    encode_inference()
    classify_documents()
    
if __name__ == "__main__":
    classify()