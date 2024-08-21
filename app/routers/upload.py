from fastapi import APIRouter, UploadFile, File
from app.services.feature_extraction import extract_frames, extract_features
from app.services.indexing import create_faiss_index
import os

router = APIRouter()


@router.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    file_location = f"data/videos/{file.filename}"

    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    # Step 1: Extract frames from the video
    frames = extract_frames(file_location)

    # Step 2: Extract features from the frames
    features = extract_features(frames)

    # Step 3: Index the extracted features using Faiss
    index_path = f"data/faiss_indexes/{file.filename}.index"
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    create_faiss_index(features, index_path)

    return {"info": f"File '{file.filename}' processed and indexed.", "index_path": index_path}
