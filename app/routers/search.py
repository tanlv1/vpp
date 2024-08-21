
from fastapi import APIRouter, File, UploadFile, Form
from app.services.search_service import search_faiss_index, search_elasticsearch, combine_search_results
from sqlalchemy.orm import Session
from app.models.database import get_db

router = APIRouter()


@router.post("/search/")
async def search(query_text: str = Form(None), query_image: UploadFile = File(None), top_k: int = 5, db: Session = Depends(get_db)):
    es_results = []
    faiss_results = []

    if query_text:
        es_results = search_elasticsearch(
            query_text, index_name="video_texts", top_k=top_k)

    if query_image:
        image_data = await query_image.read()
        query_image_np = np.frombuffer(image_data, np.uint8)
        faiss_results = search_faiss_index(
            query_image_np, index_path="data/faiss_indexes/video.index", top_k=top_k)

    combined_results = combine_search_results(
        faiss_results, es_results, session=db)

    return {"results": combined_results}
