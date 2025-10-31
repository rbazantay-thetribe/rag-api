from typing import List, Dict, Any, Optional, Union
import os
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
from app.services.embedding import embed_texts


router = APIRouter(tags=["qdrant"], prefix="")


class CreateCollectionRequest(BaseModel):
    collection_name: str
    vector_size: int = Field(..., description="Embedding dimension for vectors")
    distance: str = Field(
        default="COSINE",
        description="Distance metric: COSINE, DOT, or EUCLID",
    )


class CreateCollectionResponse(BaseModel):
    message: str


@router.post(
    "/collections/create",
    summary="Create a collection",
    description="Create a collection in Qdrant",
    response_model=CreateCollectionResponse,
)
def create_collection(req: CreateCollectionRequest) -> CreateCollectionResponse:
    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    try:
        client = QdrantClient(url=qdrant_url)
        try:
            distance_enum = qmodels.Distance[req.distance.upper()]
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid distance: {req.distance}")

        client.create_collection(
            collection_name=req.collection_name,
            vectors_config=qmodels.VectorParams(
                size=req.vector_size,
                distance=distance_enum,
            ),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qdrant error: {exc}")
    return CreateCollectionResponse(message="Collection created")


class UploadCollectionRequest(BaseModel):
    collection_name: str
    vectors: List[List[float]]
    payloads: List[Dict[str, Any]]


class UploadCollectionResponse(BaseModel):
    message: str


@router.put(
    "/collections/{collection_name}/upload",
    summary="Upload a collection",
    description="Upload a collection in Qdrant",
    response_model=UploadCollectionResponse,
)
def upload_collection(req: UploadCollectionRequest) -> UploadCollectionResponse:
    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    try:
        client = QdrantClient(url=qdrant_url)
        if len(req.vectors) != len(req.payloads):
            raise HTTPException(status_code=400, detail="vectors and payloads must have same length")

        points = [
            qmodels.PointStruct(id=i, vector=vector, payload=req.payloads[i])
            for i, vector in enumerate(req.vectors)
        ]

        client.upsert(collection_name=req.collection_name, points=points, wait=True)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qdrant error: {exc}")
    return UploadCollectionResponse(message="Collection uploaded")


class AddPointRequest(BaseModel):
    vector: List[float]
    payload: Dict[str, Any] = {}
    id: Optional[Union[int, str]] = None


class AddPointResponse(BaseModel):
    message: str
    point_id: Union[int, str]


@router.post(
    "/collections/{collection_name}/points",
    summary="Add a single point",
    description="Insert a single point into a collection",
    response_model=AddPointResponse,
)
def add_point(collection_name: str, req: AddPointRequest) -> AddPointResponse:
    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    try:
        client = QdrantClient(url=qdrant_url)
        point_id: Union[int, str] = req.id if req.id is not None else str(uuid.uuid4())
        point = qmodels.PointStruct(id=point_id, vector=req.vector, payload=req.payload)
        client.upsert(collection_name=collection_name, points=[point], wait=True)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qdrant error: {exc}")
    return AddPointResponse(message="Point added", point_id=point_id)


class SearchCollectionRequest(BaseModel):
    query_text: Optional[str] = None
    query_vector: Optional[List[float]] = None
    k: int = 10
    model: str = "sentence-transformers/all-MiniLM-L6-v2"


class SearchCollectionResponse(BaseModel):
    message: str
    results: List[Dict[str, Any]]


@router.post(
    "/collections/{collection_name}/search",
    summary="Search a collection",
    description="Search a collection in Qdrant",
    response_model=SearchCollectionResponse,
)
def search_collection(collection_name: str, req: SearchCollectionRequest) -> SearchCollectionResponse:
    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    try:
        client = QdrantClient(url=qdrant_url)

        # Build query vector
        if req.query_vector is not None:
            query_vector: List[float] = req.query_vector
        elif req.query_text is not None and req.query_text.strip():
            emb = embed_texts([req.query_text], model=req.model)
            vectors = emb.get("vectors", [])
            if not vectors or not vectors[0]:
                raise HTTPException(status_code=400, detail="Empty embedding for query_text")
            query_vector = vectors[0]
        else:
            raise HTTPException(status_code=400, detail="Provide query_vector or query_text")

        scored_points = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=req.k,
            with_payload=True,
        )

        results: List[Dict[str, Any]] = [
            {"id": p.id, "score": p.score, "payload": getattr(p, "payload", None)} for p in scored_points
        ]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qdrant error: {exc}")
    return SearchCollectionResponse(message="Search results", results=results)


class DeleteCollectionRequest(BaseModel):
    collection_name: str


class DeleteCollectionResponse(BaseModel):
    message: str


@router.delete(
    "/collections/{collection_name}",
    summary="Delete a collection",
    description="Delete a collection in Qdrant",
    response_model=DeleteCollectionResponse,
)
def delete_collection(collection_name: str) -> DeleteCollectionResponse:
    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    try:
        client = QdrantClient(url=qdrant_url)
        client.delete_collection(collection_name)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qdrant error: {exc}")
    return DeleteCollectionResponse(message="Collection deleted")




class ListCollectionsResponse(BaseModel):
    message: str
    collections: List[str]


@router.get(
    "/collections",
    summary="List collections",
    description="List collections in Qdrant",
    response_model=ListCollectionsResponse,
)
def list_collections() -> ListCollectionsResponse:
    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    try:
        client = QdrantClient(url=qdrant_url)
        resp = client.get_collections()
        collections = [c.name for c in (resp.collections or [])]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qdrant error: {exc}")
    return ListCollectionsResponse(message="Collections listed", collections=collections)


class GetCollectionRequest(BaseModel):
    collection_name: str

class GetCollectionResponse(BaseModel):
    message: str
    collection: Dict[str, Any]

@router.get(
    "/collections/{collection_name}",
    summary="Get a collection",
    description="Get a collection in Qdrant",
    response_model=GetCollectionResponse,
)
def get_collection(collection_name: str) -> GetCollectionResponse:
    qdrant_url = os.environ.get("QDRANT_URL", "http://qdrant:6333")
    try:
        client = QdrantClient(url=qdrant_url)
        collection = client.get_collection(collection_name)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Qdrant error: {exc}")
    # Ensure response is JSON-serializable
    if hasattr(collection, "dict"):
        collection_data = collection.dict()
    elif hasattr(collection, "model_dump"):
        collection_data = collection.model_dump()
    else:
        collection_data = collection  # assume already serializable
    return GetCollectionResponse(message="Collection retrieved", collection=collection_data)