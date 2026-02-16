from fastapi import APIRouter

router = APIRouter()

@router.get("/", tags=["Health"])
def read_root():
    return {"message": "Welcome to the Model Stealing Attack API!"}
