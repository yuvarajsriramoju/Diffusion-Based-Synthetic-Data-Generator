from pydantic import BaseModel

class GenRequest(BaseModel):
    count: int = 32  # how many images to sample

class GenResponse(BaseModel):
    images: list[str]  # base64-encoded PNGs
