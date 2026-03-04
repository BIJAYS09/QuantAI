from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    history: list = []


class QuickRequest(BaseModel):
    symbol: str
    asset_type: str = "stock"
