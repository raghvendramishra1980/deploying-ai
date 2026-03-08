from pydantic import BaseModel
class summary(BaseModel):
    Author: str
    Title: str
    Relevance: str
    Summary: str
    Tone: str
    Input_Tokens: int
    Output_Tokens: int