from pydantic import BaseModel

class Status(BaseModel):
    registr: str
    value: str

class CalibrateModel(BaseModel):
    go_to_cube: bool