from pydantic import BaseModel

class data(BaseModel):
    ph_2: float
    sulf_2: float
    chl_2: float
    Solids: float
    Hardness: float
    Organic_carbon: float
    Conductivity: float
    Trihalomethanes: float
    Turbidity: float
