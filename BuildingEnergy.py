from pydantic import BaseModel
# 2. Class which describes Bank Notes measurements
class BuildingEnergy(BaseModel):
    Relative_Compactness: float 
    Surface_Area: float 
    Wall_Area: float 
    Roof_Area: float
    Overall_Height: float
    Orientation: int
    Glazing_Area: float
    Glazing_Area_Distribution: int
