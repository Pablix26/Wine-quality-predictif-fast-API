from pydantic import BaseModel, Field

class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., gt=0)
    volatile_acidity: float = Field(..., ge=0)
    citric_acid: float = Field(..., ge=0)
    residual_sugar: float = Field(..., ge=0)
    chlorides: float = Field(..., ge=0)
    free_sulfur_dioxide: float = Field(..., ge=0)
    total_sulfur_dioxide: float = Field(..., ge=0)
    density: float = Field(..., gt=0)
    pH: float = Field(..., gt=0)
    sulphates: float = Field(..., ge=0)
    alcohol: float = Field(..., ge=0)

class NewWineData(WineFeatures):
    quality: int = Field(..., ge=0, le=10)
    Id: int = None  # Optionnel
