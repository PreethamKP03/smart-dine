"""Optional FastAPI/dashboard adapter for the same engine used by Streamlit."""
from contextlib import asynccontextmanager
from typing import Optional, Literal
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from .paths import BASE_DIR, DATA_DIR, METADATA_PATH, MODEL_PATH, OUTPUT_DIR
from .metadata import load_metadata
from .recommendation import SmartDineRecommender
from .group_preferences import MemberPreference as Member, GroupPreferences

STATE = {'engine': None, 'error': None}


@asynccontextmanager
async def lifespan(app):
    try:
        STATE['engine'] = SmartDineRecommender()
        STATE['error'] = None
    except Exception:
        logging.exception('SmartDine engine initialization failed')
        STATE['engine'] = None
        STATE['error'] = 'Required artifacts are unavailable or incompatible. Regenerate pipeline artifacts.'
    yield
    STATE['engine'] = None


app = FastAPI(title='SmartDine Recommendation API', version='2.0', lifespan=lifespan)


class MemberPreference(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra='forbid')
    name: str = 'Member'
    preferred_cuisines: list[str] = Field(default_factory=list)
    budget: float = Field(default=800, gt=0)
    dietary_preference: Literal['vegetarian', 'vegan', 'non-vegetarian', 'no preference'] = 'no preference'
    preferred_location: Optional[str] = None
    preferred_restaurant_type: Optional[str] = None
    min_rating: float = Field(default=3.5, ge=0, le=5)
    cuisine_importance: float = Field(default=1, ge=0)
    budget_importance: float = Field(default=1, ge=0)
    dietary_importance: float = Field(default=1, ge=0)
    restaurant_type_importance: float = Field(default=1, ge=0)
    location_importance: float = Field(default=1, ge=0)
    requires_table_booking: bool = False
    prefers_online_order: bool = False


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra='forbid')
    members: list[MemberPreference] = Field(min_length=1, max_length=50)
    top_n: int = Field(default=10, ge=1, le=50)
    filter_location: Optional[str] = None
    filter_cuisine: Optional[str] = None
    max_budget_limit: Optional[float] = Field(default=None, gt=0)
    min_rating: Optional[float] = Field(default=None, ge=0, le=5)
    require_booking: Optional[bool] = None
    require_online: Optional[bool] = None
    time_of_day: str = 'dinner'
    day_type: str = 'weekend'


def engine():
    if STATE['engine'] is None:
        raise HTTPException(503, STATE['error'] or 'Engine not initialized.')
    return STATE['engine']


@app.get('/api/health')
def health():
    current = STATE['engine']
    metadata = load_metadata()
    payload = {'status': 'healthy' if current is not None else 'unavailable',
               'model': type(current.model).__name__ if current else None,
               'total_restaurants': len(current.df) if current else 0,
               'features': current.feature_columns if current else [],
               'metadata_status': metadata['status'],
               'model_r2_score': metadata['metrics']['R2'] if metadata['status'] == 'available' else None}
    return JSONResponse(payload, status_code=200 if current else 503)


@app.get('/api/metadata')
def get_metadata():
    current = engine()
    metadata = load_metadata()
    comparison = []
    if metadata['status'] == 'available':
        try:
            comparison = pd.read_csv(OUTPUT_DIR / 'model_comparison.csv').to_dict('records')
        except (OSError, ValueError):
            pass
    return {'cuisines': sorted(set.union(set(), *current.builder.cuisines)),
            'locations': sorted(set(l for l in current.locations_list if l)),
            'restaurant_types': sorted(set.union(set(), *current.builder.types)),
            'total_restaurants': len(current.df), 'model_metadata': metadata,
            'feature_coefficients': dict(zip(current.feature_columns, current.model.coef_.tolist())) if hasattr(current.model, 'coef_') else {},
            'model_comparison': comparison, 'comparison_scope': 'validation, not test'}


@app.post('/api/recommend')
def recommend(req: RecommendationRequest):
    members = []
    for item in req.members:
        values = item.model_dump()
        values['requires_booking'] = values.pop('requires_table_booking')
        values['prefers_online'] = values.pop('prefers_online_order')
        members.append(Member(**values))
    try:
        return engine().recommend(GroupPreferences(members, req.time_of_day, req.day_type),
            top_n=req.top_n, filter_location=req.filter_location, filter_cuisine=req.filter_cuisine,
            max_budget=req.max_budget_limit, min_rating=req.min_rating,
            require_booking=req.require_booking, require_online=req.require_online)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/presets")
def get_presets():
    """Returns illustrative demo groups + quick demo personas."""
    presets = [
        {
            "id": "G001",
            "name": "Demo G001: Healthy & Mixed Diets",
            "description": "5 members in Electronic City with mixed vegan, vegetarian & non-vegetarian diets.",
            "members": [
                {"name": "Ananya", "preferred_cuisines": ["healthy food", "sandwich"], "budget": 600, "dietary_preference": "vegan", "preferred_location": "electronic city", "preferred_restaurant_type": "cafe"},
                {"name": "Rahul", "preferred_cuisines": ["burger", "fast food"], "budget": 800, "dietary_preference": "non-vegetarian", "preferred_location": "electronic city", "preferred_restaurant_type": "food court"},
                {"name": "Sneha", "preferred_cuisines": ["healthy food", "salad"], "budget": 700, "dietary_preference": "vegetarian", "preferred_location": "malleshwaram", "preferred_restaurant_type": "cafe"},
                {"name": "Vikram", "preferred_cuisines": ["seafood", "fast food"], "budget": 900, "dietary_preference": "non-vegetarian", "preferred_location": "bannerghatta road", "preferred_restaurant_type": "bar"},
                {"name": "Pooja", "preferred_cuisines": ["healthy food", "dessert"], "budget": 650, "dietary_preference": "vegetarian", "preferred_location": "electronic city", "preferred_restaurant_type": "dessert parlor"}
            ]
        },
        {
            "id": "G002",
            "name": "Demo G002: Whitefield Asian & Momos",
            "description": "4 members in Whitefield loving Asian cuisine, Momos, and Seafood.",
            "members": [
                {"name": "Karthik", "preferred_cuisines": ["momos", "asian"], "budget": 850, "dietary_preference": "non-vegetarian", "preferred_location": "whitefield", "preferred_restaurant_type": "takeaway"},
                {"name": "Meera", "preferred_cuisines": ["seafood", "asian"], "budget": 1000, "dietary_preference": "non-vegetarian", "preferred_location": "whitefield", "preferred_restaurant_type": "delivery"},
                {"name": "Arjun", "preferred_cuisines": ["ice cream", "momos"], "budget": 900, "dietary_preference": "vegetarian", "preferred_location": "whitefield", "preferred_restaurant_type": "bakery"},
                {"name": "Divya", "preferred_cuisines": ["american", "asian"], "budget": 950, "dietary_preference": "vegetarian", "preferred_location": "malleshwaram", "preferred_restaurant_type": "sweet shop"}
            ]
        },
        {
            "id": "G005",
            "name": "Demo G005: Strict Veg Budget Hangout",
            "description": "3 vegetarians looking for affordable dining under ₹400 in Whitefield/Kalyan Nagar.",
            "members": [
                {"name": "Ramesh", "preferred_cuisines": ["fast food", "salad"], "budget": 350, "dietary_preference": "vegetarian", "preferred_location": "whitefield", "preferred_restaurant_type": "beverage shop"},
                {"name": "Suresh", "preferred_cuisines": ["american", "fast food"], "budget": 380, "dietary_preference": "vegetarian", "preferred_location": "whitefield", "preferred_restaurant_type": "bakery"},
                {"name": "Mahesh", "preferred_cuisines": ["salad", "beverages"], "budget": 340, "dietary_preference": "vegetarian", "preferred_location": "kalyan nagar", "preferred_restaurant_type": "fine dining"}
            ]
        },
        {
            "id": "PRESET_TECH_LUNCH",
            "name": "Tech Team Koramangala Hangout",
            "description": "4 developers looking for Biryani, North Indian & Continental lunch in Koramangala.",
            "members": [
                {"name": "Preetham", "preferred_cuisines": ["biryani", "north indian"], "budget": 1000, "dietary_preference": "non-vegetarian", "preferred_location": "koramangala 5th block", "preferred_restaurant_type": "casual dining"},
                {"name": "Deepak", "preferred_cuisines": ["continental", "pizza"], "budget": 1200, "dietary_preference": "non-vegetarian", "preferred_location": "koramangala 5th block", "preferred_restaurant_type": "cafe"},
                {"name": "Tanvi", "preferred_cuisines": ["north indian", "chinese"], "budget": 900, "dietary_preference": "vegetarian", "preferred_location": "koramangala 5th block", "preferred_restaurant_type": "casual dining"},
                {"name": "Siddharth", "preferred_cuisines": ["biryani", "fast food"], "budget": 800, "dietary_preference": "non-vegetarian", "preferred_location": "koramangala 5th block", "preferred_restaurant_type": "quick bites"}
            ]
        }
    ]
    return presets



FRONTEND_DIR = BASE_DIR / 'frontend'
if FRONTEND_DIR.is_dir():
    app.mount('/static', StaticFiles(directory=str(FRONTEND_DIR)), name='static')

    @app.get('/')
    def index():
        return FileResponse(FRONTEND_DIR / 'index.html')
