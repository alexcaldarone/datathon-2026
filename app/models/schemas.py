from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.models.valid_values import (
    Canton,
    Feature,
    ObjectCategory,
    OfferType,
    VALID_CITIES,
    VALID_POSTAL_CODES,
)


class HardFilters(BaseModel):
    city: list[str] | None = None
    postal_code: list[str] | None = None
    canton: Canton | None = None
    min_price: int | None = Field(default=None, ge=0)
    max_price: int | None = Field(default=None, ge=0)
    min_rooms: float | None = Field(default=None, ge=0)
    max_rooms: float | None = Field(default=None, ge=0)
    latitude: float | None = None
    longitude: float | None = None
    radius_km: float | None = Field(default=None, ge=0)
    features: list[Feature] | None = None
    offer_type: OfferType | None = None
    object_category: list[ObjectCategory] | None = None
    limit: int = Field(default=20, ge=1, le=500)
    offset: int = Field(default=0, ge=0)
    sort_by: Literal["price_asc", "price_desc", "rooms_asc", "rooms_desc"] | None = None

    @field_validator("city", mode="before")
    @classmethod
    def validate_cities(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        invalid = [c for c in v if c not in VALID_CITIES]
        if invalid:
            raise ValueError(f"Unknown cities: {invalid}")
        return v

    @field_validator("postal_code", mode="before")
    @classmethod
    def validate_postal_codes(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        invalid = [p for p in v if p not in VALID_POSTAL_CODES]
        if invalid:
            raise ValueError(f"Unknown postal codes: {invalid}")
        return v


class ListingsQueryRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=25, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class ListingsSearchRequest(BaseModel):
    hard_filters: HardFilters | None = None


class ListingData(BaseModel):
    id: str
    title: str
    description: str | None = None
    street: str | None = None
    city: str | None = None
    postal_code: str | None = None
    canton: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    price_chf: int | None = None
    rooms: float | None = None
    living_area_sqm: int | None = None
    available_from: str | None = None
    image_urls: list[str] | None = None
    hero_image_url: str | None = None
    original_listing_url: str | None = None
    features: list[str] = Field(default_factory=list)
    offer_type: str | None = None
    object_category: str | None = None
    object_type: str | None = None


class RankedListingResult(BaseModel):
    listing_id: str
    score: float
    reason: str
    listing: ListingData


class ListingsResponse(BaseModel):
    listings: list[RankedListingResult]
    meta: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    is_valid: bool
    reason: str = ""
    questions: list[str] = []


class SoftPreference(BaseModel):
    dimension: str = Field(description="Anchor dimension name from the provided list")
    weight: float = Field(ge=0.0, le=1.0, description="Importance: 1.0=must, 0.7=prefer, 0.4=nice-to-have")


class SoftFacts(BaseModel):
    preferences: list[SoftPreference] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
