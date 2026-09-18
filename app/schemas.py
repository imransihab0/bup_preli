"""Request/response models mirroring Problem Statement S07 and S10 exactly."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]

BatteryAction = Literal["charge", "discharge", "idle"]

NonNegFloat = Annotated[float, Field(ge=0)]


# --------------------------------------------------------------------------- #
# Request (S07)
# --------------------------------------------------------------------------- #
class HourInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hour: int = Field(ge=0, le=23)
    demand_kwh: NonNegFloat
    solar_kwh: NonNegFloat
    tariff_bdt_per_kwh: float


class BatteryInput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    capacity_kwh: NonNegFloat
    initial_energy_kwh: NonNegFloat
    minimum_energy_kwh: NonNegFloat
    max_charge_kwh_per_hour: NonNegFloat
    max_discharge_kwh_per_hour: NonNegFloat

    @model_validator(mode="after")
    def _coherent(self) -> "BatteryInput":
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError("minimum_energy_kwh exceeds capacity_kwh")
        if not (self.minimum_energy_kwh <= self.initial_energy_kwh <= self.capacity_kwh):
            raise ValueError("initial_energy_kwh outside [minimum_energy_kwh, capacity_kwh]")
        return self


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    scenario_id: str = Field(min_length=1)
    operator_notes: list[str] = Field(min_length=1, max_length=3)
    hours: list[HourInput] = Field(min_length=24, max_length=24)
    battery: BatteryInput

    @field_validator("operator_notes")
    @classmethod
    def _notes_non_empty(cls, notes: list[str]) -> list[str]:
        if any(not note or not note.strip() for note in notes):
            raise ValueError("operator_notes entries must be non-empty strings")
        return notes

    @field_validator("hours")
    @classmethod
    def _hours_complete(cls, hours: list[HourInput]) -> list[HourInput]:
        if {h.hour for h in hours} != set(range(24)):
            raise ValueError("hours must contain exactly one entry for each hour 0..23")
        return hours

    def hours_sorted(self) -> list[HourInput]:
        return sorted(self.hours, key=lambda h: h.hour)


# --------------------------------------------------------------------------- #
# Response (S10)
# --------------------------------------------------------------------------- #
class StructuredAdjustment(BaseModel):
    """Only the keys required by the directive's shape are serialized (S04)."""

    model_config = ConfigDict(extra="forbid")

    hours: list[int]
    factor: float | None = None
    minimum_energy_kwh: float | None = None
    max_grid_kwh: float | None = None

    def serializable(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class DirectiveInterpretation(BaseModel):
    note_index: int
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: dict[str, Any] | None
    explanation: str


class HourPlan(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: BatteryAction
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: list[DirectiveInterpretation]
    hourly_plan: list[HourPlan]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str
