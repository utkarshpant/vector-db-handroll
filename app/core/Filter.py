from typing import Any, Optional
from pydantic import BaseModel, RootModel, field_validator, model_validator


class Condition(BaseModel):
    """
    Represents a set of filter conditions for a field.
    Attributes:
        eq (Optional[Any]): Value that the field must be equal to.
        ne (Optional[Any]): Value that the field must not be equal to.
        gt (Optional[Any]): Value that the field must be greater than.
        gte (Optional[Any]): Value that the field must be greater than or equal to.
        lt (Optional[Any]): Value that the field must be less than.
        lte (Optional[Any]): Value that the field must be less than or equal to.
        contains (Optional[str]): Substring that the field must contain (for string fields).
    Validators:
        - Ensures that condition values are of type str, int, float, or bool.
        - Ensures that exactly one operator is provided per field among 'eq', 'contains', 'gte', and 'lte'.
    """
    eq: Optional[Any] = None
    """
    eq (Optional[Any]): Value that the field must be equal to.
    """
    ne: Optional[Any] = None
    """
    ne (Optional[Any]): Value that the field must not be equal to.
    """
    gt: Optional[Any] = None
    """
    gt (Optional[Any]): Value that the field must be greater than.
    """
    gte: Optional[Any] = None
    """
    gte (Optional[Any]): Value that the field must be greater than or equal to.
    """
    lt: Optional[Any] = None
    """
    lt (Optional[Any]): Value that the field must be less than.
    """
    lte: Optional[Any] = None
    """
    lte (Optional[Any]): Value that the field must be less than or equal to.
    """
    contains: Optional[str] = None
    """
    contains (Optional[str]): Substring that the field must contain (for string fields).
    """

    @field_validator("eq", "ne", "gt", "gte", "lt", "lte", mode="before")
    def validate_conditions(cls, value: Any) -> Any:
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise ValueError(
                "Condition values must be of type str, int, float, or bool")
        return value

    @model_validator(mode="after")
    def validate_single_operator(cls, condition):
        """Ensure exactly one of 'eq', 'contains', 'gte', or 'lte' is set."""
        operators_present = [condition.eq, condition.contains, condition.gte, condition.lte, condition.gt, condition.lt, condition.ne]
        if sum(op is not None for op in operators_present) != 1:
            raise ValueError("Provide exactly one of: eq, contains, gte, lte")
        return condition

class Filter(RootModel[dict[str, Condition]]):
    pass