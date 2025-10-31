from __future__ import annotations

from typing import ClassVar, TypeVar

_T = TypeVar("_T", bound="BaseModel")

class BaseModel:
    model_config: ClassVar[dict[str, object]]
    def __init__(self: BaseModel, **data: object) -> None: ...
    @classmethod
    def model_validate_json(cls: type[_T], s: str) -> _T: ...
    def model_dump(self: BaseModel) -> dict[str, object]: ...
    def model_dump_json(self: BaseModel) -> str: ...

def Field(  # noqa: N802
    *,
    default: object = ...,
    description: str | None = ...,
    ge: object = ...,
    gt: object = ...,
    le: object = ...,
    lt: object = ...,
) -> object: ...
