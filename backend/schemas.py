from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class LogItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    step: int
    action: str
    column: Optional[str] = None
    affected: int = 0
    status: Literal["ok", "failed"] = "ok"
    rows_before: Optional[int] = None
    rows_after: Optional[int] = None
    fill_value: Optional[Any] = None
    error: Optional[str] = None


class ChartItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    title: str
    kind: str
    figure_json: Union[Dict[str, Any], str]


class AnalyzeResponse(BaseModel):
    job_id: str
    filename: str
    profile_before: Dict[str, Any]
    plan: Dict[str, Any]
    log: List[LogItem] = Field(default_factory=list)
    profile_after: Dict[str, Any]
    charts: List[ChartItem] = Field(default_factory=list)
    insights: str = ""
    used_fallback: bool = False
    timings: Dict[str, float] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    download_url: str = ""
    preview_before: Optional[List[Dict[str, Any]]] = None
    preview_after: Optional[List[Dict[str, Any]]] = None