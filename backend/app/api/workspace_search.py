from uuid import UUID

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

from app.db import SessionDep
from app.security.deps import CurrentUser
from app.services.workspace_search import Screen, SearchResults, WorkspaceSearch

router = APIRouter()


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(max_length=200)
    screen: Screen
    # Limit records to one site; omit to search every site (each result names its site).
    site_id: UUID | None = None


@router.post("/api/search", response_model=SearchResults)
async def search_workspace(
    payload: SearchRequest, session: SessionDep, staff: CurrentUser
) -> SearchResults:
    return await WorkspaceSearch(session).search(
        payload.query, payload.screen, staff.is_admin, payload.site_id
    )
