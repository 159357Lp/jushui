"""Runtime event log routes."""

from fastapi import APIRouter, Query

from lmloader.infra.events import event_log

router = APIRouter()


@router.get("/events")
async def get_events(
    since_id: int = Query(0, ge=0),
    limit: int = Query(500, ge=1, le=1000),
    level: str = Query("all", pattern="^(all|info|warning|error)$"),
):
    events = event_log.get(since_id=since_id, limit=limit, level=level)
    last_id = events[-1]["id"] if events else since_id
    return {"events": events, "last_id": last_id}


@router.post("/events/clear")
async def clear_events():
    event_log.clear()
    return {"status": "cleared"}
