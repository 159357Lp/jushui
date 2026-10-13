"""Config approval endpoints — let the GUI inspect / approve / reject pending
config changes submitted through the ConfigGuardian."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lmloader.infra.guardian import config_guardian

router = APIRouter()


@router.get("/config/pending")
async def list_pending():
    """All pending config-change requests (for GUI polling)."""
    pending = config_guardian.list_pending()
    return {
        "pending": [
            {
                "request_id": r.request_id,
                "endpoint": r.endpoint,
                "description": r.description,
                "parameters": r.parameters,
                "created_at": r.created_at,
            }
            for r in pending
        ]
    }


@router.get("/config/pending/next")
async def next_pending():
    """The next un-presented pending request (for GUI dialog).

    Marks the returned request as presented so the GUI does not see it
    again on the next poll if the user has not yet responded.
    """
    req = config_guardian.next_unpresented()
    if req is None:
        return {"pending": None}
    return {
        "pending": {
            "request_id": req.request_id,
            "endpoint": req.endpoint,
            "description": req.description,
            "parameters": req.parameters,
            "created_at": req.created_at,
        }
    }


@router.get("/config/pending/{request_id}")
async def get_pending(request_id: str):
    req = config_guardian.get_pending(request_id)
    if req is None:
        raise HTTPException(status_code=404, detail="未找到该审批请求")
    return {
        "request_id": req.request_id,
        "endpoint": req.endpoint,
        "description": req.description,
        "parameters": req.parameters,
        "created_at": req.created_at,
    }


class ApproveBody(BaseModel):
    pass


@router.post("/config/approve/{request_id}")
async def approve_change(request_id: str):
    if not config_guardian.approve(request_id):
        raise HTTPException(status_code=404, detail="审批请求不存在或已处理")
    return {"status": "approved", "request_id": request_id}


@router.post("/config/reject/{request_id}")
async def reject_change(request_id: str):
    if not config_guardian.reject(request_id):
        raise HTTPException(status_code=404, detail="审批请求不存在或已处理")
    return {"status": "rejected", "request_id": request_id}
