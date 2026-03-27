"""Documents endpoint — POST /documents."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile
from models.documents import IngestResponse

from services.documents import ingest_document

router = APIRouter(tags=["documents"])


@router.post("/documents", response_model=IngestResponse)
async def upload_document(file: UploadFile, req: Request) -> IngestResponse:
    if not file.filename or not file.filename.endswith(".md"):
        raise HTTPException(
            status_code=400,
            detail="Only .md files are supported.",
        )

    # Write upload to a temp file so load_file can read it from disk
    with tempfile.NamedTemporaryFile(
        suffix=".md", delete=False
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        response = ingest_document(
            path=tmp_path,
            embeddings=req.app.state.embeddings,
            session=req.app.state.session,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return response