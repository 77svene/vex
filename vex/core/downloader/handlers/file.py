from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from w3lib.url import file_uri_to_path

from vex.core.downloader.handlers.base import BaseDownloadHandler
from vex.responsetypes import responsetypes

if TYPE_CHECKING:
    from vex import Request
    from vex.http import Response


class FileDownloadHandler(BaseDownloadHandler):
    async def download_request(self, request: Request) -> Response:
        filepath = file_uri_to_path(request.url)
        body = Path(filepath).read_bytes()  # noqa: ASYNC240
        respcls = responsetypes.from_args(filename=filepath, body=body)
        return respcls(url=request.url, body=body)
