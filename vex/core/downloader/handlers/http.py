# pragma: no file cover
import warnings

from vex.core.downloader.handlers.http10 import HTTP10DownloadHandler
from vex.core.downloader.handlers.http11 import (
    HTTP11DownloadHandler as HTTPDownloadHandler,
)
from vex.exceptions import ScrapyDeprecationWarning

warnings.warn(
    "The vex.core.downloader.handlers.http module is deprecated,"
    " please import vex.core.downloader.handlers.http11.HTTP11DownloadHandler"
    " instead of its deprecated alias vex.core.downloader.handlers.http.HTTPDownloadHandler",
    ScrapyDeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "HTTP10DownloadHandler",
    "HTTPDownloadHandler",
]
