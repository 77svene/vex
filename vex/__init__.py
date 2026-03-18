"""
Scrapy - a web crawling and web scraping framework written for Python
"""

import pkgutil
import sys
import warnings
import os

# Declare top-level shortcuts
from vex.http import FormRequest, Request
from vex.item import Field, Item
from vex.selector import Selector
from vex.spiders import Spider

# Edge deployment support
from vex.edge import EdgeDeployment, EdgeSpider, EdgeRequest

__all__ = [
    "Field",
    "FormRequest",
    "Item",
    "Request",
    "Selector",
    "Spider",
    "EdgeDeployment",
    "EdgeSpider",
    "EdgeRequest",
    "__version__",
    "version_info",
    "IS_EDGE_ENV",
    "deploy_to_edge",
]


# Scrapy and Twisted versions
__version__ = (pkgutil.get_data(__package__, "VERSION") or b"").decode("ascii").strip()
version_info = tuple(int(v) if v.isdigit() else v for v in __version__.split("."))

# Check if running in edge environment
IS_EDGE_ENV = (
    os.environ.get('SCRAPY_EDGE_MODE') == '1' or
    sys.platform == 'emscripten' or
    'pyodide' in sys.modules or
    hasattr(sys, '_emscripten_info')
)

# Edge deployment configuration
EDGE_CONFIG = {
    'providers': ['cloudflare', 'vercel'],
    'regions': ['us-east', 'eu-west', 'ap-southeast'],
    'cold_start_optimization': True,
    'wasm_compatible': True,
    'auto_deploy': False,
}


def deploy_to_edge(spider_class, config=None):
    """
    Deploy a spider to edge networks for distributed scraping.
    
    Args:
        spider_class: Spider class to deploy
        config: Optional edge deployment configuration
    
    Returns:
        EdgeDeployment instance
    """
    from vex.edge import EdgeDeployment
    
    if config is None:
        config = EDGE_CONFIG.copy()
    
    deployment = EdgeDeployment(spider_class, config)
    deployment.deploy()
    return deployment


# Auto-configure for edge if detected
if IS_EDGE_ENV:
    try:
        from vex import edge_runtime
        edge_runtime.initialize()
    except ImportError:
        pass

# Ignore noisy twisted deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="twisted")

# Edge-specific imports when in WASM environment
if IS_EDGE_ENV:
    try:
        from vex.http import EdgeResponse
        from vex.core import EdgeEngine
        __all__.extend(["EdgeResponse", "EdgeEngine"])
    except ImportError:
        pass

del pkgutil
del sys
del warnings
del os