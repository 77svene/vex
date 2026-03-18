"""
Scrapy Shell

See documentation in docs/topics/shell.rst
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from threading import Thread
from typing import TYPE_CHECKING, Any, Dict, List, Set

from vex.commands import ScrapyCommand
from vex.http import Request
from vex.shell import Shell
from vex.utils.defer import _schedule_coro
from vex.utils.spider import DefaultSpider, spidercls_for_request
from vex.utils.url import guess_scheme

if TYPE_CHECKING:
    from argparse import ArgumentParser, Namespace

    from vex import Spider


class CRDTState:
    """Conflict-free Replicated Data Type for collaborative state synchronization."""
    
    def __init__(self):
        self.breakpoints: Dict[str, Set[str]] = {}  # spider_id -> set of breakpoint URLs
        self.rules: Dict[str, Dict] = {}  # rule_id -> rule data
        self.dom_inspection: Dict[str, Any] = {}  # url -> DOM snapshot
        self.cursors: Dict[str, Dict] = {}  # user_id -> cursor position
        self.version_vector: Dict[str, int] = {}  # user_id -> version
        
    def update_breakpoint(self, spider_id: str, url: str, user_id: str):
        """Add or remove breakpoint using CRDT merge semantics."""
        if spider_id not in self.breakpoints:
            self.breakpoints[spider_id] = set()
        
        if url in self.breakpoints[spider_id]:
            self.breakpoints[spider_id].remove(url)
        else:
            self.breakpoints[spider_id].add(url)
        
        self.version_vector[user_id] = self.version_vector.get(user_id, 0) + 1
        return {"type": "breakpoint_update", "spider_id": spider_id, "url": url, "user": user_id}
    
    def update_rule(self, rule_id: str, rule_data: Dict, user_id: str):
        """Merge rule updates using last-write-wins with vector clocks."""
        current_version = self.version_vector.get(user_id, 0)
        if rule_id in self.rules:
            existing_version = self.rules[rule_id].get("version", 0)
            if current_version <= existing_version:
                return None
        
        rule_data["version"] = current_version + 1
        rule_data["last_modified_by"] = user_id
        self.rules[rule_id] = rule_data
        self.version_vector[user_id] = current_version + 1
        
        return {"type": "rule_update", "rule_id": rule_id, "rule": rule_data, "user": user_id}
    
    def merge(self, other_state: Dict):
        """Merge state from another peer using CRDT merge algorithm."""
        # Merge breakpoints (additive)
        for spider_id, urls in other_state.get("breakpoints", {}).items():
            if spider_id not in self.breakpoints:
                self.breakpoints[spider_id] = set()
            self.breakpoints[spider_id].update(urls)
        
        # Merge rules (last-write-wins based on version vector)
        for rule_id, rule_data in other_state.get("rules", {}).items():
            if rule_id not in self.rules:
                self.rules[rule_id] = rule_data
            else:
                existing_version = self.rules[rule_id].get("version", 0)
                new_version = rule_data.get("version", 0)
                if new_version > existing_version:
                    self.rules[rule_id] = rule_data
        
        # Merge version vectors (take maximum)
        for user_id, version in other_state.get("version_vector", {}).items():
            self.version_vector[user_id] = max(
                self.version_vector.get(user_id, 0), version
            )


class WebSocketCollaborationServer:
    """WebSocket server for real-time collaboration."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set = set()
        self.state = CRDTState()
        self.user_sessions: Dict[str, Dict] = {}
        self.loop = None
        self.server = None
        
    async def register(self, websocket):
        """Register a new client connection."""
        self.clients.add(websocket)
        user_id = str(uuid.uuid4())
        self.user_sessions[str(websocket)] = {
            "id": user_id,
            "websocket": websocket,
            "spider": None,
            "breakpoints": set()
        }
        
        # Send current state to new client
        await websocket.send(json.dumps({
            "type": "state_sync",
            "state": self._serialize_state(),
            "user_id": user_id
        }))
        
        print(f"[Collaboration] New client connected: {user_id}")
        return user_id
    
    async def unregister(self, websocket):
        """Unregister a client connection."""
        self.clients.remove(websocket)
        user_id = self.user_sessions.get(str(websocket), {}).get("id")
        if user_id:
            del self.user_sessions[str(websocket)]
            print(f"[Collaboration] Client disconnected: {user_id}")
    
    async def broadcast(self, message: Dict, exclude: Set = None):
        """Broadcast message to all connected clients."""
        if exclude is None:
            exclude = set()
        
        message_str = json.dumps(message)
        for client in self.clients:
            if client not in exclude:
                try:
                    await client.send(message_str)
                except:
                    await self.unregister(client)
    
    async def handle_message(self, websocket, message: str):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            user_session = self.user_sessions.get(str(websocket))
            
            if not user_session:
                return
            
            if msg_type == "set_breakpoint":
                url = data.get("url")
                spider_id = data.get("spider_id")
                if url and spider_id:
                    update = self.state.update_breakpoint(
                        spider_id, url, user_session["id"]
                    )
                    if update:
                        await self.broadcast(update, exclude={websocket})
            
            elif msg_type == "update_rule":
                rule_id = data.get("rule_id")
                rule_data = data.get("rule")
                if rule_id and rule_data:
                    update = self.state.update_rule(
                        rule_id, rule_data, user_session["id"]
                    )
                    if update:
                        await self.broadcast(update, exclude={websocket})
            
            elif msg_type == "dom_inspection":
                url = data.get("url")
                dom_data = data.get("dom")
                if url and dom_data:
                    self.state.dom_inspection[url] = dom_data
                    await self.broadcast({
                        "type": "dom_update",
                        "url": url,
                        "dom": dom_data,
                        "user": user_session["id"]
                    }, exclude={websocket})
            
            elif msg_type == "cursor_move":
                position = data.get("position")
                if position:
                    self.state.cursors[user_session["id"]] = position
                    await self.broadcast({
                        "type": "cursor_update",
                        "user": user_session["id"],
                        "position": position
                    }, exclude={websocket})
            
            elif msg_type == "state_request":
                await websocket.send(json.dumps({
                    "type": "state_sync",
                    "state": self._serialize_state(),
                    "user_id": user_session["id"]
                }))
                
        except json.JSONDecodeError:
            print(f"[Collaboration] Invalid JSON received: {message}")
    
    def _serialize_state(self) -> Dict:
        """Serialize state for transmission."""
        return {
            "breakpoints": {k: list(v) for k, v in self.state.breakpoints.items()},
            "rules": self.state.rules,
            "dom_inspection": self.state.dom_inspection,
            "cursors": self.state.cursors,
            "version_vector": self.state.version_vector
        }
    
    async def _run_server(self):
        """Run the WebSocket server."""
        try:
            import websockets
        except ImportError:
            print("[Collaboration] websockets library not installed. Install with: pip install websockets")
            return
        
        async def handler(websocket, path):
            user_id = await self.register(websocket)
            try:
                async for message in websocket:
                    await self.handle_message(websocket, message)
            finally:
                await self.unregister(websocket)
        
        self.server = await websockets.serve(handler, self.host, self.port)
        print(f"[Collaboration] WebSocket server started at ws://{self.host}:{self.port}")
        
        # Keep server running
        await asyncio.Future()
    
    def start(self):
        """Start the WebSocket server in a separate thread."""
        def run_server():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._run_server())
        
        server_thread = Thread(target=run_server, daemon=True)
        server_thread.start()
        return server_thread
    
    def stop(self):
        """Stop the WebSocket server."""
        if self.server:
            self.server.close()


class CollaborativeShell(Shell):
    """Shell with real-time collaboration features."""
    
    def __init__(self, crawler, update_vars=None, code=None, collaboration_server=None):
        super().__init__(crawler, update_vars, code)
        self.collaboration_server = collaboration_server
        self.user_id = None
        self.breakpoints = set()
        self._original_fetch = self.fetch
        
    def set_user_id(self, user_id: str):
        """Set the user ID for this shell instance."""
        self.user_id = user_id
    
    def fetch(self, request_or_url, redirect=True, **kwargs):
        """Override fetch to add breakpoint checking and collaboration events."""
        url = request_or_url if isinstance(request_or_url, str) else request_or_url.url
        
        # Check for breakpoint
        if url in self.breakpoints:
            print(f"[Breakpoint] Hit breakpoint at {url}")
            if self.collaboration_server:
                asyncio.run_coroutine_threadsafe(
                    self.collaboration_server.broadcast({
                        "type": "breakpoint_hit",
                        "url": url,
                        "user": self.user_id,
                        "spider": self.spider.name if self.spider else None
                    }),
                    self.collaboration_server.loop
                )
            input("[Breakpoint] Press Enter to continue...")
        
        # Call original fetch
        result = self._original_fetch(request_or_url, redirect, **kwargs)
        
        # Broadcast fetch event
        if self.collaboration_server:
            asyncio.run_coroutine_threadsafe(
                self.collaboration_server.broadcast({
                    "type": "fetch_event",
                    "url": url,
                    "user": self.user_id,
                    "spider": self.spider.name if self.spider else None
                }),
                self.collaboration_server.loop
            )
        
        return result
    
    def inspect_response(self, response, **kwargs):
        """Override inspect_response to broadcast DOM inspection."""
        result = super().inspect_response(response, **kwargs)
        
        if self.collaboration_server and hasattr(response, 'text'):
            asyncio.run_coroutine_threadsafe(
                self.collaboration_server.broadcast({
                    "type": "dom_inspection",
                    "url": response.url,
                    "dom": response.text[:10000],  # Limit DOM size
                    "user": self.user_id
                }),
                self.collaboration_server.loop
            )
        
        return result
    
    def set_breakpoint(self, url: str):
        """Set a breakpoint at the given URL."""
        self.breakpoints.add(url)
        print(f"[Breakpoint] Set breakpoint at {url}")
        
        if self.collaboration_server:
            asyncio.run_coroutine_threadsafe(
                self.collaboration_server.broadcast({
                    "type": "breakpoint_set",
                    "url": url,
                    "user": self.user_id,
                    "spider": self.spider.name if self.spider else None
                }),
                self.collaboration_server.loop
            )
    
    def clear_breakpoint(self, url: str):
        """Clear a breakpoint at the given URL."""
        if url in self.breakpoints:
            self.breakpoints.remove(url)
            print(f"[Breakpoint] Cleared breakpoint at {url}")
            
            if self.collaboration_server:
                asyncio.run_coroutine_threadsafe(
                    self.collaboration_server.broadcast({
                        "type": "breakpoint_cleared",
                        "url": url,
                        "user": self.user_id
                    }),
                    self.collaboration_server.loop
                )
    
    def list_breakpoints(self):
        """List all active breakpoints."""
        if not self.breakpoints:
            print("[Breakpoint] No active breakpoints")
        else:
            print("[Breakpoint] Active breakpoints:")
            for url in sorted(self.breakpoints):
                print(f"  - {url}")


class Command(ScrapyCommand):
    default_settings = {
        "DUPEFILTER_CLASS": "vex.dupefilters.BaseDupeFilter",
        "KEEP_ALIVE": True,
        "LOGSTATS_INTERVAL": 0,
    }

    def __init__(self):
        super().__init__()
        self.collaboration_server = None
        self.server_thread = None

    def syntax(self) -> str:
        return "[url|file]"

    def short_desc(self) -> str:
        return "Interactive scraping console"

    def long_desc(self) -> str:
        return (
            "Interactive console for scraping the given url or file. "
            "Use ./file.html syntax or full path for local file. "
            "Supports real-time collaboration with WebSocket server."
        )

    def add_options(self, parser: ArgumentParser) -> None:
        super().add_options(parser)
        parser.add_argument(
            "-c",
            dest="code",
            help="evaluate the code in the shell, print the result and exit",
        )
        parser.add_argument("--spider", dest="spider", help="use this spider")
        parser.add_argument(
            "--no-redirect",
            dest="no_redirect",
            action="store_true",
            default=False,
            help="do not handle HTTP 3xx status codes and print response as-is",
        )
        parser.add_argument(
            "--collaborate",
            dest="collaborate",
            action="store_true",
            default=False,
            help="enable real-time collaboration mode with WebSocket server",
        )
        parser.add_argument(
            "--collab-port",
            dest="collab_port",
            type=int,
            default=8765,
            help="port for WebSocket collaboration server (default: 8765)",
        )
        parser.add_argument(
            "--collab-host",
            dest="collab_host",
            default="localhost",
            help="host for WebSocket collaboration server (default: localhost)",
        )

    def update_vars(self, vars: dict[str, Any]) -> None:  # noqa: A002
        """You can use this function to update the Scrapy objects that will be
        available in the shell
        """
        # Add collaboration commands to shell vars
        if self.collaboration_server:
            vars['set_breakpoint'] = lambda url: vars['shell'].set_breakpoint(url)
            vars['clear_breakpoint'] = lambda url: vars['shell'].clear_breakpoint(url)
            vars['list_breakpoints'] = lambda: vars['shell'].list_breakpoints()
            vars['collab_users'] = lambda: list(self.collaboration_server.user_sessions.values())

    def run(self, args: list[str], opts: Namespace) -> None:
        url = args[0] if args else None
        if url:
            # first argument may be a local file
            url = guess_scheme(url)

        assert self.crawler_process
        spider_loader = self.crawler_process.spider_loader

        spidercls: type[Spider] = DefaultSpider
        if opts.spider:
            spidercls = spider_loader.load(opts.spider)
        elif url:
            spidercls = spidercls_for_request(
                spider_loader, Request(url), spidercls, log_multiple=True
            )

        # Start collaboration server if enabled
        if opts.collaborate:
            self.collaboration_server = WebSocketCollaborationServer(
                host=opts.collab_host,
                port=opts.collab_port
            )
            self.server_thread = self.collaboration_server.start()
            print(f"[Collaboration] Real-time collaboration enabled")
            print(f"[Collaboration] Connect to ws://{opts.collab_host}:{opts.collab_port}")
            print(f"[Collaboration] Available commands:")
            print(f"  - set_breakpoint(url): Set breakpoint at URL")
            print(f"  - clear_breakpoint(url): Clear breakpoint at URL")
            print(f"  - list_breakpoints(): List all breakpoints")
            print(f"  - collab_users(): List connected users")

        # The crawler is created this way since the Shell manually handles the
        # crawling engine, so the set up in the crawl method won't work
        crawler = self.crawler_process._create_crawler(spidercls)
        crawler._apply_settings()
        # The Shell class needs a persistent engine in the crawler
        crawler.engine = crawler._create_engine()
        _schedule_coro(crawler.engine.start_async(_start_request_processing=False))

        self._start_crawler_thread()

        # Use CollaborativeShell if collaboration is enabled
        if opts.collaborate and self.collaboration_server:
            shell = CollaborativeShell(
                crawler, 
                update_vars=self.update_vars, 
                code=opts.code,
                collaboration_server=self.collaboration_server
            )
            # Generate user ID for this shell instance
            user_id = str(uuid.uuid4())[:8]
            shell.set_user_id(user_id)
            print(f"[Collaboration] Your user ID: {user_id}")
        else:
            shell = Shell(crawler, update_vars=self.update_vars, code=opts.code)
        
        shell.start(url=url, redirect=not opts.no_redirect)

    def _start_crawler_thread(self) -> None:
        assert self.crawler_process
        t = Thread(
            target=self.crawler_process.start,
            kwargs={"stop_after_crawl": False, "install_signal_handlers": False},
        )
        t.daemon = True
        t.start()

    def __del__(self):
        """Cleanup when command is destroyed."""
        if self.collaboration_server:
            self.collaboration_server.stop()