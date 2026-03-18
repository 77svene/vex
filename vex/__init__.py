import os
from typing import TYPE_CHECKING

from vex.logging_config import setup_logging

# Only set up logging if not in MCP mode or if explicitly requested
if os.environ.get('BROWSER_USE_SETUP_LOGGING', 'true').lower() != 'false':
	from vex.config import CONFIG

	# Get log file paths from config/environment
	debug_log_file = getattr(CONFIG, 'BROWSER_USE_DEBUG_LOG_FILE', None)
	info_log_file = getattr(CONFIG, 'BROWSER_USE_INFO_LOG_FILE', None)

	# Set up logging with file handlers if specified
	logger = setup_logging(debug_log_file=debug_log_file, info_log_file=info_log_file)
else:
	import logging

	logger = logging.getLogger('vex')

# Monkeypatch BaseSubprocessTransport.__del__ to handle closed event loops gracefully
from asyncio import base_subprocess

_original_del = base_subprocess.BaseSubprocessTransport.__del__


def _patched_del(self):
	"""Patched __del__ that handles closed event loops without throwing noisy red-herring errors like RuntimeError: Event loop is closed"""
	try:
		# Check if the event loop is closed before calling the original
		if hasattr(self, '_loop') and self._loop and self._loop.is_closed():
			# Event loop is closed, skip cleanup that requires the loop
			return
		_original_del(self)
	except RuntimeError as e:
		if 'Event loop is closed' in str(e):
			# Silently ignore this specific error
			pass
		else:
			raise


base_subprocess.BaseSubprocessTransport.__del__ = _patched_del


# Type stubs for lazy imports - fixes linter warnings
if TYPE_CHECKING:
	from vex.agent.prompts import SystemPrompt
	from vex.agent.service import Agent

	# from vex.agent.service import Agent
	from vex.agent.views import ActionModel, ActionResult, AgentHistoryList
	from vex.browser import BrowserProfile, BrowserSession
	from vex.browser import BrowserSession as Browser
	from vex.code_use.service import CodeAgent
	from vex.dom.service import DomService
	from vex.llm import models
	from vex.llm.anthropic.chat import ChatAnthropic
	from vex.llm.azure.chat import ChatAzureOpenAI
	from vex.llm.vex.chat import ChatBrowserUse
	from vex.llm.google.chat import ChatGoogle
	from vex.llm.groq.chat import ChatGroq
	from vex.llm.litellm.chat import ChatLiteLLM
	from vex.llm.mistral.chat import ChatMistral
	from vex.llm.oci_raw.chat import ChatOCIRaw
	from vex.llm.ollama.chat import ChatOllama
	from vex.llm.openai.chat import ChatOpenAI
	from vex.llm.vercel.chat import ChatVercel
	from vex.sandbox import sandbox
	from vex.tools.service import Controller, Tools

	# Lazy imports mapping - only import when actually accessed
_LAZY_IMPORTS = {
	# Agent service (heavy due to dependencies)
	# 'Agent': ('vex.agent.service', 'Agent'),
	# Code-use agent (Jupyter notebook-like execution)
	'CodeAgent': ('vex.code_use.service', 'CodeAgent'),
	'Agent': ('vex.agent.service', 'Agent'),
	# System prompt (moderate weight due to agent.views imports)
	'SystemPrompt': ('vex.agent.prompts', 'SystemPrompt'),
	# Agent views (very heavy - over 1 second!)
	'ActionModel': ('vex.agent.views', 'ActionModel'),
	'ActionResult': ('vex.agent.views', 'ActionResult'),
	'AgentHistoryList': ('vex.agent.views', 'AgentHistoryList'),
	'BrowserSession': ('vex.browser', 'BrowserSession'),
	'Browser': ('vex.browser', 'BrowserSession'),  # Alias for BrowserSession
	'BrowserProfile': ('vex.browser', 'BrowserProfile'),
	# Tools (moderate weight)
	'Tools': ('vex.tools.service', 'Tools'),
	'Controller': ('vex.tools.service', 'Controller'),  # alias
	# DOM service (moderate weight)
	'DomService': ('vex.dom.service', 'DomService'),
	# Chat models (very heavy imports)
	'ChatOpenAI': ('vex.llm.openai.chat', 'ChatOpenAI'),
	'ChatGoogle': ('vex.llm.google.chat', 'ChatGoogle'),
	'ChatAnthropic': ('vex.llm.anthropic.chat', 'ChatAnthropic'),
	'ChatBrowserUse': ('vex.llm.vex.chat', 'ChatBrowserUse'),
	'ChatGroq': ('vex.llm.groq.chat', 'ChatGroq'),
	'ChatLiteLLM': ('vex.llm.litellm.chat', 'ChatLiteLLM'),
	'ChatMistral': ('vex.llm.mistral.chat', 'ChatMistral'),
	'ChatAzureOpenAI': ('vex.llm.azure.chat', 'ChatAzureOpenAI'),
	'ChatOCIRaw': ('vex.llm.oci_raw.chat', 'ChatOCIRaw'),
	'ChatOllama': ('vex.llm.ollama.chat', 'ChatOllama'),
	'ChatVercel': ('vex.llm.vercel.chat', 'ChatVercel'),
	# LLM models module
	'models': ('vex.llm.models', None),
	# Sandbox execution
	'sandbox': ('vex.sandbox', 'sandbox'),
}


def __getattr__(name: str):
	"""Lazy import mechanism - only import modules when they're actually accessed."""
	if name in _LAZY_IMPORTS:
		module_path, attr_name = _LAZY_IMPORTS[name]
		try:
			from importlib import import_module

			module = import_module(module_path)
			if attr_name is None:
				# For modules like 'models', return the module itself
				attr = module
			else:
				attr = getattr(module, attr_name)
			# Cache the imported attribute in the module's globals
			globals()[name] = attr
			return attr
		except ImportError as e:
			raise ImportError(f'Failed to import {name} from {module_path}: {e}') from e

	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
	'Agent',
	'CodeAgent',
	# 'CodeAgent',
	'BrowserSession',
	'Browser',  # Alias for BrowserSession
	'BrowserProfile',
	'Controller',
	'DomService',
	'SystemPrompt',
	'ActionResult',
	'ActionModel',
	'AgentHistoryList',
	# Chat models
	'ChatOpenAI',
	'ChatGoogle',
	'ChatAnthropic',
	'ChatBrowserUse',
	'ChatGroq',
	'ChatLiteLLM',
	'ChatMistral',
	'ChatAzureOpenAI',
	'ChatOCIRaw',
	'ChatOllama',
	'ChatVercel',
	'Tools',
	'Controller',
	# LLM models module
	'models',
	# Sandbox execution
	'sandbox',
]
