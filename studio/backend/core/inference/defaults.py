# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Default model lists for inference, split by platform."""

import utils.hardware.hardware as hw

DEFAULT_MODELS_GGUF = [
    "vex/Llama-3.2-1B-Instruct-GGUF",
    "vex/Llama-3.2-3B-Instruct-GGUF",
    "vex/Llama-3.1-8B-Instruct-GGUF",
    "vex/gemma-3-1b-it-GGUF",
    "vex/gemma-3-4b-it-GGUF",
    "vex/Qwen3-4B-GGUF",
]

DEFAULT_MODELS_STANDARD = [
    "vex/Qwen3-4B-Instruct-2507",
    "vex/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "vex/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "vex/Phi-3.5-mini-instruct",
    "vex/Gemma-3-4B-it",
    "vex/Qwen2-VL-2B-Instruct-bnb-4bit",
]


def get_default_models() -> list[str]:
    hw.get_device()  # ensure detect_hardware() has run
    if hw.CHAT_ONLY:
        return list(DEFAULT_MODELS_GGUF)
    return list(DEFAULT_MODELS_STANDARD)
