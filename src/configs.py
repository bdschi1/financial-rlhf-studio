from dataclasses import dataclass
from typing import Optional, List

PROVIDERS = ["anthropic", "gemini", "openai"]

AVAILABLE_MODELS = {
    "anthropic": ["claude-sonnet-4-20250514", "claude-haiku-4-20250414", "claude-3-5-sonnet-20241022"],
    "gemini": ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-pro"],
    "openai": ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
}

ALL_MODELS = [m for models in AVAILABLE_MODELS.values() for m in models]

DEFAULT_SYSTEM_PROMPTS = {
    "Junior Analyst": "You are a junior financial analyst. Use the provided context to answer the user prompt.",
    "Senior PM": "You are a senior portfolio manager. Be concise and focus on actionable insights and risk-reward.",
    "Risk Analyst": "You are a risk analyst. Focus on downside scenarios, red flags, and what could go wrong.",
    "Quant": "You are a quantitative analyst. Focus on numerical accuracy, margins, growth rates, and valuation.",
}


def provider_for_model(model: str) -> str:
    """Infer provider from model name."""
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gemini"):
        return "gemini"
    return "openai"


@dataclass
class GenerationConfig:
    label: str
    model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7
    system_prompt: str = "You are a junior financial analyst. Use the provided context to answer the user prompt."
    max_tokens: Optional[int] = None

    @property
    def provider(self) -> str:
        return provider_for_model(self.model)

    def to_dict(self) -> dict:
        d = {
            "label": self.label,
            "model": self.model,
            "provider": self.provider,
            "temperature": self.temperature,
            "system_prompt": self.system_prompt,
        }
        if self.max_tokens is not None:
            d["max_tokens"] = self.max_tokens
        return d


def _labels(n: int) -> List[str]:
    """Generate labels A, B, C, ... for n configs."""
    return [chr(ord("A") + i) for i in range(n)]


def build_temperature_sweep(temps: List[float]) -> List[GenerationConfig]:
    return [
        GenerationConfig(label=lbl, temperature=t)
        for lbl, t in zip(_labels(len(temps)), temps)
    ]


def build_model_comparison(models: List[str]) -> List[GenerationConfig]:
    return [
        GenerationConfig(label=lbl, model=m)
        for lbl, m in zip(_labels(len(models)), models)
    ]


def build_persona_sweep() -> List[GenerationConfig]:
    items = list(DEFAULT_SYSTEM_PROMPTS.items())
    return [
        GenerationConfig(label=lbl, system_prompt=prompt)
        for lbl, (_, prompt) in zip(_labels(len(items)), items)
    ]


def build_cross_provider() -> List[GenerationConfig]:
    """Compare top models across all three providers."""
    return [
        GenerationConfig(label="A", model="claude-sonnet-4-20250514", temperature=0.7),
        GenerationConfig(label="B", model="gemini-2.0-flash", temperature=0.7),
        GenerationConfig(label="C", model="gpt-4o-mini", temperature=0.7),
    ]


# --- Flat preset dict (kept for backward compat / direct lookup) ---
PRESETS = {
    "Temperature Sweep (4 outputs)": build_temperature_sweep([0.2, 0.5, 0.8, 1.0]),
    "Anthropic (3 models)": build_model_comparison(
        ["claude-sonnet-4-20250514", "claude-haiku-4-20250414", "claude-3-5-sonnet-20241022"]
    ),
    "Gemini (4 models)": build_model_comparison(
        ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-pro"]
    ),
    "OpenAI (3 models)": build_model_comparison(
        ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
    ),
    "Cross-Provider (3 outputs)": build_cross_provider(),
    "Persona Sweep (4 outputs)": build_persona_sweep(),
}

# --- Category-based grouping for the UI ---
PRESET_CATEGORIES = {
    "Model Comparison": {
        "Anthropic (3 models)": PRESETS["Anthropic (3 models)"],
        "Gemini (4 models)": PRESETS["Gemini (4 models)"],
        "OpenAI (3 models)": PRESETS["OpenAI (3 models)"],
        "Cross-Provider (3 outputs)": PRESETS["Cross-Provider (3 outputs)"],
    },
    "Temperature Sweep": {
        "Temperature Sweep (4 outputs)": PRESETS["Temperature Sweep (4 outputs)"],
    },
    "Persona Sweep": {
        "Persona Sweep (4 outputs)": PRESETS["Persona Sweep (4 outputs)"],
    },
    "Custom": {},
}
