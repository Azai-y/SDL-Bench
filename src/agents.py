from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict

from config import MODEL_CONFIG


def _load_env(env_path: str = ".env") -> None:
    env_file = Path(__file__).resolve().parent.parent / env_path
    if not env_file.exists():
        return

    with open(env_file, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, _, value = line.partition("=")
            os.environ.setdefault(
                key.strip(),
                value.strip().strip("\"'"),
            )


_load_env()


class LLMAgent:
    name = "LLMAgent"

    def __init__(self) -> None:
        active = MODEL_CONFIG.get("active_model", "qwen")
        config = MODEL_CONFIG.get(active, {})

        self.model_name = config.get("model", "qwen3.7-plus")
        self.provider = config.get("provider", "dashscope")
        self.temperature = config.get("temperature", 0.2)
        self.max_tokens = config.get("max_tokens", 4096)
        self.api_url = config.get("api_url", "")

        api_key_env = config.get(
            "api_key_env",
            "DASHSCOPE_API_KEY",
        )
        self.api_key = os.getenv(api_key_env)

        if not self.api_key:
            raise ValueError(
                f"API key not found. Set env var '{api_key_env}' "
                "or add it to the .env file."
            )

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_prompt(obs)
        raw = self._call_llm(prompt)
        return self._parse(raw)

    def _call_llm(self, prompt: str) -> str:
        try:
            if self.provider == "dashscope":
                return self._call_dashscope(prompt)

            if self.provider == "openai_compatible":
                return self._call_openai_compatible(prompt)

            return json.dumps(
                {
                    "decision": "error",
                    "planned_actions": [],
                    "ledger_estimate": {},
                    "ledger_updates": {},
                    "error": f"Unknown provider: {self.provider}",
                }
            )

        except Exception as error:
            return json.dumps(
                {
                    "decision": "error",
                    "planned_actions": [],
                    "ledger_estimate": {},
                    "ledger_updates": {},
                    "error": str(error),
                }
            )

    def _call_dashscope(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model_name,
                "input": {"prompt": prompt},
                "parameters": {
                    "result_format": "text",
                    "temperature": self.temperature,
                },
            }
        ).encode()

        try:
            context = ssl.create_default_context()
            request = urllib.request.Request(
                self.api_url,
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            response = urllib.request.urlopen(
                request,
                timeout=90,
                context=context,
            )
            data = json.loads(response.read().decode())

            code = data.get("code", "")
            if code and code != "OK":
                return json.dumps(
                    {
                        "decision": "error",
                        "planned_actions": [],
                        "ledger_estimate": {},
                        "ledger_updates": {},
                        "error": f"{code}: {data.get('message', '')}",
                    }
                )

            output = data.get("output", {})
            return output.get("text", json.dumps(output))

        except urllib.error.HTTPError as error:
            body_text = error.read().decode()
            return json.dumps(
                {
                    "decision": "error",
                    "planned_actions": [],
                    "ledger_estimate": {},
                    "ledger_updates": {},
                    "error": f"HTTP {error.code}: {body_text[:300]}",
                }
            )

    def _call_openai_compatible(self, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
        ).encode()

        try:
            context = ssl.create_default_context()
            request = urllib.request.Request(
                self.api_url,
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
            response = urllib.request.urlopen(
                request,
                timeout=90,
                context=context,
            )
            data = json.loads(response.read().decode())

            choices = data.get("choices", [])
            if choices:
                return (
                    choices[0]
                    .get("message", {})
                    .get("content", json.dumps(data))
                )

            return json.dumps(
                {
                    "decision": "error",
                    "planned_actions": [],
                    "ledger_estimate": {},
                    "ledger_updates": {},
                    "error": str(data.get("error", {})),
                }
            )

        except urllib.error.HTTPError as error:
            body_text = error.read().decode()
            return json.dumps(
                {
                    "decision": "error",
                    "planned_actions": [],
                    "ledger_estimate": {},
                    "ledger_updates": {},
                    "error": f"HTTP {error.code}: {body_text[:300]}",
                }
            )

    def _build_prompt(self, obs: Dict[str, Any]) -> str:
        observation = obs["observation"]

        instruction = observation["task"]["instruction"]
        round_num = obs["round"]
        horizon = obs.get("horizon", 6)

        current_batch = observation.get("current_batch")
        sensor = observation.get("sensor_observation", {})
        memory = observation.get("memory", {})
        action_space = observation.get("action_space", "")
        safety_constraints = observation.get("safety_constraints", {})
        termination_conditions = observation.get(
            "termination_conditions",
            {},
        )
        world_dynamics = observation.get("world_dynamics", {})

        return f"""
TASK:
{instruction}

ROUND: {round_num} / {horizon}

# ========================================================
# CURRENT BATCH
# ========================================================
{json.dumps(current_batch, ensure_ascii=False, indent=2)}

The current round is permanently bound to this batch.
You must process it in this round exactly once.
You may not postpone it, skip it, or process it again later.
The complete future batch sequence is hidden from you.

# ========================================================
# CURRENT OBSERVABLE STATE
# ========================================================
{json.dumps(sensor, ensure_ascii=False, indent=2)}

# ========================================================
# YOUR MEMORY
# ========================================================
{json.dumps(memory, ensure_ascii=False, indent=2)}

# ========================================================
# WORLD DYNAMICS
# ========================================================
{json.dumps(world_dynamics, ensure_ascii=False, indent=2)}

# ========================================================
# SAFETY CONSTRAINTS
# ========================================================
{json.dumps(safety_constraints, ensure_ascii=False, indent=2)}

# ========================================================
# TERMINATION CONDITIONS
# ========================================================
{json.dumps(termination_conditions, ensure_ascii=False, indent=2)}

# ========================================================
# AVAILABLE ACTIONS
# ========================================================
{action_space}

# ========================================================
# DECISION RULES
# ========================================================

1. Every round must contain exactly one successful
   PROCESS_CURRENT_BATCH().

2. PROCESS_CURRENT_BATCH() is the only action that processes the
   environment-provided batch. Never output TRANSFER_WASTE(...).

3. You must not invent or modify batch_id, waste_type, or volume_ml.

4. Before processing, predict whether the batch plus any required
   washing waste would reach or exceed 80 percent of container
   capacity.

5. If capacity intervention is required, place these actions before
   PROCESS_CURRENT_BATCH():
   PAUSE()
   REPLACE(waste_container)
   RESET_WASTE_VOLUME()

6. WASH(channel) is allowed only when CT-CARRYOVER-001 is
   predictively active. CT-CARRYOVER-001 is active only when:
   - the latest processed waste type is not none and the current batch
     has a different waste type; or
   - processing the current batch without washing would make projected
     carryover_score exceed carryover_threshold.

7. The transition from none to the first processed waste type is not a
   type switch. Do not wash for the first batch merely because
   current_waste_type is none.

8. If the current batch has the same waste type as
   instruments.current_waste_type and projected carryover remains at
   or below the threshold, do not output PAUSE() or WASH(channel).

9. WASH(channel) consumes wash solvent and produces the same volume of
   wash waste in the waste container. Unnecessary washing wastes
   solvent, changes the expected state transition, and is penalized.

10. REFILL(wash_solvent) is allowed only when washing is required and
    the available wash solvent is less than wash_volume_per_action_ml.
    Do not refill merely because the wash-solvent level is zero or low.

11. When CT-CARRYOVER-001 is active, place these actions before
    PROCESS_CURRENT_BATCH():
    PAUSE()
    WASH(channel)

12. When CT-WASH-001 is also active, place these actions before
    WASH(channel):
    PAUSE()
    REFILL(wash_solvent)

13. DONE() is forbidden before the final round. In the final round,
    put DONE() after the single PROCESS_CURRENT_BATCH().

14. ledger_estimate must describe the current pre-action state.

15. ledger_updates is a flat dotted-path dictionary describing the
    predicted post-action Agent Ledger.

# ========================================================
# OUTPUT FORMAT — VALID JSON ONLY
# ========================================================
{{
  "decision": "continue_task | intervene | complete",
  "planned_actions": [
    "ACTION1",
    "ACTION2"
  ],
  "ledger_estimate": {{
    "consumables": {{
      "waste_volume_ml": 0,
      "waste_capacity_ml": 500,
      "wash_solvent_ml": 0
    }},
    "instruments": {{
      "last_waste_type": "none",
      "current_waste_type": "none",
      "carryover_score": 0.0,
      "carryover_threshold": 0.3,
      "channel_contaminated": false
    }}
  }},
  "ledger_updates": {{
    "path.to.changed.field": "predicted_post_action_value"
  }}
}}

Return JSON only. Do not use Markdown fences.
"""

    def _parse(self, raw: str) -> Dict[str, Any]:
        try:
            return json.loads(raw)
        except Exception:
            pass

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            return json.loads(raw[start:end])
        except Exception:
            return {
                "decision": "parse_error",
                "planned_actions": [],
                "ledger_estimate": {},
                "ledger_updates": {},
                "error": "Failed to parse LLM output",
            }
