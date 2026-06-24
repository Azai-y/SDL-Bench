from __future__ import annotations

from typing import Any, Dict, List, Optional

from .utils import get_path


class ObservationGenerator:
    def __init__(self, task: Dict[str, Any]):
        self.task = task

    def format_action_space(
        self,
        action_space: List[Dict[str, Any]],
    ) -> str:
        lines: List[str] = []

        for index, item in enumerate(action_space):
            action = item.get("action", "")
            description = item.get("description", "")
            lines.append(
                f"{index + 1}. {action}\n"
                f"   -> {description}"
            )

        return "\n".join(lines)

    def _build_sensor_observation(
        self,
        env_state: Dict[str, Any],
        visible_fields: List[str],
    ) -> Dict[str, Any]:
        sensor: Dict[str, Any] = {}

        for path in visible_fields:
            value = get_path(env_state, path)
            if value is not None:
                self._set_path(sensor, path, value)

        return sensor

    @staticmethod
    def _set_path(
        data: Dict[str, Any],
        path: str,
        value: Any,
    ) -> None:
        keys = path.split(".")
        current = data

        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _filter_current_batch(
        self,
        current_batch: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if current_batch is None:
            return None

        settings = self.task.get("observability_settings", {})
        visible_fields = settings.get(
            "current_batch_visible_fields",
            [
                "batch_id",
                "waste_type",
                "volume_ml",
                "is_final_batch",
            ],
        )

        return {
            key: current_batch.get(key)
            for key in visible_fields
            if key in current_batch
        }

    def make_observation(
        self,
        round_num: int,
        env_state: Dict[str, Any],
        current_batch: Optional[Dict[str, Any]],
        history: List[Dict[str, Any]],
        agent_ledger: Dict[str, Any],
    ) -> Dict[str, Any]:
        settings = self.task.get("observability_settings", {})
        visible_fields = settings.get("visible_fields", [])
        history_window = int(settings.get("history_window", 2))

        sensor_observation = self._build_sensor_observation(
            env_state,
            visible_fields,
        )

        observation = {
            "task": {
                "instruction": self.task.get("instruction", ""),
            },
            "current_batch": self._filter_current_batch(current_batch),
            "sensor_observation": sensor_observation,
            "memory": {
                "agent_ledger": agent_ledger,
                "history": history[-history_window:],
            },
            "action_space": self.format_action_space(
                self.task.get("action_space", [])
            ),
            "world_dynamics": self.task.get("world_dynamics", {}),
            "safety_constraints": self.task.get(
                "safety_constraints",
                {},
            ),
            "termination_conditions": self.task.get(
                "termination_conditions",
                {},
            ),
        }

        return {
            "round": round_num,
            "horizon": self.task.get("horizon", 6),
            "observation": observation,
        }
