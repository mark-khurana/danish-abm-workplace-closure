"""Intervention substep — applies trigger-based reactive policies at each timestep."""

import torch

from agent_torch.core.substep import SubstepTransition
from agent_torch.core.helpers import get_by_path


class ApplyInterventions(SubstepTransition):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = torch.device(self.config["simulation_metadata"]["device"])

        self._trigger_engine = None
        self._trigger_initialized = False
        self._scale_factor = self.config["simulation_metadata"].get("scale_factor", 1.0)
        self._num_agents = self.config["simulation_metadata"]["num_agents"]
        self._healthcare_capacity = self.config["simulation_metadata"].get(
            "healthcare_capacity", {"hospital_beds": 15000, "icu_beds": 400}
        )

    def _maybe_init_triggers(self):
        """Lazily initialize the trigger engine from the runtime registry."""
        if self._trigger_initialized:
            return
        self._trigger_initialized = True

        from danish_abm import runtime_get
        engine = runtime_get(self.config, "trigger_engine")
        if engine is not None:
            self._trigger_engine = engine

    def forward(self, state, action):
        """Evaluate trigger policies against current simulation metrics. Stores dynamic
        multipliers in the runtime registry for NewTransmission to pick up.
        """
        t = state["current_step"]

        self._maybe_init_triggers()

        if self._trigger_engine is None:
            return {}

        from danish_abm import runtime_get, runtime_set
        transmission_tracking = runtime_get(self.config, "transmission_tracking")
        if transmission_tracking is None:
            return {}

        from danish_abm.triggers import compute_trigger_metrics
        try:
            actual_n = state["agents"]["citizens"]["disease_stage"].shape[0]
        except (KeyError, AttributeError):
            actual_n = self._num_agents
        metrics = compute_trigger_metrics(
            state=state,
            tracking=transmission_tracking,
            t=t,
            num_agents=actual_n,
            healthcare_capacity=self._healthcare_capacity,
            scale_factor=self._scale_factor,
        )

        test_tracking = runtime_get(self.config, "testing_tracking")
        if test_tracking:
            daily_tests = test_tracking.get("daily_tests", [])
            daily_pos = test_tracking.get("daily_positives", [])
            if daily_tests and daily_pos and len(daily_tests) > 0:
                recent_tests = sum(daily_tests[-7:])
                recent_pos = sum(daily_pos[-7:])
                if recent_tests > 0:
                    metrics["positivity_rate"] = recent_pos / recent_tests

        dynamic_multiplier = self._trigger_engine.step(t, metrics)

        runtime_set(self.config, "trigger_dynamic_multiplier", dynamic_multiplier.to(self.device))

        return {}
