"""Danish population-scale agent-based model for epidemic response. Built on the
AgentTorch framework.
"""

from agent_torch.core import Runner, Registry

from .substeps.transmission.transition import NewTransmission
from .substeps.transmission.action import MakeIsolationDecision
from .substeps.disease_progression.transition import SEIRMProgression
from .substeps.interventions.transition import ApplyInterventions
from .substeps.vaccination.transition import VaccinationRollout
from .substeps.testing.transition import TestingAndIsolation
from .compartments import CompartmentModel, Transition
from .pathogen_presets import PATHOGEN_PRESETS as COMPARTMENT_PRESETS
from .pathogen_presets import get_preset_names as get_compartment_preset_names
from .disease_params import DiseaseParams, ParamSpec, PATHOGEN_PRESETS, get_pathogen, register_pathogen
from .disease_params import disease_params_from_compartment_model
from .strains import StrainDef, StrainConfig, StrainManager
from .immunity import ImmunityTracker, WaningFunction, VaccineProfile, VACCINE_PROFILES
from .npi import NPISchedule, NPIDefinition, BUILTIN_NPIS, LearnableNPISchedule
from .seasonal import SeasonalForcing, get_seasonal, SEASONAL_PRESETS
from .data_config import DataConfig
from .scenario import ScenarioSpec, ScenarioResult, ScenarioRunner
from .triggers import (
    TriggerMetric, TriggerOperator, TriggerCondition, TriggerPolicy,
    TriggerEngine, compute_trigger_metrics,
    get_trigger_preset, TRIGGER_PRESETS,
    denmark_tiered_response, icu_capacity_response, rt_based_response,
)
from .uncertainty import (
    UncertaintyConfig, UncertaintyMode, ParameterPrior, ParameterSample,
    StructuralAssumption, decompose_variance, parameter_sensitivity,
)
from .policy_outcomes import (
    PolicyMetrics, HealthcareCapacity, EnsemblePolicyMetrics,
    compute_hospital_occupancy, compute_icu_occupancy,
)
from .registry_data import (
    vaccination_csv_to_schedule,
    save_vaccination_schedule,
    build_vaccination_schedule_from_csv,
    load_individual_test_data,
    load_municipality_ground_truth,
)
from .surveillance import (
    MunicipalityGraph,
    SurveillanceState,
    SurrogateModel,
    AllocationPolicy,
    RandomPolicy,
    NodeEntropyPolicy,
    BALDPolicy,
    LocalEntropyPolicy,
    DegreeCentralityPolicy,
    PageRankPolicy,
    ReactiveInfectedPolicy,
    ProportionalToIncidencePolicy,
    SURVEILLANCE_POLICIES,
    get_surveillance_policy,
)
from .agent_properties import (
    AgentPropertyRegistry,
    AgentEventLog,
    OutcomeAnalysis,
    prepare_population_dir,
    read_tabular,
    find_file,
    find_tabular_files,
)
from .substeps.utils import (
    read_from_file,
    load_population_attribute,
    network_from_file,
    network_from_layered_files,
    load_network_for_year,
    get_lam_gamma_integrals,
    get_mean_agent_interactions,
    get_infected_time,
    get_next_stage_time,
    initialize_id,
    configure_layers,
    get_layer_config,
    LAYER_NAMES,
    LAYER_NAME_TO_IDX,
    NUM_STRUCTURED_LAYERS,
)


_RUNTIME = {}


def runtime_set(config, key, value):
    """Store a runtime object so other substeps sharing the same config can find it."""
    _RUNTIME[(id(config), key)] = value


def runtime_get(config, key, default=None):
    """Retrieve a runtime object stored by another substep."""
    return _RUNTIME.get((id(config), key), default)


def runtime_clear(config):
    """Evict all runtime entries registered under this config."""
    cid = id(config)
    for k in [k for k in _RUNTIME if k[0] == cid]:
        del _RUNTIME[k]


def get_registry():
    """Create and populate a registry with all model components."""
    reg = Registry()

    reg.register(MakeIsolationDecision, "make_isolation_decision", key="policy")
    reg.register(NewTransmission, "new_transmission", key="transition")
    reg.register(SEIRMProgression, "seirm_progression", key="transition")
    reg.register(ApplyInterventions, "apply_interventions", key="transition")
    reg.register(VaccinationRollout, "vaccination_rollout", key="transition")
    reg.register(TestingAndIsolation, "testing_and_isolation", key="transition")

    reg.register(load_population_attribute, "load_population_attribute", key="initialization")
    reg.register(read_from_file, "read_from_file", key="initialization")
    reg.register(get_lam_gamma_integrals, "get_lam_gamma_integrals", key="initialization")
    reg.register(get_mean_agent_interactions, "get_mean_agent_interactions", key="initialization")
    reg.register(get_infected_time, "get_infected_time", key="initialization")
    reg.register(get_next_stage_time, "get_next_stage_time", key="initialization")
    reg.register(initialize_id, "initialize_id", key="initialization")

    reg.register(network_from_file, "network_from_file", key="network")
    reg.register(network_from_layered_files, "network_from_layered_files", key="network")

    return reg


def get_runner(config, registry=None):
    """Create a Runner instance for this model."""
    if registry is None:
        registry = get_registry()

    import os, numpy as np
    meta = config.get("simulation_metadata", {})
    ds_file = meta.get("disease_stage_file")
    n_agents = config.get("state", {}).get("agents", {}).get("citizens", {}).get("number", 0)
    if ds_file and n_agents > 0:
        ds_path = ds_file if os.path.isabs(ds_file) else os.path.join(os.getcwd(), ds_file)
        need_write = True
        if os.path.exists(ds_path):
            n_lines = sum(1 for _ in open(ds_path))
            if n_lines == n_agents:
                need_write = False
            else:
                print(f"  [get_runner] disease_stages.csv has {n_lines:,} rows, "
                      f"need {n_agents:,} — regenerating")
        if need_write:
            os.makedirs(os.path.dirname(ds_path), exist_ok=True)
            np.savetxt(ds_path, np.zeros((n_agents, 1), dtype=int), fmt="%d")
            print(f"  [get_runner] Generated {ds_path}: {n_agents:,} rows")

    return Runner(config, registry)
