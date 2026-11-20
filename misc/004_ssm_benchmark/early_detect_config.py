"""Early-detection benchmark configuration (exprim.anchor based)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import exprim.anchor as anchor
    from exprim.anchor import BaseConfig
except ImportError as exc:
    raise SystemExit(
        "exprim not importable. Install EXPrimitives first:\n"
        "  pip install git+https://github.com/K00LDUD3/EXPrimitives.git"
    ) from exc

import numpy as np

anchor.paths.set_root(Path(__file__).resolve().parent)


@dataclass(frozen=True)
class PlantConfig(BaseConfig):
    name: str = "hopf_normal_form"
    mu: float = 0.5
    omega: float = 1.0
    dt: float = 0.01
    t_end: float = 60.0
    x0: list[float] = field(default_factory=lambda: [0.1, 0.1])

    def validate(self) -> list[str]:
        errors = super().validate()
        if self.dt <= 0:
            errors.append(f"PlantConfig: dt must be positive, got {self.dt}")
        if self.t_end <= 0:
            errors.append(f"PlantConfig: t_end must be positive, got {self.t_end}")
        if len(self.x0) != 2:
            errors.append(f"PlantConfig: x0 must have 2 elements, got {len(self.x0)}")
        return errors


@dataclass(frozen=True)
class ReferenceConfig(BaseConfig):
    freq: float = 1.0
    demand_fraction_of_healthy_ams: float = 0.85
    radius_override: float | None = None

    def validate(self) -> list[str]:
        errors = super().validate()
        if self.freq <= 0:
            errors.append(f"ReferenceConfig: freq must be positive, got {self.freq}")
        if not (0.0 < self.demand_fraction_of_healthy_ams <= 1.0):
            errors.append(
                "ReferenceConfig: demand_fraction_of_healthy_ams must be in (0,1], "
                f"got {self.demand_fraction_of_healthy_ams}"
            )
        return errors


@dataclass(frozen=True)
class ActuatorConfig(BaseConfig):
    n_actuators: int = 3
    angles_deg: list[float] = field(default_factory=lambda: [0.0, 120.0, 240.0])
    u_max: float = 1.0
    degrade_channel: int = 0
    alpha_healthy: float = 1.0
    alpha_sweep: list[float] = field(
        default_factory=lambda: [1.0, 0.85, 0.7, 0.55, 0.4, 0.3, 0.2]
    )

    def validate(self) -> list[str]:
        errors = super().validate()
        if len(self.angles_deg) != self.n_actuators:
            errors.append(
                f"ActuatorConfig: {self.n_actuators} actuators but "
                f"{len(self.angles_deg)} angles"
            )
        if self.u_max <= 0:
            errors.append(f"ActuatorConfig: u_max must be positive, got {self.u_max}")
        if not (0 <= self.degrade_channel < self.n_actuators):
            errors.append(
                f"ActuatorConfig: degrade_channel out of range, got {self.degrade_channel}"
            )
        for a in self.alpha_sweep + [self.alpha_healthy]:
            if not (0.0 < a <= 1.0):
                errors.append(f"ActuatorConfig: alpha must be in (0,1], got {a}")
        return errors


@dataclass(frozen=True)
class ControllerConfig(BaseConfig):
    kp: float = 5.0
    allocation: str = "pinv"
    use_feedforward: bool = True

    def validate(self) -> list[str]:
        errors = super().validate()
        if self.kp <= 0:
            errors.append(f"ControllerConfig: kp must be positive, got {self.kp}")
        if self.allocation not in {"pinp", "pinv"}:
            errors.append(f"ControllerConfig: unknown allocation '{self.allocation}'")
        return errors


@dataclass(frozen=True)
class FaultConfig(BaseConfig):
    t_fault: float = 30.0
    ramp_time: float = 15.0

    def validate(self) -> list[str]:
        errors = super().validate()
        if self.t_fault <= 0:
            errors.append(f"FaultConfig: t_fault must be positive, got {self.t_fault}")
        if self.ramp_time < 0:
            errors.append(f"FaultConfig: ramp_time must be >= 0, got {self.ramp_time}")
        return errors


@dataclass(frozen=True)
class ModelConfig(BaseConfig):
    name: str = "ssm_autoencoder"
    latent_dim: int = 2
    hidden_dim: int = 64
    n_hidden: int = 2
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 60
    device: str = "cuda"

    def validate(self) -> list[str]:
        errors = super().validate()
        if self.latent_dim < 1:
            errors.append(f"ModelConfig: latent_dim must be >= 1, got {self.latent_dim}")
        if self.hidden_dim < 1:
            errors.append(f"ModelConfig: hidden_dim must be >= 1, got {self.hidden_dim}")
        if self.lr <= 0:
            errors.append(f"ModelConfig: lr must be positive, got {self.lr}")
        if self.device not in {"cpu", "cuda", "mps"}:
            errors.append(f"ModelConfig: unknown device '{self.device}'")
        return errors


@dataclass(frozen=True)
class DetectionConfig(BaseConfig):
    settle_time: float = 5.0
    window: int = 50
    threshold_sigma: float = 4.0
    min_persist: int = 3
    cusum_drift_k: float = 1.0
    ewma_lambda: float = 0.2
    ewma_L: float = 4.0
    detectors: list[str] = field(
        default_factory=lambda: ["tracking_error", "model_residual", "cusum", "ewma", "ssm"]
    )

    def validate(self) -> list[str]:
        errors = super().validate()
        if self.settle_time < 0:
            errors.append(f"DetectionConfig: settle_time must be >= 0, got {self.settle_time}")
        if self.window < 1:
            errors.append(f"DetectionConfig: window must be >= 1, got {self.window}")
        if self.threshold_sigma <= 0:
            errors.append(
                f"DetectionConfig: threshold_sigma must be positive, got {self.threshold_sigma}"
            )
        if self.min_persist < 1:
            errors.append(f"DetectionConfig: min_persist must be >= 1, got {self.min_persist}")
        allowed = {"tracking_error", "model_residual", "cusum", "ewma", "ssm"}
        for d in self.detectors:
            if d not in allowed:
                errors.append(f"DetectionConfig: unknown detector '{d}'")
        return errors


@dataclass(frozen=True)
class ExperimentConfig(BaseConfig):
    name: str = "early_detect_ams"
    seed: int = 42
    output_dir: str = "runs"

    plant: PlantConfig = field(default_factory=PlantConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    actuators: ActuatorConfig = field(default_factory=ActuatorConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    fault: FaultConfig = field(default_factory=FaultConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)

    def validate(self) -> list[str]:
        errors = super().validate()
        if 1.0 not in self.actuators.alpha_sweep:
            errors.append("ExperimentConfig: alpha_sweep should include 1.0 (healthy)")
        if self.fault.t_fault >= self.plant.t_end:
            errors.append("ExperimentConfig: t_fault must be < t_end")
        if self.detection.settle_time >= self.fault.t_fault:
            errors.append("ExperimentConfig: settle_time must be < t_fault")
        return errors


def validate_or_raise(cfg: BaseConfig) -> None:
    errors = cfg.validate()
    if errors:
        raise ValueError("Config validation failed:\n" + "\n".join(errors))


def build_effectiveness_matrix(angles_deg: list[float]) -> np.ndarray:
    th = np.deg2rad(np.asarray(angles_deg, dtype=float))
    return np.vstack([np.cos(th), np.sin(th)])


def main() -> None:
    cfg = ExperimentConfig()
    validate_or_raise(cfg)

    cfg_path = anchor.paths.from_root("configs", "early_detect.yaml")
    anchor.paths.ensure_dir(cfg_path.parent)
    anchor.save_yaml(cfg, cfg_path)

    d = anchor.load_yaml(cfg_path)
    reloaded = ExperimentConfig(
        name=d["name"], seed=d["seed"], output_dir=d["output_dir"],
        plant=PlantConfig(**d["plant"]),
        reference=ReferenceConfig(**d["reference"]),
        actuators=ActuatorConfig(**d["actuators"]),
        controller=ControllerConfig(**d["controller"]),
        fault=FaultConfig(**d["fault"]),
        model=ModelConfig(**d["model"]),
        detection=DetectionConfig(**d["detection"]),
    )

    print("config OK")
    print(f"  fingerprint : {cfg.fingerprint()}")
    print(f"  round-trip  : {'equal' if reloaded == cfg else 'MISMATCH'}")
    print(f"  settle_time : {cfg.detection.settle_time}")
    print(f"  ramp_time   : {cfg.fault.ramp_time}")
    print(f"  demand_frac : {cfg.reference.demand_fraction_of_healthy_ams}")


if __name__ == "__main__":
    main()
