# Auto-generated snapshot by anchor.serialization.save_python()
# Do not edit — for inspection only.
# For programmatic reconstruction use config.yaml.

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:

    @dataclass(frozen=True)
    class PlantConfig:

        name: str = 'hopf_normal_form'
        mu: float = 0.5
        omega: float = 1.0
        dt: float = 0.01
        t_end: float = 60.0
        x0: list = [0.1, 0.1]

    @dataclass(frozen=True)
    class ReferenceConfig:

        freq: float = 1.0
        demand_fraction_of_healthy_ams: float = 0.55
        radius_override: NoneType = None

    @dataclass(frozen=True)
    class ActuatorConfig:

        n_actuators: int = 3
        angles_deg: list = [0.0, 120.0, 240.0]
        u_max: float = 1.0
        degrade_channel: int = 0
        alpha_healthy: float = 1.0
        alpha_sweep: list = [1.0, 0.85, 0.7, 0.55, 0.4, 0.3, 0.2]

    @dataclass(frozen=True)
    class ControllerConfig:

        kp: float = 5.0
        allocation: str = 'pinv'
        use_feedforward: bool = True

    @dataclass(frozen=True)
    class FaultConfig:

        t_fault: float = 30.0

    @dataclass(frozen=True)
    class ModelConfig:

        name: str = 'ssm_autoencoder'
        latent_dim: int = 2
        hidden_dim: int = 64
        n_hidden: int = 2
        batch_size: int = 256
        lr: float = 0.001
        epochs: int = 60
        device: str = 'cuda'

    @dataclass(frozen=True)
    class DetectionConfig:

        window: int = 50
        threshold_sigma: float = 4.0
        cusum_drift_k: float = 0.5
        ewma_lambda: float = 0.2
        ewma_L: float = 3.0
        detectors: list = ['tracking_error', 'model_residual', 'cusum', 'ewma', 'ssm']

    name: str = 'early_detect_ams'
    seed: int = 43
    output_dir: str = 'runs'
    plant: PlantConfig = PlantConfig()
    reference: ReferenceConfig = ReferenceConfig()
    actuators: ActuatorConfig = ActuatorConfig()
    controller: ControllerConfig = ControllerConfig()
    fault: FaultConfig = FaultConfig()
    model: ModelConfig = ModelConfig()
    detection: DetectionConfig = DetectionConfig()