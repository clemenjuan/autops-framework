"""Pure inter-satellite-link budget helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_BOLTZMANN_J_PER_K = 1.38e-23
_LIGHT_SPEED_M_S = 3.0e8


@dataclass(frozen=True, slots=True)
class ISLConfig:
    """Configured radio parameters without mission-specific defaults."""

    tx_power_w: float
    rx_gain_db: float
    rx_loss_db: float
    tx_gain_db: float
    tx_loss_db: float
    frequency_hz: float
    bandwidth_hz: float
    symbol_rate_hz: float
    modulation_order: int
    sensitivity_dbw: float
    noise_temperature_k: float

    def __post_init__(self) -> None:
        positive = (
            self.tx_power_w,
            self.frequency_hz,
            self.bandwidth_hz,
            self.symbol_rate_hz,
            self.noise_temperature_k,
        )
        if any(value <= 0.0 for value in positive) or self.modulation_order < 2:
            raise ValueError("ISL power, rates, temperature, and modulation must be positive")

    @classmethod
    def from_mapping(cls, config: Mapping[str, object]) -> ISLConfig:
        """Read RF constants from the mission's ``isl`` mapping."""

        return cls(
            tx_power_w=float(config["tx_power_w"]),
            rx_gain_db=float(config["rx_gain_db"]),
            rx_loss_db=float(config["rx_loss_db"]),
            tx_gain_db=float(config["tx_gain_db"]),
            tx_loss_db=float(config["tx_loss_db"]),
            frequency_hz=float(config["frequency_hz"]),
            bandwidth_hz=float(config["bandwidth_hz"]),
            symbol_rate_hz=float(config["symbol_rate_hz"]),
            modulation_order=int(config["modulation_order"]),
            sensitivity_dbw=float(config["sensitivity_dbw"]),
            noise_temperature_k=float(config["noise_temperature_k"]),
        )


def vector_range_km(endpoint_a_km: Sequence[float], endpoint_b_km: Sequence[float]) -> float:
    if len(endpoint_a_km) != 3 or len(endpoint_b_km) != 3:
        raise ValueError("vector_range_km expects two 3D vectors")
    return math.sqrt(
        sum(
            (float(right) - float(left)) ** 2
            for left, right in zip(endpoint_a_km, endpoint_b_km, strict=True)
        )
    )


def free_space_loss_db(distance_m: float, frequency_hz: float) -> float:
    if distance_m <= 0.0 or frequency_hz <= 0.0:
        raise ValueError("distance_m and frequency_hz must be positive")
    return 20.0 * math.log10(4.0 * math.pi * distance_m * frequency_hz / _LIGHT_SPEED_M_S)


def noise_power_dbw(config: ISLConfig) -> float:
    return 10.0 * math.log10(_BOLTZMANN_J_PER_K * config.noise_temperature_k * config.bandwidth_hz)


def received_power_dbw(distance_m: float, config: ISLConfig) -> float:
    return (
        10.0 * math.log10(config.tx_power_w)
        + config.rx_gain_db
        + config.tx_gain_db
        - config.rx_loss_db
        - config.tx_loss_db
        - free_space_loss_db(distance_m, config.frequency_hz)
    )


def snr_db(distance_m: float, config: ISLConfig) -> float:
    return received_power_dbw(distance_m, config) - noise_power_dbw(config)


def ideal_data_rate_bps(distance_m: float, config: ISLConfig) -> float:
    snr_linear = 10.0 ** (snr_db(distance_m, config) / 10.0)
    shannon_rate = config.bandwidth_hz * math.log2(1.0 + snr_linear)
    modulation_cap = config.symbol_rate_hz * math.log2(config.modulation_order)
    return min(shannon_rate, modulation_cap)


def bit_error_rate(distance_m: float, config: ISLConfig) -> float:
    bits_per_symbol = math.log2(config.modulation_order)
    spectral_efficiency = config.symbol_rate_hz * bits_per_symbol / config.bandwidth_hz
    ebn0 = 10.0 ** (snr_db(distance_m, config) / 10.0) / spectral_efficiency
    return math.erfc(math.sqrt(2.0 * ebn0)) / bits_per_symbol


def effective_data_rate_bps(distance_m: float, config: ISLConfig) -> float:
    if received_power_dbw(distance_m, config) < config.sensitivity_dbw:
        return 0.0
    return ideal_data_rate_bps(distance_m, config) * (1.0 - bit_error_rate(distance_m, config))


def isl_link_budget(distance_m: float, config: ISLConfig) -> dict[str, float]:
    received_dbw = received_power_dbw(distance_m, config)
    return {
        "distance_m": float(distance_m),
        "free_space_loss_db": free_space_loss_db(distance_m, config.frequency_hz),
        "received_power_dbw": received_dbw,
        "sensitivity_dbw": config.sensitivity_dbw,
        "margin_db": received_dbw - config.sensitivity_dbw,
        "noise_power_dbw": noise_power_dbw(config),
        "snr_db": snr_db(distance_m, config),
        "ideal_data_rate_bps": ideal_data_rate_bps(distance_m, config),
        "bit_error_rate": bit_error_rate(distance_m, config),
        "effective_data_rate_bps": effective_data_rate_bps(distance_m, config),
    }


def is_isl_feasible(
    endpoint_a_km: Sequence[float],
    endpoint_b_km: Sequence[float],
    *,
    endpoint_a_idle: bool,
    endpoint_b_idle: bool,
    config: ISLConfig,
) -> bool:
    if not endpoint_a_idle or not endpoint_b_idle:
        return False
    distance_m = vector_range_km(endpoint_a_km, endpoint_b_km) * 1000.0
    return received_power_dbw(distance_m, config) >= config.sensitivity_dbw
