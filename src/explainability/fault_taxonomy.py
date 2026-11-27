"""
Fault Taxonomy: Maps anomalous signal patterns → ECU subsystem
Aligns with FMEA (Failure Mode & Effects Analysis) — key Bosch process.
"""

FAULT_TAXONOMY = {
    "throttle_position": {
        "stuck_at":       "ETC (Electronic Throttle Control) — motor jam or encoder failure",
        "voltage_spike":  "ETC — power supply transient, check 5V reference",
        "signal_dropout": "ETC — CAN timeout, check wiring harness",
        "noise_burst":    "ETC — EMI interference, check shielding",
    },
    "engine_rpm": {
        "stuck_at":       "Crankshaft Position Sensor — signal lost, engine stall risk",
        "noise_burst":    "Ignition system EMI — check coil suppression",
        "signal_dropout": "CKP sensor open circuit",
    },
    "battery_voltage": {
        "voltage_spike":  "Alternator load dump transient — check TVS diodes",
        "signal_dropout": "BMS CAN message timeout",
        "noise_burst":    "PWM noise from DC/DC converter",
    },
    "coolant_temp": {
        "stuck_at":       "NTC thermistor open circuit — possible engine overtemp risk",
        "voltage_spike":  "Wiring short to 5V reference",
    },
    "lambda_sensor": {
        "noise_burst":    "Heater circuit degradation — response time increasing",
        "stuck_at":       "Sensor poisoning or wiring fault",
        "signal_dropout": "ECU lambda monitoring timeout — failsafe mode engaged",
    },
}

def diagnose(signal_id: str, fault_type: str) -> dict:
    subsystem = FAULT_TAXONOMY.get(signal_id, {}).get(fault_type, "Unknown — manual inspection required")
    return {
        "signal": signal_id,
        "fault_type": fault_type,
        "probable_cause": subsystem,
        "fmea_reference": f"FMEA-{signal_id.upper()[:3]}-{fault_type[:3].upper()}"
    }
