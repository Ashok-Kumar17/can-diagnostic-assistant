"""
generate_demo_log.py

Generates a synthetic CAN log (candump-style .log format) simulating ~60 seconds
of EV driving where sustained high torque demand causes the motor controller to
overheat, eventually tripping a DTC (P0C31_CONTROLLER_OVERTEMP) and forcing the
drive into a derate state.

This is synthetic/demo data only -- not real vehicle data. Swap this script's
output for your own decoded rover/vehicle CAN logs when doing a real demo.
"""
import cantools
import random
import math

DBC_PATH = "dbc/vehicle_demo.dbc"
OUT_PATH = "data/demo_drive.log"

db = cantools.database.load_file(DBC_PATH)

msg_motor_status = db.get_message_by_name("MOTOR_STATUS")
msg_motor_torque = db.get_message_by_name("MOTOR_TORQUE")
msg_bms_pack = db.get_message_by_name("BMS_PACK_STATUS")
msg_bms_cell = db.get_message_by_name("BMS_CELL_STATUS")
msg_dtc = db.get_message_by_name("DTC_REPORT")

random.seed(42)

lines = []
t = 0.0
dt = 0.05  # 20 Hz bus traffic
duration = 60.0

dtc_fired = False
dtc_cleared_state = False
ctrl_temp = 45.0
fault_time = None

while t < duration:
    # Driving profile: throttle ramps up around t=15s and stays high (sustained climb/load),
    # then backs off for 5s once the controller is in derate (driver reacts to power loss).
    if dtc_fired and (t - fault_time) < 5:
        throttle = 78  # driver still pushing for a moment before noticing derate
    elif dtc_fired:
        throttle = 15  # backed off after derate is evident
    elif t < 15:
        throttle = 20 + 5 * math.sin(t)
    else:
        throttle = 78 + 4 * math.sin(t * 0.5)

    throttle = max(0, min(100, throttle))
    motor_rpm = int(2000 + throttle * 55 + random.uniform(-30, 30))
    motor_current = throttle * 4.2 + random.uniform(-2, 2)
    torque_cmd = throttle * 8.5
    torque_actual = torque_cmd * (0.35 if dtc_fired else 0.97)  # derate cuts torque after fault

    # Controller temp: climbs while throttle has been sustained high, cools once derated.
    # State-based (depends on accumulated heating/cooling), not a hardcoded time window,
    # so the fault is a genuine consequence of the heating trend, not a scripted jump.
    if not dtc_fired:
        heating_rate = 0.1 + (throttle / 100.0) * 4.0  # degC/sec, scales with sustained throttle
        ctrl_temp += heating_rate * dt
    else:
        ctrl_temp -= 2.2 * dt  # degC/sec active cooling once derated

    ctrl_temp = min(214, max(40, ctrl_temp))
    motor_temp = min(214, max(-39, ctrl_temp - random.uniform(5, 10)))

    # Fault fires once controller temp crosses the thermal limit
    motor_state = 1  # Running
    if ctrl_temp >= 145 and not dtc_fired:
        dtc_fired = True
        fault_time = t
    if dtc_fired:
        motor_state = 4  # Derate

    data_motor = msg_motor_status.encode({
        "MotorRPM": min(8000, max(0, motor_rpm)),
        "MotorCurrent": min(499, max(-499, motor_current)),
        "MotorTemp": motor_temp,
        "ControllerTemp": ctrl_temp,
        "MotorState": motor_state,
    })
    data_torque = msg_motor_torque.encode({
        "TorqueCommand": min(999, max(-999, torque_cmd)),
        "TorqueActual": min(999, max(-999, torque_actual)),
        "ThrottlePosition": throttle,
    })
    pack_soc = max(20, 95 - t * 0.3)
    data_bms = msg_bms_pack.encode({
        "PackVoltage": min(500, max(0, 360 - (100 - pack_soc) * 0.4)),
        "PackCurrent": min(295, max(-295, motor_current * 0.85)),
        "PackSOC": pack_soc,
        "PackTemp": min(214, 28 + t * 0.05),
    })
    data_cell = msg_bms_cell.encode({
        "MaxCellVoltage": min(4.99, max(0, 4.05 - (100 - pack_soc) * 0.002)),
        "MinCellVoltage": min(4.98, max(0, 3.95 - (100 - pack_soc) * 0.002)),
        "CellTempMax": min(214, 30 + t * 0.05),
    })

    def fmt(can_id, data):
        hexstr = "".join(f"{b:02X}" for b in data)
        return f"({t:.6f}) can0 {can_id:03X}#{hexstr}"

    lines.append(fmt(msg_motor_status.frame_id, data_motor))
    lines.append(fmt(msg_motor_torque.frame_id, data_torque))
    lines.append(fmt(msg_bms_pack.frame_id, data_bms))
    lines.append(fmt(msg_bms_cell.frame_id, data_cell))

    # Fire the DTC frame exactly once, right when the fault trips
    if dtc_fired and not dtc_cleared_state:
        data_dtc = msg_dtc.encode({
            "DTC_Code": 4365,  # P0C31_CONTROLLER_OVERTEMP
            "DTC_Status": 1,   # SET
            "DTC_Source": 1,   # MotorCtrl
        })
        lines.append(fmt(msg_dtc.frame_id, data_dtc))
        dtc_cleared_state = True

    t += dt

with open(OUT_PATH, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"Wrote {len(lines)} frames to {OUT_PATH}")
if fault_time is not None:
    print(f"Fault tripped at t={fault_time:.2f}s (controller temp crossed 145degC)")
else:
    print("No fault tripped in this run -- adjust throttle/heating profile if you wanted one.")
