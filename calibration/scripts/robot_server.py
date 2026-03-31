#!/usr/bin/env python3
"""Formal Raspberry Pi robot server entrypoint.

This wraps the existing pick-place bridge with production-style defaults:
- service name `robot_server`
- eager hardware warm-up on process start
- keep robot/camera alive across client session open/close calls
- optional CAN activation using the local pyAgxArm helper
"""

from pi_pick_place_bridge import main


if __name__ == "__main__":
    main(
        description="Persistent Pi-side robot server",
        service_name="robot_server",
        default_keep_hardware_alive=True,
        default_eager_open=True,
        default_can_activate_script="/home/pi/pyAgxArm/pyAgxArm/scripts/linux/can_activate.sh",
        default_can_channel="can0",
        default_can_bitrate=1000000,
        default_can_usb_address="2-1:1.0",
    )
