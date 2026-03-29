from pathlib import Path
import subprocess


LOG_PATH = Path("/home/pi/touch_3mm_serial.log")
CMD = [
    "python3",
    "-u",
    "/home/pi/touch_board_target.py",
    "--handeye",
    "/home/pi/handeye_dataset/session1/handeye_result.json",
    "--tcp",
    "/home/pi/handeye_dataset/session1/gripper_tcp_left_front_tip_samples_004_006.json",
    "--locked-rpy-json",
    "/home/pi/safe_reverse_5mm_pose.json",
    "--camera-device",
    "auto",
    "--target",
    "board_xy",
    "--board-xy",
    "0.075",
    "0.06",
    "--lift-first-mm",
    "100",
    "--approach-mm",
    "3",
    "--pretouch-mm",
    "2",
    "--approach-only",
    "--execute",
    "--speed-percent",
    "5",
    "--approach-sign",
    "negative",
]


def main() -> None:
    LOG_PATH.write_text("", encoding="utf-8")
    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write("[runner] starting 3mm serial touch flow\n")
        log_file.flush()
        proc = subprocess.run(CMD, stdout=log_file, stderr=log_file, text=True)
        log_file.write(f"[runner] return_code={proc.returncode}\n")
        log_file.flush()


if __name__ == "__main__":
    main()
