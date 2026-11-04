"""Environment Pre-Flight Verification Script for FastAgent.

Checks:
  (a) ADB connectivity & RTT jitter to the target device.
  (b) HyperOS security setting (persist.security.adbinput == 1).
  (c) Scrcpy video/control socket availability (port 8010).
  (d) vLLM OpenAI-compatible endpoint health and loaded model verification.
"""

from __future__ import annotations

import json
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from fast_agent.config import AgentConfig


@dataclass
class PreFlightResult:
    adb_connected: bool
    adb_rtt_min_ms: float
    adb_rtt_avg_ms: float
    adb_rtt_max_ms: float
    adb_rtt_mdev_ms: float
    hyperos_security_ok: bool
    scrcpy_mirror_ok: bool
    vllm_healthy: bool
    vllm_model_id: Optional[str]
    all_passed: bool
    details: Dict[str, str]


def measure_ping_rtt(ip: str, count: int = 5) -> Tuple[float, float, float, float]:
    """Measure ping RTT statistics (min, avg, max, mdev in ms)."""
    try:
        res = subprocess.run(
            ["ping", "-c", str(count), "-W", "2", ip],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if res.returncode == 0:
            match = re.search(r"rtt min/avg/max/mdev = ([0-9.]+)/([0-9.]+)/([0-9.]+)/([0-9.]+)", res.stdout)
            if match:
                return (
                    float(match.group(1)),
                    float(match.group(2)),
                    float(match.group(3)),
                    float(match.group(4)),
                )
    except Exception:
        pass
    return (0.0, 0.0, 0.0, 0.0)


def check_adb_connectivity(adb_path: str, serial: str) -> bool:
    """Verify ADB connection to the specified serial."""
    try:
        # Attempt connection first if it's an IP:port
        if ":" in serial:
            subprocess.run([adb_path, "connect", serial], capture_output=True, timeout=5)
        res = subprocess.run([adb_path, "devices"], capture_output=True, text=True, timeout=5)
        for line in res.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == serial and parts[1] == "device":
                return True
    except Exception:
        pass
    return False


def check_hyperos_security(adb_path: str, serial: str) -> bool:
    """Verify persist.security.adbinput == 1 on Xiaomi / HyperOS devices."""
    try:
        res = subprocess.run(
            [adb_path, "-s", serial, "shell", "getprop", "persist.security.adbinput"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        val = res.stdout.strip()
        return val == "1"
    except Exception:
        return False


def check_socket_port(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check TCP connectivity to host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def check_vllm_health(base_url: str, api_key: str, timeout: float = 4.0) -> Tuple[bool, Optional[str]]:
    """Check vLLM / OpenAI-compatible endpoint health and discover active model name."""
    url = f"{base_url.rstrip('/')}/models"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                payload = json.loads(response.read().decode("utf-8"))
                models = payload.get("data", [])
                if models and isinstance(models, list):
                    model_id = str(models[0].get("id"))
                    return True, model_id
                return True, None
    except Exception:
        pass
    return False, None


def run_preflight(config: Optional[AgentConfig] = None) -> PreFlightResult:
    """Run comprehensive environment pre-flight verification."""
    cfg = config or AgentConfig()
    details: Dict[str, str] = {}

    # Extract target host from serial (if IP:PORT)
    ip_match = re.match(r"^([0-9.]+):[0-9]+$", cfg.device_serial)
    target_ip = ip_match.group(1) if ip_match else "127.0.0.1"

    # (a) ADB & RTT
    rtt_min, rtt_avg, rtt_max, rtt_mdev = measure_ping_rtt(target_ip, count=5)
    adb_ok = check_adb_connectivity(cfg.adb_path, cfg.device_serial)
    details["rtt"] = f"min={rtt_min:.1f}ms, avg={rtt_avg:.1f}ms, max={rtt_max:.1f}ms, jitter={rtt_mdev:.1f}ms"
    details["adb"] = "CONNECTED" if adb_ok else "DISCONNECTED"

    # (b) HyperOS Security
    hyperos_ok = check_hyperos_security(cfg.adb_path, cfg.device_serial) if adb_ok else False
    details["hyperos_security"] = (
        "ENABLED (persist.security.adbinput=1)" if hyperos_ok else "DISABLED or UNVERIFIED (value != 1)"
    )

    # (c) Scrcpy Mirror
    scrcpy_ok = check_socket_port(cfg.scrcpy_host, cfg.scrcpy_port)
    details["scrcpy"] = f"LISTENING on {cfg.scrcpy_host}:{cfg.scrcpy_port}" if scrcpy_ok else "UNAVAILABLE"

    # (d) vLLM Endpoint
    vllm_ok, model_id = check_vllm_health(cfg.vllm_base_url, cfg.vllm_api_key)
    details["vllm"] = f"HEALTHY (model: {model_id})" if vllm_ok else "DOWN or UNREACHABLE"

    all_passed = adb_ok and hyperos_ok and scrcpy_ok and vllm_ok

    return PreFlightResult(
        adb_connected=adb_ok,
        adb_rtt_min_ms=rtt_min,
        adb_rtt_avg_ms=rtt_avg,
        adb_rtt_max_ms=rtt_max,
        adb_rtt_mdev_ms=rtt_mdev,
        hyperos_security_ok=hyperos_ok,
        scrcpy_mirror_ok=scrcpy_ok,
        vllm_healthy=vllm_ok,
        vllm_model_id=model_id,
        all_passed=all_passed,
        details=details,
    )


def print_preflight_report(res: PreFlightResult) -> None:
    print("\n" + "=" * 65)
    print("FASTAGENT PHASE 0: ENVIRONMENT PRE-FLIGHT VERIFICATION")
    print("=" * 65)
    print(f"1. ADB Connectivity:    {'[PASS] OK' if res.adb_connected else '[FAIL] DISCONNECTED'}")
    print(f"   Link RTT & Jitter:   {res.details['rtt']}")
    print(f"2. HyperOS Security:    {'[PASS] OK' if res.hyperos_security_ok else '[FAIL] adbinput != 1'}")
    print(f"3. Scrcpy Video/Socket: {'[PASS] OK' if res.scrcpy_mirror_ok else '[FAIL] Socket unreachable'}")
    print(f"   Status:              {res.details['scrcpy']}")
    print(f"4. vLLM Inference Engine: {'[PASS] OK' if res.vllm_healthy else '[FAIL] Endpoint down'}")
    print(f"   Model:               {res.vllm_model_id or 'None'}")
    print("-" * 65)
    print(f"OVERALL STATUS:         {'>>> GO: ALL GATES PASSED <<<' if res.all_passed else '>>> NO-GO: PREREQUISITES FAILED <<<'}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    result = run_preflight()
    print_preflight_report(result)
    sys.exit(0 if result.all_passed else 1)
