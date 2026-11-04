"""Device Abstraction and Execution Drivers.

Provides:
1. AndroidDevice: Base abstract protocol for device interaction.
2. AdbDevice: Live hardware ADB execution driver enforcing the Coordinate Contract.
3. MockAndroidDevice: Fully deterministic in-memory mock for unplugged testing & CI.
"""

from __future__ import annotations

import io
import re
import shlex
import subprocess
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from fast_agent.adaptive_settle import AdaptiveSettler
from fast_agent.config import AgentConfig
from fast_agent.contracts import NormalizedPoint, NormalizedRect
from fast_agent.scrcpy_client import ScrcpyControlClient
from fast_agent.stream_capture import ScrcpyStreamCapture

COMMON_APP_MAP: Dict[str, str] = {
    "subito": "it.subito",
    "vinted": "fr.vinted",
    "wallapop": "com.wallapop",
    "telegram": "org.telegram.messenger",
    "whatsapp": "com.whatsapp",
    "settings": "com.android.settings",
    "impostazioni": "com.android.settings",
    "notes": "com.miui.notes",
    "note": "com.miui.notes",
    "gallery": "com.miui.gallery",
    "galleria": "com.miui.gallery",
    "calculator": "com.miui.calculator",
    "calcolatrice": "com.miui.calculator",
    "youtube": "app.revanced.android.youtube",
    "chrome": "com.android.chrome",
    "browser": "com.android.browser",
    "playstore": "com.android.vending",
    "maps": "com.google.android.apps.maps",
    "camera": "com.android.camera",
    "clock": "com.android.deskclock",
    "orologio": "com.android.deskclock",
    "sveglia": "com.android.deskclock",
    "contacts": "com.android.contacts",
    "rubrica": "com.android.contacts",
    "gmail": "com.google.android.gm",
    "messages": "com.google.android.apps.messaging",
    "messaggi": "com.google.android.apps.messaging",
    "security": "com.miui.securitycenter",
    "sicurezza": "com.miui.securitycenter",
}


class AndroidDevice(ABC):
    """Abstract interface for Android device automation."""

    @abstractmethod
    def screencap(self) -> bytes:
        """Capture the current screen and return raw PNG bytes."""
        raise NotImplementedError

    @abstractmethod
    def get_latest_image(self) -> Optional[Image.Image]:
        """Return latest PIL Image directly (0-copy if supported, bypassing PNG encoding)."""
        raise NotImplementedError

    @abstractmethod
    def tap(self, point: NormalizedPoint) -> None:
        """Tap at normalized coordinate (0.0 to 1.0)."""
        raise NotImplementedError

    @abstractmethod
    def swipe(
        self,
        start: NormalizedPoint,
        end: NormalizedPoint,
        duration_ms: int = 300,
    ) -> None:
        """Swipe from normalized start to normalized end over duration_ms."""
        raise NotImplementedError

    @abstractmethod
    def input_text(self, text: str) -> None:
        """Type text into the focused field."""
        raise NotImplementedError

    @abstractmethod
    def press_key(self, keycode: int) -> None:
        """Inject an Android keyevent (e.g. 3 for HOME, 4 for BACK, 66 for ENTER)."""
        raise NotImplementedError

    @abstractmethod
    def press_home(self) -> None:
        """Return to home launcher."""
        raise NotImplementedError

    @abstractmethod
    def press_back(self) -> None:
        """Press Back button."""
        raise NotImplementedError

    @abstractmethod
    def get_clipboard(self) -> str:
        """Read device clipboard."""
        raise NotImplementedError

    @abstractmethod
    def set_clipboard(self, text: str) -> None:
        """Set device clipboard contents."""
        raise NotImplementedError

    @abstractmethod
    def get_foreground_package(self) -> str:
        """Get the currently focused package name."""
        raise NotImplementedError

    @abstractmethod
    def launch_package(self, package_name: str) -> None:
        """Launch the main activity of an application."""
        raise NotImplementedError

    @abstractmethod
    def resolve_package_name(self, app_name: str) -> Optional[str]:
        """Resolve a colloquial or localized application name to an Android package name."""
        raise NotImplementedError

    @abstractmethod
    def resolve_and_launch_app(self, app_name: str) -> bool:
        """Resolve and launch an application by colloquial name or package."""
        raise NotImplementedError

    @abstractmethod
    def force_stop(self, package_name: str) -> None:
        """Force-stop an application."""
        raise NotImplementedError

    @abstractmethod
    def wake_screen(self) -> None:
        """Ensure device is awake and keyguard is dismissed."""
        raise NotImplementedError

    @abstractmethod
    def settle(self, timeout_s: Optional[float] = None) -> float:
        """Wait for screen stability and return elapsed milliseconds."""
        raise NotImplementedError

    @abstractmethod
    def pre_run_reset(self, packages_to_stop: Optional[List[str]] = None) -> None:
        """Deterministic state reset before benchmark runs."""
        raise NotImplementedError

    def get_last_delta_mask(self) -> Optional[np.ndarray[Any, Any]]:
        """Optional spatial delta heatmap from adaptive settling."""
        return None


class AdbDevice(AndroidDevice):
    """Live ADB execution driver for physical/wireless devices."""

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        self.config = config or AgentConfig()
        self.serial = self.config.device_serial
        self.adb = self.config.adb_path
        self.width = self.config.physical_width
        self.height = self.config.physical_height
        self._scrcpy: Optional[ScrcpyControlClient] = None
        self._stream_capture: Optional[ScrcpyStreamCapture] = None
        self._settler: Optional[AdaptiveSettler] = None
        self._heartbeat_running = False
        self._heartbeat_thread: Optional[subprocess.Popen[bytes]] = None
        self._installed_packages_cache: Optional[List[str]] = None

        if self.config.enable_scrcpy_socket:
            self._scrcpy = ScrcpyControlClient(self.config)
            self._scrcpy.connect()

        if self.config.enable_stream_capture:
            self._stream_capture = ScrcpyStreamCapture(self.config)
            self._stream_capture.start()

        if self.config.enable_adaptive_settling:
            self._settler = AdaptiveSettler()

        if self.config.enable_os_tuning:
            self.setup_os_tuning()

    def setup_os_tuning(self) -> None:
        """Phase 1a: Disable animations and start Wi-Fi anti-PSM heartbeat."""
        self._run_shell("settings put global window_animation_scale 0")
        self._run_shell("settings put global transition_animation_scale 0")
        self._run_shell("settings put global animator_duration_scale 0")
        # Start background ping heartbeat to prevent Wi-Fi Power Save Mode (PSM)
        if not self._heartbeat_running:
            target_ip = self.serial.split(":")[0] if ":" in self.serial else self.serial
            try:
                self._heartbeat_thread = subprocess.Popen(
                    ["ping", "-i", "0.25", target_ip],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._heartbeat_running = True
            except Exception:
                pass

    def teardown_os_tuning(self) -> None:
        """Phase 1a: Restore system animation scales and kill heartbeat."""
        self._run_shell("settings put global window_animation_scale 1")
        self._run_shell("settings put global transition_animation_scale 1")
        self._run_shell("settings put global animator_duration_scale 1")
        if self._heartbeat_thread:
            try:
                self._heartbeat_thread.terminate()
            except Exception:
                pass
            self._heartbeat_thread = None
        self._heartbeat_running = False

    def close(self) -> None:
        if self._stream_capture:
            self._stream_capture.stop()
            self._stream_capture = None
        if self._scrcpy:
            self._scrcpy.close()
            self._scrcpy = None
        if self.config.enable_os_tuning:
            self.teardown_os_tuning()

    def _run_adb(
        self,
        args: List[str],
        timeout: float = 15.0,
        check: bool = False,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [self.adb, "-s", self.serial] + args
        return subprocess.run(
            cmd,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            check=check,
        )

    def _run_shell(
        self,
        shell_cmd: str,
        timeout: float = 15.0,
        check: bool = False,
    ) -> str:
        res = self._run_adb(["shell", shell_cmd], timeout=timeout, check=check)
        return res.stdout.strip()

    def screencap(self) -> bytes:
        """Screencap via Zero-Copy stream if enabled, else adb exec-out screencap -p."""
        if self.config.enable_stream_capture and self._stream_capture:
            return self._stream_capture.get_latest_png_bytes()
        cmd = [self.adb, "-s", self.serial, "exec-out", "screencap", "-p"]
        res = subprocess.run(cmd, capture_output=True, timeout=10.0, check=True)
        return res.stdout

    def get_latest_image(self) -> Optional[Image.Image]:
        """Return latest PIL Image directly (0-copy, bypassing PNG encoding)."""
        if self.config.enable_stream_capture and self._stream_capture:
            return self._stream_capture.get_latest_image()
        raw_bytes = self.screencap()
        return Image.open(io.BytesIO(raw_bytes))

    def settle(self, timeout_s: Optional[float] = None) -> float:
        """Phase 1d: Adaptive stabilization wait (replaces static sleep)."""
        if self.config.enable_adaptive_settling and self._settler:
            return self._settler.wait_until_settled(self.get_latest_image, timeout_s=timeout_s)
        time.sleep(0.5)
        return 500.0

    def get_last_delta_mask(self) -> Optional[np.ndarray[Any, Any]]:
        if self._settler:
            return self._settler.last_delta_mask
        return None

    def tap(self, point: NormalizedPoint) -> None:
        """Convert normalized coordinates to physical pixels and tap."""
        if self.config.enable_scrcpy_socket and self._scrcpy:
            if self._scrcpy.tap(point):
                return
        # ADB fallback
        px, py = point.to_physical(self.width, self.height)
        self._run_shell(f"input tap {px} {py}")

    def swipe(
        self,
        start: NormalizedPoint,
        end: NormalizedPoint,
        duration_ms: int = 300,
    ) -> None:
        """Convert normalized coordinates to physical pixels and swipe."""
        if self.config.enable_scrcpy_socket and self._scrcpy:
            if self._scrcpy.swipe(start, end, duration_ms):
                return
        # ADB fallback
        x1, y1 = start.to_physical(self.width, self.height)
        x2, y2 = end.to_physical(self.width, self.height)
        self._run_shell(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")

    def input_text(self, text: str) -> None:
        """Inject text: Sub-15ms clipboard injection if enabled, else ADB fallback."""
        if self.config.enable_fast_clipboard and self.config.enable_scrcpy_socket and self._scrcpy:
            if self._scrcpy.inject_text_instant(text):
                return

        # Baseline adb shell input text serialization
        formatted = text.replace(" ", "%s")
        escaped = ""
        for char in formatted:
            if char in r"'\$()<>|;&*~`":
                escaped += f"\\{char}"
            else:
                escaped += char
        timeout = max(15.0, len(text) * 0.1 + 10.0)
        self._run_shell(f"input text {escaped}", timeout=timeout)

    def press_key(self, keycode: int) -> None:
        if self.config.enable_scrcpy_socket and self._scrcpy:
            if self._scrcpy.press_key(keycode):
                return
        self._run_shell(f"input keyevent {keycode}")

    def press_home(self) -> None:
        self.press_key(3)

    def press_back(self) -> None:
        self.press_key(4)

    def get_clipboard(self) -> str:
        """Read clipboard via service call clipboard (or fallback empty string)."""
        try:
            res = self._run_shell("cmd clipboard get", timeout=5.0)
            return res
        except Exception:
            return ""

    def set_clipboard(self, text: str) -> None:
        """Set clipboard text: Instant socket injection if enabled, else adb."""
        if self.config.enable_scrcpy_socket and self._scrcpy:
            if self._scrcpy.set_clipboard(text):
                return
        try:
            escaped = shlex.quote(text)
            self._run_shell(f"cmd clipboard set {escaped}", timeout=5.0)
        except Exception:
            pass

    def get_foreground_package(self) -> str:
        """Extract focused package from dumpsys window."""
        res = self._run_shell("dumpsys window | grep -E 'mCurrentFocus|mFocusedApp'")
        match = re.search(r"([a-zA-Z0-9_.]+)/[a-zA-Z0-9_.]+", res)
        if match:
            return match.group(1)
        return ""

    def get_installed_packages(self, refresh: bool = False) -> List[str]:
        """Fetch and cache list of installed package names on device."""
        if self._installed_packages_cache is not None and not refresh:
            return self._installed_packages_cache
        res = self._run_shell("pm list packages")
        pkgs: List[str] = []
        for line in res.splitlines():
            line = line.strip()
            if line.startswith("package:"):
                line = line[len("package:"):].strip()
            if line:
                pkgs.append(line)
        self._installed_packages_cache = sorted(set(pkgs))
        return self._installed_packages_cache

    def resolve_package_name(self, app_name: str) -> Optional[str]:
        """Resolve colloquial, localized or partial app name to installed Android package."""
        raw = app_name.strip()
        if not raw:
            return None
        norm = raw.lower().replace(" ", "").replace("-", "").replace("_", "")

        # 1. Alias dictionary
        for alias, pkg in COMMON_APP_MAP.items():
            if alias == norm or alias in norm or norm in alias:
                return pkg

        # 2. Substring match against installed packages
        pkgs = self.get_installed_packages()
        for pkg in pkgs:
            clean = pkg.lower().replace(".", "").replace("-", "").replace("_", "")
            if norm in clean:
                return pkg

        # 3. Direct check if raw string looks like a valid package
        if "." in raw and raw in pkgs:
            return raw

        return None

    def resolve_and_launch_app(self, app_name: str) -> bool:
        """Resolve app name and launch reliably using am/monkey."""
        pkg = self.resolve_package_name(app_name)
        if not pkg:
            pkg = app_name.strip()

        if pkg == "com.android.settings":
            self._run_shell("am start -a android.settings.SETTINGS")
            return True

        # Try monkey launcher first (fastest, standard category launcher)
        res = self._run_shell(f"monkey -p {pkg} -c android.intent.category.LAUNCHER 1")
        if "Events injected: 1" in res or "No activities" not in res:
            return True

        # Fallback to cmd package resolve-activity
        resolve_out = self._run_shell(f"cmd package resolve-activity --brief {pkg}")
        lines = [line.strip() for line in resolve_out.splitlines() if line.strip() and not line.startswith("priority=")]
        if lines:
            component = lines[-1]
            if "/" in component:
                self._run_shell(f"am start -n {component}")
                return True

        # Fallback to direct monkey
        self._run_shell(f"monkey -p {pkg} 1")
        return True

    def launch_package(self, package_name: str) -> None:
        """Launch package main activity using monkey or am start."""
        self.resolve_and_launch_app(package_name)

    def force_stop(self, package_name: str) -> None:
        self._run_shell(f"am force-stop {package_name}")

    def wake_screen(self) -> None:
        """Ensure device display is awake and dismiss keyguard."""
        self._run_shell("input keyevent 224")  # KEYCODE_WAKEUP
        self._run_shell("wm dismiss-keyguard")

    def pre_run_reset(self, packages_to_stop: Optional[List[str]] = None) -> None:
        """Deterministic pre-run reset:
        1. Wake screen & dismiss lock
        2. Force-stop target packages
        3. Clear clipboard
        4. Settle screen
        """
        self.wake_screen()
        pkgs = packages_to_stop or [
            "com.wallapop",
            "com.miui.notes",
            "com.android.settings",
            "org.telegram.messenger",
            "com.miui.gallery",
            "com.miui.calculator",
        ]
        for pkg in pkgs:
            self.force_stop(pkg)

        self.set_clipboard("")
        if pkgs:
            self.press_home()
            time.sleep(0.4)


class MockAndroidDevice(AndroidDevice):
    """Deterministic in-memory mock for unplugged testing and CI."""

    def __init__(
        self,
        width: int = 1220,
        height: int = 2712,
        simulated_capture_delay_s: float = 0.0,
        simulated_input_delay_s: float = 0.0,
    ) -> None:
        self.width = width
        self.height = height
        self.simulated_capture_delay_s = simulated_capture_delay_s
        self.simulated_input_delay_s = simulated_input_delay_s
        self.action_history: List[Tuple[str, Any]] = []
        self.current_package: str = "com.miui.home"
        self.clipboard: str = ""
        self.is_screen_on: bool = True
        self.canned_screenshot: Optional[bytes] = None

    def screencap(self) -> bytes:
        if self.simulated_capture_delay_s > 0:
            time.sleep(self.simulated_capture_delay_s)
        self.action_history.append(("screencap", None))
        if self.canned_screenshot:
            return self.canned_screenshot
        # Generate synthetic test frame
        img = Image.new("RGB", (self.width, self.height), color=(30, 30, 30))
        draw = ImageDraw.Draw(img)
        draw.text((100, 200), f"Active: {self.current_package}", fill=(255, 255, 255))
        draw.rectangle([50, 50, 200, 150], outline=(0, 255, 0), width=3)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def get_latest_image(self) -> Optional[Image.Image]:
        raw = self.screencap()
        return Image.open(io.BytesIO(raw))

    def tap(self, point: NormalizedPoint) -> None:
        if self.simulated_input_delay_s > 0:
            time.sleep(self.simulated_input_delay_s)
        px, py = point.to_physical(self.width, self.height)
        self.action_history.append(("tap", (point, px, py)))

    def swipe(
        self,
        start: NormalizedPoint,
        end: NormalizedPoint,
        duration_ms: int = 300,
    ) -> None:
        if self.simulated_input_delay_s > 0:
            time.sleep(self.simulated_input_delay_s)
        p1 = start.to_physical(self.width, self.height)
        p2 = end.to_physical(self.width, self.height)
        self.action_history.append(("swipe", (p1, p2, duration_ms)))

    def input_text(self, text: str) -> None:
        if self.simulated_input_delay_s > 0:
            time.sleep(self.simulated_input_delay_s)
        self.action_history.append(("input_text", text))

    def press_key(self, keycode: int) -> None:
        self.action_history.append(("press_key", keycode))

    def press_home(self) -> None:
        self.action_history.append(("press_home", None))
        self.current_package = "com.miui.home"

    def press_back(self) -> None:
        self.action_history.append(("press_back", None))

    def get_clipboard(self) -> str:
        self.action_history.append(("get_clipboard", None))
        return self.clipboard

    def set_clipboard(self, text: str) -> None:
        self.action_history.append(("set_clipboard", text))
        self.clipboard = text

    def get_foreground_package(self) -> str:
        self.action_history.append(("get_foreground_package", None))
        return self.current_package

    def resolve_package_name(self, app_name: str) -> Optional[str]:
        norm = app_name.lower().strip()
        for k, v in COMMON_APP_MAP.items():
            if k in norm or norm in k:
                return v
        return app_name

    def resolve_and_launch_app(self, app_name: str) -> bool:
        pkg = self.resolve_package_name(app_name) or app_name
        self.launch_package(pkg)
        return True

    def launch_package(self, package_name: str) -> None:
        self.action_history.append(("launch_package", package_name))
        self.current_package = package_name

    def force_stop(self, package_name: str) -> None:
        self.action_history.append(("force_stop", package_name))
        if self.current_package == package_name:
            self.current_package = "com.miui.home"

    def wake_screen(self) -> None:
        self.action_history.append(("wake_screen", None))
        self.is_screen_on = True

    def settle(self, timeout_s: Optional[float] = None) -> float:
        self.action_history.append(("settle", timeout_s))
        return 50.0

    def pre_run_reset(self, packages_to_stop: Optional[List[str]] = None) -> None:
        self.action_history.append(("pre_run_reset", packages_to_stop))
        self.wake_screen()
        pkgs = packages_to_stop or []
        for pkg in pkgs:
            self.force_stop(pkg)
        self.set_clipboard("")
        self.press_home()
