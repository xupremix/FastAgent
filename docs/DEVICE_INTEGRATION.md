# Universal Android Device Integration Guide

FastAgent is designed to operate on **any Android device** (Android 10 through 16+) running physical hardware or emulators. This guide explains how to prepare, connect, and tune any device brand for low-latency automation.

---

## 1. Universal Prerequisites

Regardless of device manufacturer, every Android phone requires:

1. **Developer Options Enabled:**
   - Go to **Settings** > **About Phone**.
   - Tap **Build Number** (or OS Version) **7 times** until "You are now a developer!" appears.

2. **USB Debugging Enabled:**
   - Go to **Settings** > **System** (or **Additional Settings**) > **Developer Options**.
   - Enable **USB Debugging**.

3. **Wireless ADB Setup (Zero-Cable):**
   - Connect the phone and host PC to the same Wi-Fi network (5GHz recommended for <3ms RTT).
   - If using Android 11+: Enable **Wireless Debugging** in Developer Options.
   - Initial one-time USB pairing:
     ```bash
     adb tcpip 5555
     adb connect <PHONE_IP>:5555
     ```
   - Verify connection:
     ```bash
     adb devices
     # Output: <PHONE_IP>:5555    device
     ```

4. **Zero-Latency UI Animation Tuning:**
   - Eliminate system animation overhead to allow sub-second screen transitions:
     ```bash
     adb -s <SERIAL> shell settings put global window_animation_scale 0.0
     adb -s <SERIAL> shell settings put global transition_animation_scale 0.0
     adb -s <SERIAL> shell settings put global animator_duration_scale 0.0
     ```

---

## 2. Manufacturer-Specific Setup & Quirks

Different Android skins (HyperOS, One UI, OxygenOS) enforce custom security policies on touch injection. Follow your manufacturer's specific instructions below:

### A. Xiaomi / POCO / Redmi (HyperOS & MIUI)

HyperOS blocks programmatic touch injection by default unless an explicit security toggle is active.

1. **Mandatory Security Permission:**
   - In Developer Options, enable **"USB debugging (Security settings)"** (Allow granting permissions and simulating input via USB debugging).
   - *Note:* Xiaomi requires a SIM card to be inserted during initial toggle authorization.
   - Verify that the property is set:
     ```bash
     adb shell getprop persist.security.adbinput
     # Must return 1
     ```
2. **Disable MIUI/HyperOS Optimizations (Optional):**
   - Scroll to the bottom of Developer Options and disable "Turn on system optimizations" if background daemons are prematurely killed.

---

### B. Google Pixel & Stock Android (Motorola, Nothing OS, Sony)

Stock Android requires zero proprietary security flags:

1. Enable **USB Debugging** and **Wireless Debugging**.
2. Run FastAgent directly:
   ```bash
   adb connect <PIXEL_IP>:5555
   python3 -m fast_agent.preflight
   ```

---

### C. Samsung Galaxy (One UI 5.x / 6.x / 7.x)

1. **Disable Auto Blocker:**
   - Go to **Settings** > **Security and privacy** > **Auto Blocker**.
   - Toggle **Off** (Auto Blocker restricts ADB script commands and sideloading).
2. **Disable Knox Permission Prompts:**
   - Ensure "Verify apps over USB" is disabled in Developer Options to prevent blocking background Activity launches.

---

### D. OnePlus / OPPO / Realme (OxygenOS & ColorOS)

1. **Disable Permission Monitoring:**
   - In Developer Options, enable **"Disable permission monitoring"**.
   - This prevents ColorOS from popping up interactive permission prompts whenever FastAgent sends key codes or touch coordinates.

---

## 3. High-Throughput Scrcpy Server Setup

FastAgent achieves sub-50ms frame grab latency by consuming a real-time H.264 stream from `scrcpy-server`.

1. Install Scrcpy or run the bundled `tango-mirror` daemon:
   ```bash
   scrcpy --tcpip=<PHONE_IP>:5555 --video-codec=h264 --max-size=1600 --max-fps=60 --no-audio --tunnel-host=127.0.0.1 --port=8010
   ```
2. For headless Linux servers (without physical display attached), run inside Xvfb:
   ```bash
   xvfb-run -a scrcpy --tcpip=<PHONE_IP>:5555 --no-playback --port=8010
   ```

---

## 4. Multi-Device Fleet Configuration

To run FastAgent across a fleet of devices, specify the serial number in `fast_agent/config.py` or via command-line arguments:

```bash
# POCO X6 Pro on Wi-Fi
python3 fast_agent/benchmark.py --serial 192.168.10.108:5555

# Pixel 8 Pro on USB
python3 fast_agent/benchmark.py --serial 3A151JEHN02319

# Samsung S24 on Wi-Fi
python3 fast_agent/benchmark.py --serial 192.168.10.115:5555
```
