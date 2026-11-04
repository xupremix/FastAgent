# FastAgent Architectural & Cognitive Optimizations

This document details the systems-level, algorithmic, and cognitive optimizations implemented in **FastAgent**, along with novel roadmap optimizations that push mobile agents to sub-second execution.

---

## 1. Implemented Optimizations (Phases 1 – 5)

### A. Zero-Copy Video Streaming & Binary Control Sockets (Phase 1)
- **Bottleneck:** `adb exec-out screencap -p` takes 0.6s–1.1s over USB and 2.2s–4.5s over Wi-Fi. Text injection via `adb shell input text` takes ~9.7s for 200 characters due to character-by-character serialization.
- **Implementation:** PyAV in-memory NAL decoder connected to `tango-mirror` H.264 stream. Direct binary socket for touch events. Instant clipboard injection via `set_clipboard()` and keycode paste (Ctrl+V / KEYCODE_PASTE).
- **Impact:** Frame grab dropped to **12ms – 70ms** (up to 100x speedup); text injection dropped from 4,504ms to **20.7ms** (217x speedup).

### B. Direct Android Intent Teleportation (Phase 2a)
- **Bottleneck:** Opening an app, waiting for splash screens, clicking search bars, and typing keywords takes 4–5 GUI steps and 30–45s.
- **Implementation:** Pattern-matched Activity intent filter synthesis (`am start -a android.intent.action.VIEW -d ...` and `android.intent.action.SEND`).
- **Impact:** Bypasses 3–4 intermediate GUI screens entirely, jumping to the target destination in **100ms – 120ms**.

### C. Zero-VLM CPU Micro-Perception & Snapping (Phase 2b & 2c)
- **Bottleneck:** Deterministic UI buttons ("Filtri", "Prezzo", "Cerca") do not require expensive GPU Vision-Language reasoning.
- **Implementation:** RapidOCR on host Xeon CPU (~35ms). Coordinate snapping pulls agent clicks to the exact geometric centroid of text bounding boxes.
- **Impact:** 100% click landing precision; bypasses GPU inference for deterministic sub-routines.

### D. Self-Evolving Procedural Memory & Theme Invariance (Phase 3a)
- **Bottleneck:** Standard perceptual hashing fails when an app toggles between Light Mode and Dark Mode, or when ad banners displace screen coordinates vertically.
- **Implementation:** Sobel edge gradient perceptual hashing (`sobel_edge_dhash64`) isolates structural contours from luminance. Vertical sliding-window micro-search `[-6%, +9%]` searches for shifted controls in <1.5ms. Transitions are saved in SQLite with Exponential Moving Average (EMA) confidence scores.
- **Impact:** Dark/Light mode produces **0-bit distance**. Repeated tasks replay cached macros in **15ms**, slashing step execution from 21.6s to **2.2s (9.5x speedup)**.

### E. Dynamic Foveated ROI Cropping (Phase 3b)
- **Bottleneck:** Full 320k-pixel frames produce ~660 visual tokens, costing 5.7s–9.3s in ViT prefill.
- **Implementation:** Crops active Region of Interest (ROI) at ~76k–80k pixels (~98 tokens) aligned to multiples of 28 for ViT patch grids, paired with a global 64x128 thumbnail for spatial context.
- **Impact:** ViT prefill dropped from 9,322ms down to **2,736ms – 2,856ms (3.3x speedup)**.

### F. Closed-Loop Scroll Dead-Reckoning (Phase 4a)
- **Bottleneck:** Mobile agents scroll blindly and invoke a 6.5s VLM turn just to notice that the list reached the bottom.
- **Implementation:** 2D Fourier Phase Correlation (`cv2.phaseCorrelate`) on 128x256 thumbnails on CPU in **1.40ms**. Detects list boundaries when commanded swipes produce $\le 6.0\text{px}$ movement.
- **Impact:** Eliminates wasted 6.5s VLM turns at container boundaries in <20ms.

### G. Streaming Speculative Action Execution (Phase 4b)
- **Bottleneck:** Waiting for full VLM token decode (1.5s–2.5s) stalls execution.
- **Implementation:** Stream-parses token chunks from the vLLM completion stream and dispatches the first completed action (`OPN`, `CLK`) ahead of full decode.
- **Impact:** Hides 1.5s–2.5s of model decode latency behind physical device app launch animations.

### H. Persistent UI Knowledge Graph & TAG-Nav (Phase 5a & 5b)
- **Bottleneck:** Screen coordinates drift across app updates. Redundant reasoning occurs for familiar multi-screen navigation paths.
- **Implementation:** SQLite Knowledge Graph tracking Bayesian Gaussian priors `(mean_x, mean_y, std_x, std_y)` for frequent controls via Welford's algorithm. Dijkstra / A* pathfinder (`TAG-Nav`) executes 0-VLM navigation prefixes before invoking the model for final manipulation.
- **Impact:** Sub-3ms screen localization; 0 VLM calls for prefix paths.

### I. Prior-Guided Set-of-Mark (SoM) Visual Tagger (Phase 5c)
- **Bottleneck:** Vision models suffer from coordinate regression drift when outputting raw float coordinates.
- **Implementation:** Draws numbered badges `[1]`, `[2]` directly onto the frame buffer using OpenCV in <2.5ms. Forces model to predict `CLICK_TAG(id)` mapped to exact graph centroids.

### J. Safe Autonomous Reconnaissance Crawler (Phase 5d)
- **Bottleneck:** Manual exploration of new apps is tedious and risks destructive actions.
- **Implementation:** Local-BFS crawler traversing apps via CPU OCR while enforcing a strict blacklist (`delete`, `buy`, `pay`, `logout`, `reset`).

### K. Hermes-Style Reviewer & Information Revision Loop (Phase 5e)
- **Bottleneck:** Agents output unverified extracted values; broken edges from app updates cause repeated failures.
- **Implementation:** Non-blocking background auditor validates telemetry, self-heals broken edges, promotes verified sequences into macro skills, and runs a structured **Information Revision Loop** cross-referencing extracted prices, percentages, and hardware identifiers.

---

## 2. Novel Roadmap Optimizations

### 1. Grammar-Constrained Ultra-Compact Action Tokenization
- **Concept:** Force the inference engine (vLLM / llama.cpp) to sample strictly against a BNF grammar (e.g. `CLK([0-9]{3},[0-9]{3})`).
- **Benefit:** Reduces decode length from ~95 tokens to ~8 tokens, dropping decode time from 2.39s to **<0.25s**.

### 2. Speculative Token Drafting via Compact Student Model
- **Concept:** Run a lightweight 2B visual model (e.g. Qwen2-VL-2B) on CPU/GPU to draft next actions, with the large MoE model verifying candidate drafts in parallel.
- **Benefit:** 2x–3x faster decoding on commodity hardware.

### 3. Keypoint Optical Flow for Real-Time Gesture Tracking
- **Concept:** Track Lucas-Kanade optical flow vectors during animations to detect the exact millisecond a page finishes rendering.
- **Benefit:** Drops settling detection from 77ms down to **<15ms**.
