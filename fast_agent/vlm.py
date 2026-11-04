"""High-resolution VLM Client with Prefill & Generation Timing.

Connects to OpenAI-compatible vLLM endpoints with streaming enabled to measure:
1. image_encode_ms (Resizing to factor-28 patch grid + JPEG encoding)
2. vit_prefill_ms (Time To First Token / TTFT)
3. decode_ms (Token generation duration)
4. prompt_tokens & completion_tokens (exact usage metadata)
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from openai import OpenAI
from PIL import Image

from fast_agent.config import AgentConfig
from fast_agent.contracts import NormalizedPoint, StepTiming, align_to_patch_grid
from fast_agent.foveated_roi import FoveatedCropResult


TOOL_CALL_REGEX = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
DSL_ACTION_RE = re.compile(
    r"(CLK\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)|"
    r"TYP\s*\(\s*\"([^\"]+)\"\s*\)|"
    r"SWP\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)|"
    r"KEY\s*\(\s*(\w+)\s*\)|"
    r"OPN\s*\(\s*\"([^\"]+)\"\s*\)|"
    r"WAT\s*\(\s*([0-9.]+)\s*\)|"
    r"TRM\s*\(\s*(\w+)\s*\))"
)


COMPACT_DSL_SYSTEM_PROMPT = """# Tools

You are an ultra-fast, zero-overhead mobile device automation agent.
Screen coordinates are normalized [0, 1000]. Screen resolution is 1220x2712.

# Output Format:
Keep Observation strictly under 6 words.
Then output the compact tool call immediately:
<tool_call>ACTION>ACTION</tool_call>

Available compact actions:
- CLK(x,y): Click coordinate [0-1000]
- TYP("text"): Instant text typing
- SWP(x1,y1,x2,y2): Fast swipe gesture
- KEY(enter|back|home): System key
- OPN("AppName"): Launch app
- WAT(sec): Explicit wait (ONLY when waiting for network spinner; never between taps)
- TRM(success|failure): Terminate task

Rules:
1. Act like a fast human user: do NOT emit WAT after CLK or OPN. The underlying engine adaptively settles in <30ms.
2. Chain deterministic actions in a single tool call (e.g. click search bar + type query + enter).

Examples:
1. Search in app:
Observation: Open Vinted and search query
<tool_call>OPN("Vinted")>CLK(500,80)>TYP("be quiet silent base 800")>KEY(enter)</tool_call>

2. Click listing:
Observation: Open first item
<tool_call>CLK(500,450)</tool_call>

3. Note taking:
Observation: Record item details
<tool_call>OPN("Notes")>CLK(836,853)>TYP("Silent Base 800 - 350 EUR")>TRM(success)</tool_call>

4. Terminate when done:
Observation: Goal completed
<tool_call>TRM(success)</tool_call>
"""


def parse_dsl_actions(text: str) -> List[Dict[str, Any]]:
    """Parse ultra-compact DSL actions (e.g. CLK(500,80)>TYP(\"query\"))."""
    actions: List[Dict[str, Any]] = []
    for m in DSL_ACTION_RE.finditer(text):
        full = m.group(1)
        if full.startswith("CLK"):
            actions.append({"action": "click", "coordinate": [int(m.group(2)), int(m.group(3))]})
        elif full.startswith("TYP"):
            actions.append({"action": "type", "text": m.group(4)})
        elif full.startswith("SWP"):
            actions.append(
                {
                    "action": "swipe",
                    "coordinate": [int(m.group(5)), int(m.group(6))],
                    "coordinate2": [int(m.group(7)), int(m.group(8))],
                }
            )
        elif full.startswith("KEY"):
            actions.append({"action": "key", "text": m.group(9)})
        elif full.startswith("OPN"):
            actions.append({"action": "open", "text": m.group(10)})
        elif full.startswith("WAT"):
            actions.append({"action": "wait", "time": float(m.group(11))})
        elif full.startswith("TRM"):
            actions.append({"action": "terminate", "status": m.group(12)})
    return actions

SYSTEM_PROMPT = """# Tools

You are an expert mobile device automation assistant. You interact with a touchscreen device.
Screen coordinates are normalized integers [0, 1000].
The physical screen dimensions are 1220x2712.

Available actions in tool 'mobile_use':
- click: {"action": "click", "coordinate": [x, y]} (where x, y in [0, 1000])
- type: {"action": "type", "text": "string"}
- swipe: {"action": "swipe", "coordinate": [x1, y1], "coordinate2": [x2, y2]}
- key: {"action": "key", "text": "enter"|"back"|"home"}
- open: {"action": "open", "text": "App Name"}
- wait: {"action": "wait", "time": 1.0}
- terminate: {"action": "terminate", "status": "success"|"failure"}

# Action Compacting (Chained Actions)
When predicting a sequence that does not require intermediate visual verification (e.g., clicking search bar + typing query, or opening an app and clicking create), CHAIN THEM in a JSON list:
<tool_call>
{"name": "mobile_use", "arguments": [
  {"action": "click", "coordinate": [500, 80]},
  {"action": "type", "text": "3060 12gb"}
]}
</tool_call>

# Output Format:
Keep your thinking concise (under 25 words).
If you observe important facts (e.g. price, text, status), state them in a 1-line Observation first:
Observation: <observed key facts>

Then output the <tool_call> block immediately:
<tool_call>
{"name": "mobile_use", "arguments": ...}
</tool_call>

When the goal of the instruction is satisfied, you MUST call:
<tool_call>
{"name": "mobile_use", "arguments": {"action": "terminate", "status": "success"}}
</tool_call>
"""


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract action dictionaries from model output supporting JSON and Compact DSL."""
    # 1. Try Compact DSL
    dsl_actions = parse_dsl_actions(text)
    if dsl_actions:
        return dsl_actions

    # 2. Standard JSON <tool_call> tags
    actions: List[Dict[str, Any]] = []
    matches = TOOL_CALL_REGEX.findall(text)
    if matches:
        for m in matches:
            content = m.strip()
            # Check if inner content is DSL
            inner_dsl = parse_dsl_actions(content)
            if inner_dsl:
                return inner_dsl
            try:
                data = json.loads(content)
                args = data.get("arguments", data)
                if isinstance(args, list):
                    for a in args:
                        if isinstance(a, dict):
                            actions.append(a)
                elif isinstance(args, dict):
                    actions.append(args)
            except Exception:
                pass
    if not actions:
        # Fallback: attempt direct JSON search
        try:
            bracket_match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
            if bracket_match:
                parsed = json.loads(bracket_match.group(1))
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "action" in item:
                            actions.append(item)
                elif isinstance(parsed, dict):
                    if "arguments" in parsed:
                        args = parsed["arguments"]
                        if isinstance(args, list):
                            actions.extend([x for x in args if isinstance(x, dict)])
                        elif isinstance(args, dict):
                            actions.append(args)
                    elif "action" in parsed:
                        actions.append(parsed)
        except Exception:
            pass
    return actions


class VLMClient:
    """OpenAI-compatible VLM client with streaming instrumentation."""

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        self.config = config or AgentConfig()
        self.client = OpenAI(
            base_url=self.config.vllm_base_url,
            api_key=self.config.vllm_api_key,
            timeout=45.0,
        )
        self.model_name = self.config.model_name

    def _ensure_model_name(self) -> str:
        if self.model_name:
            return self.model_name
        # Auto-discover loaded model
        try:
            models = self.client.models.list()
            if models.data:
                self.model_name = str(models.data[0].id)
                return self.model_name
        except Exception:
            pass
        return "qwen3.8-flash-next-unsloth-ud-iq4_xs"

    def prepare_image(
        self,
        screenshot: Union[bytes, Image.Image],
        max_pixels: Optional[int] = None,
    ) -> Tuple[str, float]:
        """Resize screenshot to factor-28 patch grid and encode to base64 JPEG."""
        t0 = time.perf_counter()
        img: Image.Image = (
            Image.open(io.BytesIO(screenshot)) if isinstance(screenshot, bytes) else screenshot
        )
        orig_w, orig_h = img.size

        pixel_limit = max_pixels or self.config.max_pixels
        target_w, target_h = align_to_patch_grid(
            orig_w,
            orig_h,
            factor=self.config.vit_patch_size,
            max_pixels=pixel_limit,
        )
        resized = img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        if resized.mode != "RGB":
            resized = resized.convert("RGB")

        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        encode_ms = (time.perf_counter() - t0) * 1000.0
        return b64, encode_ms

    def query(
        self,
        screenshot_png: Union[bytes, Image.Image],
        instruction: str,
        step_index: int,
        history_summary: str = "",
        plan: str = "",
        foveated_crop: Optional[FoveatedCropResult] = None,
        speculative_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Tuple[List[Dict[str, Any]], StepTiming, str]:
        """Query VLM with streaming and measure per-stage latency."""
        model = self._ensure_model_name()
        timing = StepTiming(step_index=step_index)

        def map_action_coords(act: Dict[str, Any]) -> Dict[str, Any]:
            mapped = dict(act)
            if foveated_crop is not None:
                if "coordinate" in mapped:
                    coords = mapped["coordinate"]
                    norm_x = float(coords[0]) / 1000.0
                    norm_y = float(coords[1]) / 1000.0
                    pt = foveated_crop.map_roi_to_global(norm_x, norm_y)
                    mapped["coordinate"] = [int(round(pt.x * 1000.0)), int(round(pt.y * 1000.0))]
                if "coordinate2" in mapped:
                    coords2 = mapped["coordinate2"]
                    norm_x2 = float(coords2[0]) / 1000.0
                    norm_y2 = float(coords2[1]) / 1000.0
                    pt2 = foveated_crop.map_roi_to_global(norm_x2, norm_y2)
                    mapped["coordinate2"] = [int(round(pt2.x * 1000.0)), int(round(pt2.y * 1000.0))]
            return mapped

        dispatcher = None
        if self.config.enable_speculative_execution and speculative_callback:
            from fast_agent.speculative_execution import StreamingActionDispatcher

            def spec_callback_mapped(act: Dict[str, Any]) -> None:
                speculative_callback(map_action_coords(act))

            dispatcher = StreamingActionDispatcher(dispatch_callback=spec_callback_mapped)

        input_to_encode = foveated_crop.roi_image if foveated_crop else screenshot_png
        effective_max = self.config.foveated_roi_pixels if foveated_crop else self.config.max_pixels
        b64_image, encode_ms = self.prepare_image(input_to_encode, max_pixels=effective_max)
        timing.image_encode_ms = round(encode_ms, 2)

        user_prompt = f"Goal: {instruction}\n"
        if plan:
            user_prompt += f"Plan:\n{plan}\n"
        if history_summary:
            user_prompt += f"Previous Steps:\n{history_summary}\n"
        user_prompt += "Inspect the current screenshot and output the next action(s)."

        sys_prompt = COMPACT_DSL_SYSTEM_PROMPT if self.config.enable_compact_dsl else SYSTEM_PROMPT
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": sys_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                    },
                ],
            },
        ]

        t_request = time.perf_counter()
        stream = self.client.chat.completions.create(  # type: ignore[call-overload]
            model=model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        first_token_time: Optional[float] = None
        collected_content = ""
        collected_reasoning = ""
        prompt_tokens = 0
        completion_tokens = 0

        for chunk in stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                reasoning = getattr(delta, "reasoning_content", None)
                content = getattr(delta, "content", None)

                if reasoning or content:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    if reasoning:
                        collected_reasoning += reasoning
                    if content:
                        collected_content += content
                        if dispatcher is not None:
                            dispatcher.feed_token(content)

            if getattr(chunk, "usage", None):
                usage = chunk.usage
                if usage:
                    prompt_tokens = usage.prompt_tokens or 0
                    completion_tokens = usage.completion_tokens or 0

        t_finish = time.perf_counter()

        if first_token_time is not None:
            timing.vit_prefill_ms = round((first_token_time - t_request) * 1000.0, 2)
            timing.decode_ms = round((t_finish - first_token_time) * 1000.0, 2)
        else:
            timing.vit_prefill_ms = round((t_finish - t_request) * 1000.0, 2)
            timing.decode_ms = 0.0

        timing.prompt_tokens = prompt_tokens
        timing.completion_tokens = completion_tokens

        # If no explicit content was produced, inspect reasoning block for tool calls
        raw_text = collected_content if collected_content.strip() else collected_reasoning

        if dispatcher is not None:
            time_saved = dispatcher.finalize_stream(
                raw_text,
                decode_end_ts=t_finish,
                success=True,
            )
            timing.speculative_time_saved_ms = time_saved

        actions = [map_action_coords(a) for a in parse_tool_calls(raw_text)]

        timing.actions_executed = [a.get("action", "unknown") for a in actions]

        return actions, timing, raw_text
