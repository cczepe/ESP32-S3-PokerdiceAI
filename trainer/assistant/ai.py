"""Calls to the Anthropic API (Claude). The key only comes from settings.api_key()."""
import base64
import json
import os
import re

import settings


class AIError(Exception):
    pass


def _client():
    try:
        import anthropic
    except ImportError:
        raise AIError("Package “anthropic” is missing. Install it: pip install anthropic")
    key, _ = settings.api_key()
    if not key:
        raise AIError("No API key entered (step 0 “Setup”).")
    return anthropic, anthropic.Anthropic(api_key=key, timeout=300.0, max_retries=2)


def _error_text(anthropic, e):
    if isinstance(e, anthropic.AuthenticationError):
        return "Key invalid (401). Please check it in the setup."
    if isinstance(e, anthropic.PermissionDeniedError):
        return "No permission (403) for this model/key."
    if isinstance(e, anthropic.NotFoundError):
        return "Model not found (404). Please check the model name in the setup."
    if isinstance(e, anthropic.RateLimitError):
        return "Too many requests or credit limit (429). Wait a moment or check your credit."
    if isinstance(e, anthropic.APIConnectionError):
        return "No connection to the Anthropic API (internet?)."
    if isinstance(e, anthropic.APIStatusError):
        return f"API error {e.status_code}: {str(e)[:200]}"
    return f"Error: {e}"


def _log(model, tool, attempts, answer):
    """Raw answer in ~/.pokerdice/ai_last_answer.txt (local, without key) - to look at when something goes wrong."""
    try:
        p = os.path.join(settings.config_folder(), "ai_last_answer.txt")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"Model: {model}\nTool: {tool['name'] if tool else '-'}\nAttempts: {attempts}\n")
            f.write(f"stop_reason: {answer.stop_reason}\nTokens: {answer.usage.input_tokens} in, {answer.usage.output_tokens} out\n\n")
            for b in answer.content:
                kind = getattr(b, "type", "?")
                content = getattr(b, "text", None) or getattr(b, "thinking", None) or json.dumps(getattr(b, "input", ""), ensure_ascii=False)
                f.write(f"--- block {kind} ({len(content or '')} characters) ---\n{content}\n\n")
        return p
    except OSError:
        return None


def _blocks(answer):
    """Short description of the answer blocks, e.g. “text 25000 characters, tool_use 0 characters”."""
    parts = []
    for b in answer.content:
        kind = getattr(b, "type", "?")
        content = getattr(b, "text", None) or getattr(b, "thinking", None) or json.dumps(getattr(b, "input", ""), ensure_ascii=False)
        parts.append(f"{kind} {len(content or '')} characters")
    return ", ".join(parts) or "empty"


def _request(model, content, max_tokens=1500, tool=None, max_tokens_free=None):
    """Without tool: returns (text, usage).
    With tool (tool use with a fixed schema): Claude must fill in exactly this form;
    the API returns the fields as valid data -> (dict, usage). No JSON parsing from text.
    max_tokens_free: limit if the model refuses the forced tool and answers freely (with reasoning first)."""
    anthropic, client = _client()
    extra = {}
    attempts = []
    if tool:
        # force the tool: Claude fills in the form right away instead of writing freely first
        extra = {"tools": [tool], "tool_choice": {"type": "tool", "name": tool["name"]}}
    limit = max_tokens
    try:
        try:
            attempts.append("tool forced" if tool else "text")
            answer = client.messages.create(model=model, max_tokens=limit,
                                            messages=[{"role": "user", "content": content}], **extra)
        except anthropic.BadRequestError as e:
            # Some models do not allow forcing -> "auto" with a clear instruction in the prompt;
            # then Claude often writes its reasoning first -> more room.
            if not tool or "tool_choice" not in str(e):
                raise
            attempts[-1] += f" -> refused ({str(e)[:120]})"
            extra["tool_choice"] = {"type": "auto"}
            limit = max_tokens_free or max_tokens
            attempts.append(f"free (auto), limit {limit}")
            answer = client.messages.create(model=model, max_tokens=limit,
                                            messages=[{"role": "user", "content": content}], **extra)
    except anthropic.APIError as e:
        raise AIError(_error_text(anthropic, e))
    usage = {"input_tokens": answer.usage.input_tokens, "output_tokens": answer.usage.output_tokens}
    file = _log(model, tool, " | ".join(attempts), answer)
    print(f"[AI] {model} · {' | '.join(attempts)} · stop {answer.stop_reason} · "
          f"{answer.usage.output_tokens} tokens · {_blocks(answer)}", flush=True)
    if tool:
        if answer.stop_reason == "max_tokens":
            raise AIError(f"Answer was cut off (limit of {limit} tokens reached; {' | '.join(attempts)}; "
                          f"answer: {_blocks(answer)}). The whole answer is in {file or 'ai_last_answer.txt'}.")
        for b in answer.content:
            if getattr(b, "type", "") == "tool_use" and b.name == tool["name"]:
                return dict(b.input), usage
        # fallback: form not used -> read JSON from the text (as well as possible)
        text = "".join(getattr(b, "text", "") for b in answer.content)
        return json_from_text(text), usage
    text = "".join(getattr(b, "text", "") for b in answer.content)
    return text, usage


def connection_test(model):
    text, usage = _request(model, "Answer only with the word: OK", max_tokens=10)
    return {"answer": text.strip(), "usage": usage, "model": model}


def _image(jpeg_bytes):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.b64encode(jpeg_bytes).decode("ascii")}}


def json_from_text(text):
    text = re.sub(r"```(?:json)?", "", text)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise AIError("Answer contained no JSON: " + text[:200])
    try:
        return json.loads(m.group(0))
    except ValueError as e:
        raise AIError(f"JSON in the answer not readable ({e}): " + text[:200])


def _hints(title, text):
    return ("\n\n" + title + "\n" + text.strip()) if (text or "").strip() else ""


# ====================================================================== step 1: camera setup
CAMERA_SETUP_PROMPT = """You help with a feasibility check for a dice recognition.
Image 1: photo of a small camera board (OV2640/OV3660 class sensor) of poker dice on a surface (planned: dice tray with green felt).
Image 2: enlarged crops of the automatically found spots, numbered 1 to {n} (pixels deliberately enlarged coarsely,
so that the real resolution is visible). A spot may also be no die or contain two dice.
Expected symbols (not confirmed yet): {symbols}. The dice may also carry other symbols (e.g. pips or pictures) -
then describe the top symbol briefly in your own words. Some symbols may be coloured.

For each numbered spot, judge whether the symbol on the TOP face can be DETERMINED unambiguously.
Everything counts: colour, kind (pips or figure) and drawing - the drawing does not have to be sharply readable
if colour and kind already make the symbol unambiguous.{features}
Also estimate the camera elevation (0° = flat from the side, 90° = straight from above).
IMPORTANT: answer only by calling the tool “camera_setup_result” - no other text. Short texts, in English."""

CAMERA_SETUP_TOOL = {
    "name": "camera_setup_result",
    "description": "Enter the result of the assessment of a camera board photo.",
    "input_schema": {
        "type": "object",
        "properties": {
            "dice": {
                "type": "array",
                "description": "one entry per numbered spot from image 2",
                "items": {
                    "type": "object",
                    "properties": {
                        "nr": {"type": "integer"},
                        "is_die": {"type": "boolean"},
                        "several": {"type": "boolean", "description": "spot contains several dice"},
                        "top_symbol": {"type": "string", "description": "symbol on the top face, or ? if unclear"},
                        "readability": {"type": "string", "enum": ["good", "partly", "no"],
                                        "description": "good = unambiguous, partly = unsure, no = cannot be determined"},
                        "remark": {"type": "string", "description": "at most one short sentence"},
                    },
                    "required": ["nr", "is_die", "top_symbol", "readability"],
                },
            },
            "dice_in_image": {"type": "integer"},
            "elevation_deg": {"type": "number"},
            "image_quality": {"type": "string", "description": "short: sharpness, light, colours"},
            "recommendation": {"type": "string", "description": "one sentence on what could be improved in the camera position"},
        },
        "required": ["dice", "dice_in_image", "elevation_deg", "image_quality", "recommendation"],
    },
}


def assess_camera_setup(model, original_jpeg, mosaic_jpeg, n, symbols="9, 10, J, Q, K, A", features=""):
    content = [
        _image(original_jpeg),
        _image(mosaic_jpeg),
        {"type": "text", "text": CAMERA_SETUP_PROMPT.format(
            n=n, symbols=symbols, features=_hints("How to recognise the symbols of these dice (owner's notes):", features))},
    ]
    data, usage = _request(model, content, max_tokens=6000, tool=CAMERA_SETUP_TOOL, max_tokens_free=12000)
    data["_usage"] = usage
    data["_model"] = model
    return data


# ====================================================================== step 3: symbols of the 6 faces
FACES_PROMPT = """You see {n} photos of the faces of ONE poker die (each photographed from above, die on white paper,
maybe with a ruler next to it). The images are in this order: {names}.
Determine the symbol on the die face for each photo. Usual poker dice have 9, 10, J (jack), Q (queen), K (king), A (ace);
the symbols may be shown as pips, figures or signs and may be coloured.{features}
For each photo give:
- short_name: short class name, preferably one of 9, 10, J, Q, K, A (otherwise an own short name)
- kind: pips, figure, sign, letter, number or other
- color: main colour of the symbol (e.g. black, red, light blue, dark blue)
- position: where on the face (e.g. centred, spread over the whole face)
- description: one short sentence how to recognise the symbol
All 6 short_names must be different. Answer only via the tool “faces_result”. English."""

FACES_TOOL = {
    "name": "faces_result",
    "description": "Enter the recognised symbols of the die faces.",
    "input_schema": {
        "type": "object",
        "properties": {
            "faces": {"type": "array", "items": {"type": "object", "properties": {
                "photo": {"type": "string"},
                "short_name": {"type": "string"},
                "kind": {"type": "string", "enum": ["pips", "figure", "sign", "letter", "number", "other"]},
                "color": {"type": "string"},
                "position": {"type": "string"},
                "description": {"type": "string"},
            }, "required": ["photo", "short_name", "kind", "color"]}},
            "remark": {"type": "string"},
        },
        "required": ["faces"],
    },
}


def recognise_faces(model, images, features=""):
    """images: list of (photo_name, jpeg_bytes)."""
    content = []
    for name, b in images:
        content.append({"type": "text", "text": f"Photo {name}:"})
        content.append(_image(b))
    content.append({"type": "text", "text": FACES_PROMPT.format(
        n=len(images), names=", ".join(n for n, _ in images),
        features=_hints("Owner's notes on the dice:", features))})
    data, usage = _request(model, content, max_tokens=6000, tool=FACES_TOOL, max_tokens_free=12000)
    data["_usage"] = usage
    data["_model"] = model
    return data


# ====================================================================== step 4: layout (die net)
LAYOUT_PROMPT = """You help to record the layout of the symbols on a poker die as a die net.
First you see the 6 faces one by one (cropped, with their name), the way they are placed into the net.
Then {m} corner photos: each taken at an angle onto a corner, so that 3 adjacent faces are visible.

The net (unfolded die, seen from outside) has this cross shape:
            [back ]
    [left ] [ top ] [right]
            [front]
            [bottom]
“top” borders back, left, right, front. “front” also borders “bottom” at its lower edge.
Opposite faces: top-bottom, front-back, left-right.
“rotation” (0, 90, 180, 270) = by how many degrees clockwise the face image (as shown above on its own)
must be rotated in the net cell so that it fits its neighbours when the net is looked at normally.

Choose freely which symbol lies on “top” (e.g. one that is clearly visible on the corner photos) and derive the rest
from the corner photos: which symbols are adjacent, which are opposite, and how they are rotated to each other.
Symbols: {classes}. Each exactly once.
Answer only via the tool “layout_result”, without explanation before it. English, remark at most 2 sentences."""

NET_FACES = ["back", "left", "top", "right", "front", "bottom"]

LAYOUT_TOOL = {
    "name": "layout_result",
    "description": "Enter the die net.",
    "input_schema": {
        "type": "object",
        "properties": {
            "net": {"type": "object", "properties": {
                f: {"type": "object", "properties": {
                    "symbol": {"type": "string"},
                    "rotation": {"type": "integer", "enum": [0, 90, 180, 270]}},
                    "required": ["symbol", "rotation"]}
                for f in NET_FACES},
                "required": NET_FACES},
            "opposite": {"type": "array", "items": {"type": "array", "items": {"type": "string"}},
                         "description": "3 pairs of opposite symbols"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "remark": {"type": "string"},
        },
        "required": ["net", "opposite"],
    },
}


def suggest_layout(model, faces, corners, classes):
    """faces: list (symbol, jpeg_face_crop); corners: list (photo_name, jpeg)."""
    content = []
    for sym, b in faces:
        content.append({"type": "text", "text": f"Face “{sym}”:"})
        content.append(_image(b))
    for name, b in corners:
        content.append({"type": "text", "text": f"Corner photo {name}:"})
        content.append(_image(b))
    content.append({"type": "text", "text": LAYOUT_PROMPT.format(m=len(corners), classes=", ".join(classes))})
    # The layout often needs longer thinking before the form (corner by corner) -> generous limit.
    # Only what is actually used is paid for.
    data, usage = _request(model, content, max_tokens=4000, tool=LAYOUT_TOOL, max_tokens_free=16000)
    data["_usage"] = usage
    data["_model"] = model
    return data


# ====================================================================== step 9: label real throws
THROW_PROMPT = """You help to label test images for a dice recognition.
Image 1: photo of the camera board (OV2640/OV3660 class sensor) of poker dice in a dice tray.
Image 2: enlarged crops of the automatically found spots, numbered 1 to {n}. A spot may also be no die
or contain several dice.
Possible symbols (classes): {classes}. “?” means: the die is tilted or the top symbol is not unambiguous.{features}

Determine the symbol on the TOP face for each numbered spot. Also count all dice in the whole image 1.
Answer only via the tool “throw_result”."""


def throw_tool(classes):
    return {
        "name": "throw_result",
        "description": "Enter the symbols of the dice of a test image.",
        "input_schema": {
            "type": "object",
            "properties": {
                "dice": {"type": "array", "items": {"type": "object", "properties": {
                    "nr": {"type": "integer"},
                    "is_die": {"type": "boolean"},
                    "several": {"type": "boolean"},
                    "symbol": {"type": "string", "enum": list(classes)},
                    "sure": {"type": "boolean", "description": "true if you are sure about the symbol"},
                }, "required": ["nr", "is_die", "symbol"]}},
                "dice_in_image": {"type": "integer"},
                "remark": {"type": "string"},
            },
            "required": ["dice", "dice_in_image"],
        },
    }


def label_throw(model, original_jpeg, mosaic_jpeg, n, classes, features=""):
    content = [_image(original_jpeg), _image(mosaic_jpeg), {"type": "text", "text": THROW_PROMPT.format(
        n=n, classes=", ".join(classes), features=_hints("How to recognise the symbols of these dice:", features))}]
    data, usage = _request(model, content, max_tokens=6000, tool=throw_tool(classes), max_tokens_free=12000)
    data["_usage"] = usage
    data["_model"] = model
    return data
