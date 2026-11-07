"""Die net (step 4): which symbol lies on which face, and how it is rotated.

Net (cross shape), as you look at the unfolded die from outside:

            [back  ]
    [left ] [ top  ] [right]
            [front ]
            [bottom]

"rotation" = by how many degrees (clockwise) the symbol is rotated in the NET:
0 = the symbol stands upright when you look at the net normally.
For the 3D preview (CSS) a fixed offset per face is added (derived geometrically):
top/front/bottom 0°, right +90°, left −90°, back 180°.
"""

FIELDS = ["back", "left", "top", "right", "front", "bottom"]
OPPOSITE = {"top": "bottom", "bottom": "top", "front": "back", "back": "front", "left": "right", "right": "left"}
CSS_OFFSET = {"top": 0, "front": 0, "bottom": 0, "right": 90, "left": -90, "back": 180}
NEIGHBOURS = {f: [g for g in FIELDS if g != f and g != OPPOSITE[f]] for f in FIELDS}


def empty():
    return {"net": {f: {"symbol": None, "rotation": 0} for f in FIELDS}, "cross_check": {}, "confirmed": False, "source": None}


def normalise(layout):
    a = empty()
    if not layout:
        return a
    for f in FIELDS:
        e = (layout.get("net") or {}).get(f) or {}
        a["net"][f]["symbol"] = e.get("symbol") or None
        try:
            a["net"][f]["rotation"] = int(e.get("rotation", 0)) % 360 // 90 * 90
        except (TypeError, ValueError):
            a["net"][f]["rotation"] = 0
    a["cross_check"] = dict(layout.get("cross_check") or {})
    a["confirmed"] = bool(layout.get("confirmed"))
    a["source"] = layout.get("source")
    for k in ("confirmed_at", "ai_remark"):
        if layout.get(k):
            a[k] = layout[k]
    return a


def opposite_from_net(layout):
    """{symbol: opposite symbol} according to the net."""
    net = layout["net"]
    result = {}
    for f in FIELDS:
        s, t = net[f]["symbol"], net[OPPOSITE[f]]["symbol"]
        if s and t:
            result[s] = t
    return result


def check(layout, classes):
    """Returns a list of problems (empty = everything consistent)."""
    problems = []
    symbols = [layout["net"][f]["symbol"] for f in FIELDS]
    missing = [k for k in classes if k not in symbols]
    double = sorted({s for s in symbols if s and symbols.count(s) > 1})
    if any(s is None for s in symbols):
        problems.append("Not all 6 faces have a symbol.")
    if double:
        problems.append("Used twice: " + ", ".join(double))
    if missing and not any(s is None for s in symbols):
        problems.append("Missing in the net: " + ", ".join(missing))
    net_opp = opposite_from_net(layout)
    cc = layout.get("cross_check") or {}
    for s, t in cc.items():
        if not t:
            continue
        if cc.get(t) and cc.get(t) != s:
            problems.append(f"Cross-check contradicts itself: {s}↔{t}, but {t}↔{cc.get(t)}.")
        if s in net_opp and net_opp[s] != t:
            problems.append(f"Cross-check says “{s} opposite {t}”, but in the net {net_opp[s]} lies opposite {s}.")
    return problems


def cross_check_complete(layout, classes):
    cc = layout.get("cross_check") or {}
    return all(cc.get(k) for k in classes)
