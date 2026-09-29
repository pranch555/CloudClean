"""Revopoint turntable Bluetooth LE protocol: command encoders, reply parser, name classifier.

Pure Python (no Bluetooth, no numpy) so it can be unit tested anywhere. Everything here was recovered from
Revo Metro V5.8.7 (`RevoMetro.exe`, classes `DualAxisTurntable` / `HumanTuentable` / `PartManagement`) and checked
against Revo Metro's own logs of the user's turntable; see `docs/turntable.md` for the evidence behind each line.
Labels: [verified] read from code or seen in real traffic, [inferred] strong reading, [unknown] open.

Transport [verified]: one GATT characteristic whose UUID starts with ``0000ffe1`` (Revo Metro selects it by parsing
the first UUID group as hex and comparing with 0xFFE1). Commands are ASCII strings written *without response*
(SimpleBLE ``write_command``), split into MTU-sized chunks; replies arrive as notifications on the same
characteristic, one message per notification.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from dataclasses import field as _dc_field

# --------------------------------------------------------------------------- transport constants
DATA_CHAR_SHORT = 0xFFE1                                       # [verified] dual-axis + large turntable
DATA_CHAR_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"        # [inferred] the usual 128-bit form of 0xFFE1
DATA_SERVICE_UUID = "0000ffe0-0000-1000-8000-00805f9b34fb"     # [inferred] Revo Metro does not check the service
STABILIZER_CHAR_SHORT = 0xFFB1                                 # [verified] handheld stabilizer (not supported)

# Advertised names [verified: RevoMetro.exe 0x14080b2e0, exact comparisons]
NAME_DUAL_AXIS = "REVO_DUAL_AXIS_TABLE"
NAME_LARGE = "REVO_TA500"
NAME_STABILIZER = ("REVO_STABILIZER", "RevoStabilizer")

# Revo Metro's firmware switch: below "3.28" a turn is only acknowledged with "+OK;"; from 3.28 the table also
# sends "+OK,TURNANGLE=<position>" when the move has finished [verified: 0x14080e9e0 and the user's logs, fw 3.28].
COMPLETION_FIRMWARE = (3, 28)

# Continuous rotation sentinel Revo Metro passes as the "angle" [verified: +-2^28 compared in 0x1408122c0]
CONTINUOUS = 268435456.0


def classify_name(name: str | None) -> str | None:
    """Device kind from the advertised name exactly as Revo Metro does it, or None for anything else."""
    if not name:
        return None
    if name == NAME_DUAL_AXIS:
        return "dual_axis"
    if name == NAME_LARGE:
        return "large"
    if name == NAME_STABILIZER[0] or NAME_STABILIZER[1] in name:
        return "stabilizer"
    return None


def is_data_characteristic(uuid: str, short: int = DATA_CHAR_SHORT) -> bool:
    """Revo Metro's test: first UUID group parsed as hex equals 0xFFE1 (works for 16- and 128-bit forms)."""
    head = str(uuid).strip().lower().split("-")[0]
    try:
        return int(head, 16) == short
    except ValueError:
        return False


def whole_degrees(value: float, what: str = "angle") -> int:
    """The protocol carries integers (Revo Metro truncates with cvttss2si). We round to nearest and let the
    caller report requested vs commanded. Raises ValueError for non-finite input."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"The {what} must be a finite number")
    return int(round(value))


# --------------------------------------------------------------------------- dual-axis turntable commands
# "CT" = command turn (rotation axis, "X"), "CR" = command tilt ("Y"); "QT"/"QR" the matching queries.
def dual_turn(degrees: int) -> str:
    """Relative rotation by whole degrees; positive = the direction Revo Metro labels 'Clockwise' [verified
    mapping in the UI code; physical sense seen from above: inferred]."""
    return f"+CT,TURNANGLE={int(degrees)};"


def dual_turn_continuous(positive: bool) -> str:
    """Rotate until stopped ("Turntable Sync" uses the negative one) [verified]."""
    return "+CT,TURNCONTINUE=1;" if positive else "+CT,TURNCONTINUE=-1;"


def dual_tilt(degrees: int) -> str:
    """Absolute tilt in whole degrees (Revo Metro: -30..30) [verified]."""
    return f"+CR,TILTVALUE={int(degrees)};"


def dual_stop(axis: str = "turn") -> str:
    return "+CT,STOP;" if axis == "turn" else "+CR,STOP;"


def dual_to_zero(axis: str = "turn") -> str:
    return "+CT,TOZERO;" if axis == "turn" else "+CR,TOZERO;"


def dual_speed(axis: str, value: int) -> str:
    """Turn speed in seconds per revolution [verified by timing: 10 deg at 53-54 s/rev took 1.49-1.53 s].
    Tilt speed: firmware 3.28 answers Revo Metro's '+CR,TILTSPEED=10;' with '+FAIL,ERR=007;' [verified]."""
    return f"+CT,TURNSPEED={int(value)};" if axis == "turn" else f"+CR,TILTSPEED={int(value)};"


QUERY_VERSION = "+QR,VERSION;"          # reply "+DATA=<x>,V<major.minor>..." [inferred from the parser]
QUERY_TURN_POSITION = "+QT,CHANGEANGLE;"  # reply "+DATA=<deg>" [inferred]
QUERY_TILT_POSITION = "+QR,TILTVALUE;"    # reply "+DATA=<deg>" [inferred]
QUERY_TURN_RANGE = "+QT,TURNANGLE;"       # reply "+DATA=a,b,c" (three floats, meaning unknown)
QUERY_TILT_RANGE = "+QR,TILTANGLE;"       # reply "+DATA=a,b,c"
QUERY_TURN_SPEED_RANGE = "+QT,TURNSPEED;"  # reply "+DATA=a,b,c"; Revo Metro sets speed = a + (b - a) / 2
QUERY_TILT_SPEED_RANGE = "+QR,TILTSPEED;"  # reply "+DATA=a,b,c"


# --------------------------------------------------------------------------- large turntable (REVO_TA500)
def large_turn(degrees: int) -> str:
    """Relative rotation: CT+TRUNSINGLE(<dir>,<abs deg>); dir 1 for degrees <= 0 [verified 0x140818d50]."""
    degrees = int(degrees)
    return f"CT+TRUNSINGLE({1 if degrees <= 0 else 0},{abs(degrees)});"


def large_turn_continuous(positive: bool) -> str:
    return "CT+START(0,0,0,1,0,0);" if positive else "CT+START(1,0,0,1,0,0);"


def large_speed_level(s_per_rev: float) -> int:
    """Revo Metro's conversion of the UI speed (s/rev) to the table's level 1..20 [verified 0x140818c20]."""
    s = int(s_per_rev)
    if s > 92:
        return 20
    if s < 35:
        return 1
    return int((s - 32) / 3.0)


def large_speed(s_per_rev: float) -> str:
    return f"CT+SETSPEED({large_speed_level(s_per_rev)});"


def large_direction(clockwise: bool) -> str:
    """CT+SETDIR(<not flag>); the flag's meaning is not traced to the UI [inferred]."""
    return f"CT+SETDIR({0 if clockwise else 1});"


LARGE_STOP = "CT+SETSTOP();"
LARGE_TO_ZERO = "CT+TOZERO();"
LARGE_MUTE = "CT+SPKMUTE(1);"   # Revo Metro sends this right after connecting; CloudClean does not


# --------------------------------------------------------------------------- replies
@dataclass
class Reply:
    """One parsed notification. kind: ok | ok_field | fail | data | unknown."""
    raw: str
    kind: str
    field: str | None = None        # "TURNANGLE", "TOZERO", ... for ok_field
    value: str | None = None        # text after "=" for ok_field / data
    code: str | None = None         # "007" for "+FAIL,ERR=007;"
    numbers: list[float] = _dc_field(default_factory=list)

    @property
    def number(self) -> float | None:
        return self.numbers[0] if self.numbers else None

    def to_dict(self) -> dict:
        return {"raw": self.raw, "kind": self.kind, "field": self.field, "value": self.value, "code": self.code,
                "numbers": self.numbers}


_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _numbers(text: str | None) -> list[float]:
    out = []
    for part in (text or "").split(","):
        m = _NUM.search(part)
        if m:
            out.append(float(m.group()))
    return out


def parse_reply(text: str) -> Reply:
    """Parse '+OK;', '+OK,TURNANGLE=-145.72', '+OK,TOZERO;', '+FAIL,ERR=007;', '+DATA=a,b,c', 'CR+OK;'.
    Tolerant of a missing ';' (seen in real replies) and surrounding whitespace / NUL bytes."""
    raw = text
    body = text.strip().strip("\x00").strip()
    if body.endswith(";"):
        body = body[:-1].rstrip()
    upper = body.upper()
    if upper in ("+OK", "CR+OK"):
        return Reply(raw, "ok")
    if upper.startswith("+OK,"):
        rest = body[4:]
        name, _, value = rest.partition("=")
        return Reply(raw, "ok_field", field=name.strip().upper(), value=value.strip() if _ else None,
                     numbers=_numbers(value) if _ else [])
    if upper.startswith("+FAIL") or upper.startswith("CR+FAIL"):
        m = re.search(r"ERR\s*=\s*([0-9A-Za-z]+)", body)
        return Reply(raw, "fail", code=m.group(1) if m else None)
    if upper.startswith("+DATA"):
        _, _, value = body.partition("=")
        return Reply(raw, "data", value=value.strip(), numbers=_numbers(value))
    return Reply(raw, "unknown")


def parse_version(reply: Reply | str | None) -> str | None:
    """Revo Metro splits '+DATA=' on ',' and takes 4 characters after 'V' of the second field. We accept any
    'V<digits>.<digits>' (or a bare number) in the payload."""
    text = reply.value if isinstance(reply, Reply) else reply
    if not text:
        return None
    m = re.search(r"[Vv](\d+\.\d+)", text) or re.search(r"(\d+\.\d+)", text)
    return m.group(1) if m else None


def version_tuple(version: str | None) -> tuple[int, ...] | None:
    if not version:
        return None
    try:
        return tuple(int(p) for p in version.split("."))
    except ValueError:
        return None


def reports_completion(version: str | None) -> bool | None:
    """True when the firmware announces the end of a turn ('+OK,TURNANGLE=...'), None when unknown."""
    v = version_tuple(version)
    return None if v is None else v >= COMPLETION_FIRMWARE


class MessageAssembler:
    """Turns raw notification payloads into messages.

    Real replies are one message per notification, but a message may be split when the negotiated MTU is small
    and one notification could in principle carry two. Rules: split on ';' (kept); a chunk that does not start a
    new message ('+' or 'CR+') continues the pending one; an unterminated message is released by `flush()`
    (the BLE driver calls it ~80 ms after the last notification) or when the next message starts."""

    def __init__(self):
        self.pending = ""

    @staticmethod
    def _starts_message(text: str) -> bool:
        return text.startswith("+") or text.upper().startswith("CR+") or text.upper().startswith("CT+")

    def feed(self, data: bytes | str) -> list[str]:
        text = data.decode("ascii", errors="replace") if isinstance(data, (bytes, bytearray)) else str(data)
        text = text.replace("\x00", "").replace("\r", "").replace("\n", "")
        out: list[str] = []
        if not text:
            return out
        if self.pending and self._starts_message(text):
            out.append(self.pending)
            self.pending = ""
        buf = self.pending + text
        self.pending = ""
        while ";" in buf:
            msg, _, buf = buf.partition(";")
            if msg.strip():
                out.append(msg + ";")
        self.pending = buf.strip()
        return out

    def flush(self) -> list[str]:
        out = [self.pending] if self.pending else []
        self.pending = ""
        return out
