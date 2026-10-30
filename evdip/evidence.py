"""Pluggable evidence interface: every sensor emits the same record."""
from dataclasses import dataclass, field, asdict


@dataclass
class Evidence:
    source: str             # visible | thermal | acoustic | humidity | flow
    wall: str
    x_cm: float
    y_cm: float
    confidence: float       # 0..1
    timestamp: float        # hours (or epoch hours)
    details: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def fuse_evidence(items, radius_cm=25.0):
    """Merge evidence from different sources that point to the same place (noisy-OR on confidence)."""
    groups = []
    for e in sorted(items, key=lambda e: -e.confidence):
        for g in groups:
            if g[0].wall == e.wall and ((g[0].x_cm - e.x_cm) ** 2 + (g[0].y_cm - e.y_cm) ** 2) ** 0.5 < radius_cm:
                g.append(e); break
        else:
            groups.append([e])
    out = []
    for g in groups:
        w = sum(e.confidence for e in g) + 1e-9
        out.append({"wall": g[0].wall,
                    "x_cm": sum(e.x_cm * e.confidence for e in g) / w,
                    "y_cm": sum(e.y_cm * e.confidence for e in g) / w,
                    "confidence": 1 - float(__import__("numpy").prod([1 - e.confidence for e in g])),
                    "sources": sorted({e.source for e in g})})
    return sorted(out, key=lambda d: -d["confidence"])
