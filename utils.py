# ============================================================
# utils.py — Small general-purpose helpers
# ============================================================


def clamp(value, min_value, max_value):
    """Clamp *value* between *min_value* and *max_value*."""
    return max(min_value, min(max_value, value))


def lerp(a, b, t):
    """Linear interpolation: returns a when t=0, b when t=1."""
    return a + (b - a) * t


def map_range(value, in_min, in_max, out_min, out_max):
    """Map *value* from [in_min, in_max] to [out_min, out_max]."""
    if abs(in_max - in_min) < 1e-9:
        return (out_min + out_max) / 2.0
    t = (value - in_min) / (in_max - in_min)
    return out_min + t * (out_max - out_min)


def timestamp_ms():
    """Return current time in milliseconds (float)."""
    import time
    return time.time() * 1000.0