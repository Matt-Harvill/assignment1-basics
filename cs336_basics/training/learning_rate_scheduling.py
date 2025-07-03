import math


def get_lr_cosine_schedule(t: int, alpha_max: float, alpha_min: float, Tw: int, Tc: int) -> float:
    if t < Tw:
        return t / Tw * alpha_max
    elif t <= Tc:
        return alpha_min + 0.5 * (1 + math.cos(math.pi * (t - Tw) / (Tc - Tw))) * (alpha_max - alpha_min)
    else:
        return alpha_min
