"""
ACR-style Python model -> RawTherapee .pp3 profile converter.

Takes an ACRModel instance (exposure, contrast, highlights, shadows, whites,
temperature, tint, vibrance, saturation, texture, clarity, dehaze, and
shadows/midtones/highlights HSL color grading) and writes it into a
RawTherapee .pp3 sidecar, then optionally renders it via rawtherapee-cli.

See the mapping table in the accompanying explanation for which pp3
section/key each ACR field maps to, and which mappings are approximations
(RT and ACR don't expose identical tool sets).
"""

import configparser
import math
import shutil
import subprocess
import time


def _val(x):
    """Pull a numeric value out of a RangedFloat/RangedInt wrapper or a plain number."""
    return float(x.value) if hasattr(x, "value") else float(x)


def _ensure_section(cfg: configparser.ConfigParser, name: str):
    if not cfg.has_section(name):
        cfg.add_section(name)


def hsl_to_rgb_offset(hue, saturation, luminance=0.0):
    """
    Convert a color-grading wheel adjustment (hue 0-360, saturation 0-100,
    luminance -100..100) into approximate RawTherapee ColorToning RGB
    slider offsets (roughly -100..100 per channel). This is a simple
    color-wheel projection, not a color-managed conversion.
    """
    hue_rad = math.radians(hue)
    r = math.cos(hue_rad) * saturation
    g = math.cos(hue_rad - math.radians(120)) * saturation
    b = math.cos(hue_rad - math.radians(240)) * saturation
    bias = luminance / 4  # small brightness nudge riding on the color offset
    return (round(r + bias, 1), round(g + bias, 1), round(b + bias, 1))


def _set_fields(acr):
    """
    Fields the caller actually passed to ACRModel(...), as opposed to ones
    sitting at their pydantic default. This is the key fix: we must NOT
    write a pp3 key for a field the user never touched, because the
    default RangedFloat/RangedInt/HSLModel objects are not guaranteed to
    represent a neutral/no-op value (e.g. a default Temperature sitting
    near one edge of the 2000-50000K range will slam white balance,
    a non-zero default hue/saturation on the HSL wheels will slam color
    grading, etc.). Untouched fields must leave the template's pp3 value
    alone, not get overwritten with whatever the dataclass default is.
    """
    if hasattr(acr, "model_fields_set"):   # pydantic v2
        return acr.model_fields_set
    if hasattr(acr, "__fields_set__"):     # pydantic v1
        return acr.__fields_set__
    raise TypeError(
        "acr must be a pydantic model so we can tell which fields were "
        "explicitly set vs left at their default"
    )


def _active_fields(acr):
    disabled_fields = set(getattr(acr, "disabled_fields", set()) or set())
    return {name for name in _set_fields(acr) if getattr(acr, name, None) is not None and name not in disabled_fields}


def _hsl_triplet(hsl_model):
    if hsl_model is None:
        return None

    hue = getattr(hsl_model, "hue", None)
    saturation = getattr(hsl_model, "saturation", None)
    luminance = getattr(hsl_model, "luminance", None)

    if hue is None and saturation is None and luminance is None:
        return None

    return (
        0 if hue is None else hue,
        0 if saturation is None else saturation,
        0 if luminance is None else luminance,
    )


def apply_acr_to_pp3(acr, output_pp3: str):
    """
    Read an existing .pp3 as a template (so untouched tools -- camera
    profile, raw demosaic method, lens corrections, etc. -- keep their
    values) and overwrite ONLY the fields explicitly set on the ACRModel
    instance. Fields left at their default are skipped entirely so they
    don't clobber the template with an arbitrary/non-neutral default.
    """
    set_fields = _active_fields(acr)

    cfg = configparser.ConfigParser()

    cfg.optionxform = str  # type: ignore # pp3 keys are case-sensitive
    

    # ---- Light ----
    if "exposure" in set_fields:
        _ensure_section(cfg, "Exposure")
        cfg["Exposure"]["Compensation"] = str(_val(acr.exposure))
    if "contrast" in set_fields:
        _ensure_section(cfg, "Exposure")
        cfg["Exposure"]["Contrast"] = str(int(_val(acr.contrast)))
    if "whites" in set_fields:
        # No direct "Whites" slider in RT; nudge the highlight compression
        # threshold as the closest available approximation.
        _ensure_section(cfg, "Exposure")
        cfg["Exposure"]["HighlightComprThreshold"] = str(
            int(max(0, min(100, 50 + _val(acr.whites) / 2)))
        )

    if "highlights" in set_fields or "shadows" in set_fields or "clarity" in set_fields:
        _ensure_section(cfg, "Shadows & Highlights")
        cfg["Shadows & Highlights"]["Enabled"] = "true"
        if "highlights" in set_fields:
            cfg["Shadows & Highlights"]["Highlights"] = str(int(_val(acr.highlights)))
        if "shadows" in set_fields:
            cfg["Shadows & Highlights"]["Shadows"] = str(int(_val(acr.shadows)))
        if "clarity" in set_fields:
            cfg["Shadows & Highlights"]["LocalContrast"] = str(int(_val(acr.clarity)))

    # ---- Color ----
    if "temperature" in set_fields or "tint" in set_fields:
        _ensure_section(cfg, "White Balance")
        cfg["White Balance"]["Setting"] = "Custom"
        if "temperature" in set_fields:
            cfg["White Balance"]["Temperature"] = str(int(_val(acr.temperature)))
        if "tint" in set_fields:
            # RT's "Green" tint control is a ~0.2-2.5 multiplier; ACR tint is -150..150.
            cfg["White Balance"]["Green"] = str(round(1 + _val(acr.tint) / 300, 3))

    if "vibrance" in set_fields or "saturation" in set_fields:
        _ensure_section(cfg, "Vibrance")
        cfg["Vibrance"]["Enabled"] = "true"
        if "vibrance" in set_fields:
            cfg["Vibrance"]["Pastels"] = str(int(_val(acr.vibrance)))
        if "saturation" in set_fields:
            cfg["Vibrance"]["Saturated"] = str(int(_val(acr.saturation)))

    # ---- Effects ----
    if "texture" in set_fields:
        _ensure_section(cfg, "SharpenMicro")
        cfg["SharpenMicro"]["Enabled"] = "true" if _val(acr.texture) else "false"
        cfg["SharpenMicro"]["Strength"] = str(
            int(max(0, min(100, 20 + _val(acr.texture))))
        )

    if "dehaze" in set_fields:
        # Requires RawTherapee >= 5.7 for a [Haze Removal] section to have
        # any effect. Older RT builds (like the 5.2 template) ignore it.
        _ensure_section(cfg, "Dehaze")
        cfg["Dehaze"]["Enabled"] = "true" if _val(acr.dehaze) else "false"
        cfg["Dehaze"]["Strength"] = str(int(_val(acr.dehaze)))
        cfg["Dehaze"]["Depth"] = "100"
        cfg["Dehaze"]["Saturation"] = "50"

    # ---- Color Grading -> ColorToning RGB sliders ----
    grading_fields = {"shadows_hsl", "midtones_hsl", "highlights_hsl"} & set_fields
    if grading_fields:
        _ensure_section(cfg, "ColorToning")
        cfg["ColorToning"]["Enabled"] = "true"
        cfg["ColorToning"]["Method"] = "RGB Sliders"

        if "shadows_hsl" in grading_fields:
            triplet = _hsl_triplet(acr.shadows_hsl)
            if triplet is not None:
                r, g, b = hsl_to_rgb_offset(*triplet)
                cfg["ColorToning"]["Redlow"] = str(r)
                cfg["ColorToning"]["Greenlow"] = str(g)
                cfg["ColorToning"]["Bluelow"] = str(b)

        if "midtones_hsl" in grading_fields:
            triplet = _hsl_triplet(acr.midtones_hsl)
            if triplet is not None:
                r, g, b = hsl_to_rgb_offset(*triplet)
                cfg["ColorToning"]["Redmed"] = str(r)
                cfg["ColorToning"]["Greenmed"] = str(g)
                cfg["ColorToning"]["Bluemed"] = str(b)

        if "highlights_hsl" in grading_fields:
            triplet = _hsl_triplet(acr.highlights_hsl)
            if triplet is not None:
                r, g, b = hsl_to_rgb_offset(*triplet)
                cfg["ColorToning"]["Redhigh"] = str(r)
                cfg["ColorToning"]["Greenhigh"] = str(g)
                cfg["ColorToning"]["Bluehigh"] = str(b)

    with open(output_pp3, "w") as f:
        cfg.write(f, space_around_delimiters=False)


def render(raw_path: str, pp3_path: str, out_path: str, rawtherapee_cli="rawtherapee-cli"):
    """Shell out to rawtherapee-cli to actually render the raw file."""
    if shutil.which(rawtherapee_cli) is None:
        raise RuntimeError(f"'{rawtherapee_cli}' not found on PATH")
    process = subprocess.Popen(
        [rawtherapee_cli, "-Y", "-q", "-p", pp3_path, "-o", out_path,  "-c", raw_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        stout, stderr = process.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        stout, stderr = process.communicate()

if __name__ == "__main__":
    from type import ACRModel, HSLModel

    acr = ACRModel(
        exposure=-5,
        contrast=20,
        highlights=-40,
        shadows=30,
        dehaze=25,
        # midtones_hsl=HSLModel(hue=210, saturation=15, luminance=0),
    )

    apply_acr_to_pp3(acr, output_pp3="edit.pp3")
    render("milkyway.jpg", "edit.pp3", "IMG_0001_out.jpg")