import pydantic


class HSLModel(pydantic.BaseModel):
    hue: int | None = pydantic.Field(default=None, ge=0, le=360)
    saturation: int | None = pydantic.Field(default=None, ge=0, le=100)
    luminance: int | None = pydantic.Field(default=None, ge=0, le=100)


class ACRModel(pydantic.BaseModel):

    disabled_fields: set[str] = pydantic.Field(default_factory=set, exclude=True)

    # Light
    exposure: float | None = pydantic.Field(default=None, ge=-5.0, le=5.0)
    contrast: int | None = pydantic.Field(default=None, ge=-100, le=100)
    highlights: int | None = pydantic.Field(default=None, ge=-100, le=100)
    shadows: int | None = pydantic.Field(default=None, ge=-100, le=100)
    whites: int | None = pydantic.Field(default=None, ge=-100, le=100)

    # Color
    temperature: int | None = pydantic.Field(default=None, ge=2000, le=50000)
    tint: int | None = pydantic.Field(default=None, ge=-150, le=150)
    vibrance: int | None = pydantic.Field(default=None, ge=-100, le=100)
    saturation: int | None = pydantic.Field(default=None, ge=-100, le=100)

    # Effects
    texture: int | None = pydantic.Field(default=None, ge=-100, le=100)
    clarity: int | None = pydantic.Field(default=None, ge=-100, le=100)
    dehaze: int | None = pydantic.Field(default=None, ge=0, le=100)

    # Color Grading
    shadows_hsl: HSLModel | None = pydantic.Field(default=None)
    midtones_hsl: HSLModel | None = pydantic.Field(default=None)
    highlights_hsl: HSLModel | None = pydantic.Field(default=None)

    def model_post_init(self, __context):
        for field_name in self.disabled_fields:
            if hasattr(self, field_name):
                setattr(self, field_name, None)


def acr_from_array(arr, disabled_field_names=None):
    """Convert a numpy array of ACR parameters to an ACRModel instance."""
    disabled_fields: set[str] = set(disabled_field_names or ())
    acr_values = {
        "exposure": arr[0],
        "contrast": arr[1],
        "highlights": arr[2],
        "shadows": arr[3],
        "whites": arr[4],
        "temperature": arr[5],
        "tint": arr[6],
        "vibrance": arr[7],
        "saturation": arr[8],
        "texture": arr[9],
        "clarity": arr[10],
        "dehaze": arr[11],
        "shadows_hsl": {"hue": arr[12], "saturation": arr[13], "luminance": arr[14]},
        "midtones_hsl": {"hue": arr[15], "saturation": arr[16], "luminance": arr[17]},
        "highlights_hsl": {"hue": arr[18], "saturation": arr[19], "luminance": arr[20]},
    }
    acr_dict = {name: value for name, value in acr_values.items() if name not in disabled_fields}
    return ACRModel(disabled_fields=disabled_fields, **acr_dict)