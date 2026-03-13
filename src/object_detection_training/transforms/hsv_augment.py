"""HSV color space augmentation matching YOLOX/RTMDet.

Operates in HSV color space with integer gains, matching the official
YOLOX ``augment_hsv`` and RTMDet ``YOLOXHSVRandomAug`` implementations.
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from torchvision.transforms import v2


class HSVRandomAug(v2.Transform):
    """HSV color space augmentation matching YOLOX/RTMDet.

    For each channel (H, S, V), a random gain in ``[-1, 1] * magnitude``
    is added.  The augmentation converts to HSV, applies gains, then
    converts back to RGB.

    This matches the official implementations:
    - YOLOX: ``augment_hsv(img, hgain=5, sgain=30, vgain=30)``
    - RTMDet: ``YOLOXHSVRandomAug(hue_delta=5, saturation_delta=30,
      value_delta=30)``

    Unlike ``torchvision.transforms.ColorJitter`` which operates in RGB
    space with fractional multipliers, this works in HSV space with integer
    additive deltas — matching the YOLO family convention.

    Must be placed in the pipeline BEFORE ``ToFloat32Tensor`` so that
    images are still PIL.
    """

    def __init__(
        self,
        hue_delta: int = 5,
        saturation_delta: int = 30,
        value_delta: int = 30,
    ) -> None:
        """Initialize HSVRandomAug.

        Args:
            hue_delta: Maximum hue shift (wraps around 0-255 in PIL HSV).
                Official YOLOX/RTMDet: 5.
            saturation_delta: Maximum saturation shift.
                Official YOLOX/RTMDet: 30.
            value_delta: Maximum value (brightness) shift.
                Official YOLOX/RTMDet: 30.
        """
        super().__init__()
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta

    def forward(self, *inputs: object) -> object:
        """Apply HSV augmentation to the image in the sample.

        Overrides ``v2.Transform.forward`` directly because this operates
        on PIL Images (not tensors), and only modifies the image — boxes
        and labels pass through unchanged.
        """
        # v2.Compose passes (image, target) tuples
        sample = inputs if len(inputs) > 1 else inputs[0]
        if not isinstance(sample, tuple) or len(sample) != 2:
            return sample

        img, target = sample
        if isinstance(img, Image.Image):
            img = self._augment_pil(img)
        return img, target

    def _augment_pil(self, img: Image.Image) -> Image.Image:
        """Apply HSV augmentation to a PIL Image.

        Uses numpy for fast integer HSV manipulation, matching the official
        YOLOX/RTMDet implementation.
        """
        # Random gains for each channel
        r = np.random.uniform(-1, 1, 3)
        h_gain = int(r[0] * self.hue_delta)
        s_gain = int(r[1] * self.saturation_delta)
        v_gain = int(r[2] * self.value_delta)

        if h_gain == 0 and s_gain == 0 and v_gain == 0:
            return img

        # Convert to HSV via PIL (H: 0-255, S: 0-255, V: 0-255)
        hsv = img.convert("HSV")
        hsv_arr = np.array(hsv, dtype=np.int16)

        # Apply gains
        hsv_arr[:, :, 0] = (hsv_arr[:, :, 0] + h_gain) % 256  # Hue wraps
        hsv_arr[:, :, 1] = np.clip(hsv_arr[:, :, 1] + s_gain, 0, 255)
        hsv_arr[:, :, 2] = np.clip(hsv_arr[:, :, 2] + v_gain, 0, 255)

        # Convert back to RGB
        hsv_result = Image.fromarray(hsv_arr.astype(np.uint8), mode="HSV")
        return hsv_result.convert("RGB")
