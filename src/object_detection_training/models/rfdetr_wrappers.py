from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

from object_detection_training.utils.hydra import register


@register
class RFDETRLarge(RFDETRLarge):
    pass


@register
class RFDETRMedium(RFDETRMedium):
    pass


@register
class RFDETRSmall(RFDETRSmall):
    pass


@register
class RFDETRNano(RFDETRNano):
    pass
