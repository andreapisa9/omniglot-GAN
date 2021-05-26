import torchvision.transforms.functional as TF

class InvertTransform:
    """Invert the b/w scale."""

    def __init__(self):
        super().__init__()

    def __call__(self, image):
        return TF.invert(image)