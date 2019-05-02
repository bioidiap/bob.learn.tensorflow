class ComposeTransofrm:
    """Composes several transforms together"""

    def __init__(self, transforms, **kwargs):
        super().__init__(**kwargs)
        self.transforms = transforms

    def __call__(self, features, labels=None):
        for transform in self.transforms:
            features, labels = transform(features, labels)
        return features, labels
