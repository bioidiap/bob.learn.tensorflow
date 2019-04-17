from .Base import Base


class Generic(Base):
    def __init__(self, architecture, **kwargs):

        self.architecture = architecture
        super().__init__(**kwargs)

    def get_output(self, data, mode):
        self.end_points = self.architecture(data, mode=mode)[1]
        return self.end_points[self.output_name]
