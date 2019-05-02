from .generic import Reader


class EPSCLabels(Reader):
    def __init__(self, database, mode, **kwargs):
        super().__init__(database=database, mode=mode, **kwargs)
        files = self.database.objects(
            groups=self.pad_groups[mode], protocol=self.database.protocol
        )
        client_id_maps = sorted(set(str(f.client_id) for f in files))
        self.client_id_maps = dict(zip(client_id_maps, range(len(client_id_maps))))

    def call(self, inputs):
        f = inputs["db_smaple"]
        labels = {
            "bio": self.client_id_maps[str(f.client_id)],
            "pad": f.attack_type is None,
        }
        inputs["labels"] = labels
        return inputs


class ByteKeys(Reader):
    def call(self, inputs):
        f = inputs["db_smaple"]
        inputs["features"]["key"] = str(f.make_path("", "")).encode()
        return inputs
