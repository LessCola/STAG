import yaml


class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)

    def items(self):
        return [
            (key, value)
            for key, value in self.__dict__.items()
            if not key.startswith("__")
        ]

    def get(self, key, default=None):
        keys = key.split(".")
        current = self
        for k in keys:
            if hasattr(current, k):
                current = getattr(current, k)
            else:
                return default
        return current

    def update(self, key, value):
        keys = key.split(".")
        current = self
        for k in keys[:-1]:
            current = getattr(current, k)
        setattr(current, keys[-1], value)

    def delete(self, key):
        keys = key.split(".")
        current = self

        # Traverse to the second last key
        for k in keys[:-1]:
            if hasattr(current, k):
                current = getattr(current, k)
            else:
                # Key not found in the path
                return

        # Delete the final key, if it exists
        if hasattr(current, keys[-1]):
            delattr(current, keys[-1])

    def insert(self, key, value):
        keys = key.split(".")
        current = self

        # Traverse to the second last key, creating nested Config objects if necessary
        for k in keys[:-1]:
            if not hasattr(current, k):
                setattr(current, k, Config({}))
            current = getattr(current, k)

        # Insert the value at the final key
        setattr(current, keys[-1], value)

    def save(self, filename):
        def to_dict(obj):
            if isinstance(obj, Config):
                return {k: to_dict(v) for k, v in obj.items()}
            else:
                return obj

        with open(filename, "w") as f:
            yaml.dump(to_dict(self), f, default_flow_style=False)

    def copy(self):
        """
        Create a deep copy of the configuration.
        """

        def recursive_copy(obj):
            if isinstance(obj, Config):
                return {k: recursive_copy(v) for k, v in obj.items()}
            else:
                return obj

        copied_dict = recursive_copy(self)
        return Config(copied_dict)

    def pretty_print(self, indent=0):
        for key, value in self.items():
            if isinstance(value, Config):
                print("  " * indent + str(key) + ":")
                value.pretty_print(indent + 1)
            else:
                print("  " * indent + "{}: {}".format(key, value))


def load_config(file_path):
    with open(file_path, "r") as file:
        config_dict = yaml.safe_load(file)
    return Config(config_dict)
