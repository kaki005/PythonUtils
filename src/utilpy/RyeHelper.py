import subprocess

import yaml  # type: ignore


class RyeHelper:
    def AddFromYml(self, yml_path):
        with open(yml_path) as yml:
            config = yaml.safe_load(yml)
            channel = config["channels"]
            depend = config["dependencies"]
            for item in channel:
                subprocess.call(f"rye add {item}", shell=True)
            for item in depend:
                subprocess.call(f"rye add {item}", shell=True)
