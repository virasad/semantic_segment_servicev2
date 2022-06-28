import requests
import json
import os


class BaseCallback:
    def __init__(self):
        pass

    def write(self, metrics):
        pass


class FileCallback(BaseCallback):
    def __init__(self, log_dir):
        super().__init__()
        self.file_path = log_dir

    def write(self, metrics):
        with open(os.path.join(self.file_path, 'result.json'), 'w') as f:
            json.dump(metrics, f)


class PostCallback(BaseCallback):
    def __init__(self, url):
        super().__init__()
        self.url = url

    def write(self, metrics):
        requests.post(self.url, json=metrics)