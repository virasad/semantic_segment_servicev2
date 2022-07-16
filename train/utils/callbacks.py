import json
import os

import redis
import requests


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


class RedisCallback(BaseCallback):
    def __init__(self, task_id, host='localhost', port=6379, db=1):
        super().__init__()
        self.rd = redis.Redis(host=host, port=port, db=db)
        self.task_id = task_id

    def write(self, metrics):
        self.rd.set(self.task_id, json.dumps(metrics))

    def get(self):
        try:
            return json.loads(self.rd.get(self.task_id))
        except:
            return {'status': 'error', 'message': 'No result found'}
