import requests


class BaseCallBack():
    def __init__(self):
        pass

    def write(self, metrics):
        pass


class PostCallBack(BaseCallBack):
    def __init__(self, url):
        self.url = url

    def write(self, metrics):
        requests.post(self.url, json=metrics)


class FileCallBack(BaseCallBack):
    def __init__(self, file_path):
        self.file_path = file_path

    def write(self, metrics):
        with open(self.file_path, 'w') as f:
            f.write(str(metrics))
