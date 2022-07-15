import requests


def inference_request(image_path: str,
                      return_image,
                      return_coco,
                      ):
    url = 'http://localhost:8000/predict'
    params = {
        'return_image': return_image,
        'return_coco': return_coco,
    }
    files = {'image': open(image_path, 'rb')}
    response = requests.post(url, params=params, files=files)
    print(response.json())
    return response.json()


def set_model_request(model_path: str):
    url = 'http://localhost:8000/set-model'
    params = {
        'model_path': model_path,
    }
    response = requests.post(url, params=params)
    print(response.json())
    return response.json()


def upload_model_request(model_file: str):
    url = 'http://localhost:8000/upload-model'
    files = {'model_file': open(model_file, 'rb')}
    response = requests.post(url, files=files)
    print(response.json())
    return response.json()


def main():
    set_model_request('your/model/path/model_name.pt')
    inference_request('/images/test.jpg', False, True)
