import requests
import time


def train_request(images_dir: int,
                  masks_dir: int,
                  n_classes: int,
                  batch_size: int,
                  epochs: int,
                  model_name: str,
                  w_size: int,
                  h_size: int,
                  response_url: str = None,
                  log_url: str = None,
                  task_id: str = None,
                  data_type: str = 'coco',
                  redis_cb: bool = True,
                  file_cb: bool = False,
                  ):
    url = 'http://localhost:5554/train'
    payload = {
        'images_dir': images_dir,
        'masks_dir': masks_dir,
        'n_classes': n_classes,
        'batch_size': batch_size,
        'epochs': epochs,
        'model_name': model_name,
        'w_size': w_size,
        'h_size': h_size,
        'response_url': response_url,
        'log_url': log_url,
        'task_id': task_id,
        'data_type': data_type,
        'redis_cb': redis_cb,
        'file_cb': file_cb,
    }

    response = requests.post(url, params=payload)
    print(response.json())
    return response.json()


def get_status(task_id: int):
    url = 'http://localhost:5554/get-status'
    payload = {
        'task_id': task_id,
    }

    response = requests.get(url, params=payload)
    print(response.json())
    return response.json()


def main():
    train_request('/dataset/images',
                  '/dataset/instances_default.json',
                  3,
                  2,
                  10,
                  'sample_model',
                  128,
                  128,
                  task_id=1,
                  data_type='coco',
                  redis_cb=True,
                  file_cb=False)


if __name__ == '__main__':
    # main()
    # time.sleep(60)
    get_status(1)