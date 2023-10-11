
import requests 
import os
import time
import socket
from multiprocessing import Barrier
from multiprocessing.managers import SyncManager
import io
import base64

SYNC_MANAGER_BASE_PORT  =  10042
SYNC_MANAGER_AUTH_KEY   = b"aime_api_worker"

class MyManager(SyncManager):
    pass


class APIWorkerInterface():

    manager = None
    barrier = None

    @staticmethod
    def get_barrier():
        return APIWorkerInterface.barrier

    def __init__(self, api_server, job_type, auth_key, gpu_id=0, world_size=1, rank=0):
        self.api_server = api_server
        self.world_size = world_size
        self.worker_name = self.__make_worker_name(gpu_id)
        self.rank = rank
        self.job_type = job_type
        self.auth_key = auth_key
        MyManager.register("barrier", APIWorkerInterface.get_barrier)
        APIWorkerInterface.manager = MyManager(("127.0.0.1", SYNC_MANAGER_BASE_PORT + gpu_id), authkey=SYNC_MANAGER_AUTH_KEY)
        if world_size > 1:
            # multi GPU synchronization required
            if rank == 0:
                APIWorkerInterface.barrier = Barrier(world_size)
                APIWorkerInterface.manager.start()
            else:
                time.sleep(2)   # manager has to be started first to connect
                APIWorkerInterface.manager.connect()
                APIWorkerInterface.barrier = APIWorkerInterface.manager.barrier()

    def __make_worker_name(self, gpu_id):
        world_size = int(os.environ.get("WORLD_SIZE", -1))
        worker_name = socket.gethostname()
        for id in range(world_size):
            worker_name += f'_GPU{id+gpu_id}'
        return worker_name

    def job_request(self):
        job_data = None
        if self.rank == 0:
            have_job = False
            counter = 0
            while not have_job:
                request = { "auth": self.worker_name, "job_type": self.job_type, "auth_key": self.auth_key }
                try:
                    response = self.__fetch('/worker_job_request', request)
                except requests.exceptions.ConnectionError:
                    APIWorkerInterface.check_periodically_if_server_online(self.api_server)
                    continue
                
                if response.status_code == 200:
                    job_data = response.json()   
                    cmd = job_data.get('cmd', None)
                    if cmd == 'job':
                        have_job = True
                    elif cmd == 'error':
                        exit("! API server responded with error: " + str(job_data.get('msg', "unknown")))
                    elif cmd == 'warning':
                        print("! API server responded with warning: " + str(job_data.get('msg', "unknown")))
                        counter += 1
                        APIWorkerInterface.check_periodically_if_server_online(self.api_server)
                        if counter > 3:
                            exit("API server responded with warning: " + str(job_data.get('msg', "unknown")))
                    else:
                        exit("! API server responded with unknown cmd: " + str(cmd))

        if self.world_size > 1:
            # hold all GPU processes here until we have a new job                     
            APIWorkerInterface.barrier.wait()

        return job_data


    def send_job_results(self, job_data, results):
        for parameter in ['job_id', 'start_time', 'start_time_compute']:
            results[parameter] = job_data[parameter]

        output_descriptions = job_data['output_descriptions']
        for output_name, output_description in output_descriptions.items():
            if output_name in job_data:
                results[output_name] = job_data[output_name]
            if output_description.get('type') == 'image_list':
                if output_name in results:
                    results[output_name] = self.convert_image_to_base64_string(results[output_name], output_description.get('image_format', 'PNG'))

        response = self.__fetch('/worker_job_result', results)
        return response


    def send_progress(self, job_data, progress, progress_data=None):
        payload = {'progress': progress, 'job_id': job_data['job_id']}
        progress_desciptions = job_data['progress_desciptions']
        if progress_data:
            for progress_parameter_name, progress_desciption in progress_desciptions.items():
                if progress_desciption.get('type') == 'image_list':
                    if progress_parameter_name in progress_data:
                        payload[progress_parameter_name] = self.convert_image_to_base64_string(progress_data[progress_parameter_name], progress_desciptions.get('image_format', 'PNG'))

        response = self.__fetch('/worker_job_progress', payload)
        return response


    def convert_image_to_base64_string(self, list_images, image_format):
        image_64 = ''
        for image in list_images:
            with io.BytesIO() as buffer:
                image.save(buffer, format=image_format)
                image_64 += f'data:image/{image_format};base64,' + base64.b64encode(buffer.getvalue()).decode('utf-8')
        return image_64

    @staticmethod
    def check_periodically_if_server_online(api_server):
        server_offline = True
        while server_offline:
            try:
                response = requests.post(api_server + '/worker_check_server_status')
                server_offline = False
                print('Server back online')
                return True
            except requests.exceptions.ConnectionError:
                print(f'API Server {api_server} not available. Trying to reconnect...')
                time.sleep(2)

    def __fetch(self, url, json):
        return requests.post(self.api_server + url, json=json)


class ProgressCallback():
    def __init__(self, api_worker):
        self.api_worker = api_worker
        self.job_data = None

    def send_progress_to_api_server(self, progress, progress_data=None):
        self.api_worker.send_progress(self.job_data, progress, progress_data)
