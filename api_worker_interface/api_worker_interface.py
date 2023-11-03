
import requests 
import os
import time
import socket
from multiprocessing import Barrier
from multiprocessing.managers import SyncManager
import io
import base64
from PIL.PngImagePlugin import PngInfo
from multiprocessing.dummy import Pool

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

    def __init__(self, api_server, job_type, auth_key, gpu_id=0, world_size=1, rank=0, gpu_name=None):
        self.api_server = api_server
        self.world_size = world_size
        self.gpu_name = gpu_name
        self.rank = rank
        self.job_type = job_type
        self.auth_key = auth_key
        self.worker_name = self.__make_worker_name(gpu_id)
        self.progress_data_received = True
        

        if world_size > 1:
            MyManager.register("barrier", APIWorkerInterface.get_barrier)
            APIWorkerInterface.manager = MyManager(("127.0.0.1", SYNC_MANAGER_BASE_PORT + gpu_id), authkey=SYNC_MANAGER_AUTH_KEY)
            # multi GPU synchronization required
            if rank == 0:
                APIWorkerInterface.barrier = Barrier(world_size)
                APIWorkerInterface.manager.start()
            else:
                time.sleep(2)   # manager has to be started first to connect
                APIWorkerInterface.manager.connect()
                APIWorkerInterface.barrier = APIWorkerInterface.manager.barrier()


    def __make_worker_name(self, gpu_id):
        worker_name = socket.gethostname()
        for id in range(self.world_size):
            if self.gpu_name:
                worker_name += f'_{self.gpu_name}_{id+gpu_id}'
            else:
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
        for parameter in ['job_id', 'start_time', 'start_time_compute', 'auth']:
            results[parameter] = job_data.get(parameter)
        output_descriptions = job_data['output_descriptions']
        for output_name, output_description in output_descriptions.items():
            if output_name in job_data and output_name not in results:
                results[output_name] = job_data[output_name]
            # convert output types to a string representation
            if output_name in results:
                output_type = output_description.get('type') 
                if output_type == 'image':
                    results[output_name] = self.convert_image_to_base64_string(
                        results[output_name], output_description.get('image_format', 'PNG'), job_data)
                elif output_type == 'image_list':
                    results[output_name] = self.convert_image_list_to_base64_string(
                        results[output_name], output_description.get('image_format', 'PNG'), job_data)
        try:
            response = self.__fetch('/worker_job_result', results)
        except requests.exceptions.ConnectionError:
            print('Server not reachable')
            return
        return response


    def send_progress(self, job_data, progress, progress_data=None):
        payload = {'progress': progress, 'job_id': job_data['job_id']}
        progress_descriptions = job_data['progress_descriptions']
        if progress_data:
            # convert output types to a string representation
            for output_name, output_description in progress_descriptions.items():
                if output_name in progress_data:
                    output_type = output_description.get('type') 
                    if output_type == 'image':
                        progress_data[output_name] = self.convert_image_to_base64_string(
                            progress_data[output_name], progress_descriptions.get('image_format', 'PNG'), job_data)
                    elif output_type == 'image_list':
                        progress_data[output_name] = self.convert_image_list_to_base64_string(
                            progress_data[output_name], progress_descriptions.get('image_format', 'PNG'), job_data)

        payload['progress_data'] = progress_data
        self.progress_data_received = False
        response = self.__fetch_async('/worker_job_progress', payload)
        return response


    def convert_image_to_base64_string(self, image, image_format, job_data):
        image_64 = ''
        with io.BytesIO() as buffer:
            if image_format == 'PNG':
                png_metadata = self.get_pnginfo_metadata(job_data)
                image.save(buffer, format=image_format, pnginfo=png_metadata)
            else:
                image.save(buffer, format=image_format)
            
            image_64 = f'data:image/{image_format};base64,' + base64.b64encode(buffer.getvalue()).decode('utf-8')
        return image_64

    def convert_image_list_to_base64_string(self, list_images, image_format, job_data):
        image_64 = ''
        for image in list_images:            
            image_64 += self.convert_image_to_base64_string(image, image_format, job_data)
        return image_64


    def get_pnginfo_metadata(self, job_data):
        image_metadata_choice = [
            'prompt', 'negative_prompt', 'seed', 'base_steps', 'refine_steps', 'scale', 
            'aesthetic_score', 'negative_aesthetic_score', 'img2img_strength', 'base_sampler', 
            'refine_sampler', 'base_discretization', 'refine_discretization'
                          ]
        metadata = PngInfo()
        for parameter in image_metadata_choice:
            metadata.add_text(parameter, str(job_data.get(parameter)))
        metadata.add_text('Comment', 'Generated with AIME ML API')
        
        return metadata

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



    def __fetch_async(self, url, json):
        pool = Pool()
        pool.apply_async(self.__fetch, args=[url, json], callback=self.request_finished_callback)

    def request_finished_callback(self, result):
        self.progress_data_received = True

class ProgressCallback():
    def __init__(self, api_worker):
        self.api_worker = api_worker
        self.job_data = None

    def send_progress_to_api_server(self, progress, progress_data=None):
        self.api_worker.send_progress(self.job_data, progress, progress_data)

