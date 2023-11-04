
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

DEFAULT_IMAGE_METADATA = [
    'prompt', 'negative_prompt', 'seed', 'base_steps', 'refine_steps', 'scale', 
    'aesthetic_score', 'negative_aesthetic_score', 'img2img_strength', 'base_sampler', 
    'refine_sampler', 'base_discretization', 'refine_discretization'
                          ]

class MyManager(SyncManager):
    pass


class APIWorkerInterface():
    """Interface for deep learning models to communicate with AIME-ML-API
    """    

    manager = None
    barrier = None

    @staticmethod
    def get_barrier():
        return APIWorkerInterface.barrier

    def __init__(self, api_server, job_type, auth_key, gpu_id=0, world_size=1, rank=0, gpu_name=None, progress_received_callback=None, image_metadata_params=DEFAULT_IMAGE_METADATA):
        """Constructor

        Args:
            api_server (str): Address of API Server. Example: 'http://api.aime.team'
            job_type (str): Type of job . Example: "stable_diffusion_xl_txt2img"
            auth_key (str): key to authorize worker to connect with API Server
            gpu_id (int, optional): ID of GPU the worker runs on. Defaults to 0.
            world_size (int, optional): Number of used GPUs the worker runs on. Defaults to 1.
            rank (int, optional): ID of current GPU if world_size > 1. Defaults to 0.
            gpu_name (str, optional): Name of GPU the worker runs on. Defaults to None.
        """        
        self.api_server = api_server
        self.job_type = job_type
        self.auth_key = auth_key
        self.gpu_id = gpu_id
        self.world_size = world_size
        self.rank = rank
        self.gpu_name = gpu_name
        self.progress_received_callback = progress_received_callback
        self.image_metadata_params = image_metadata_params
        self.progress_data_received = True
        self.worker_name = self.__make_worker_name()
        self.__init_manager_and_barrier()
        self.first_request = True
        
        
    def __init_manager_and_barrier(self):
        """Register barrier in MyManager, initialize MyManager and assign them to APIWorkerInterface.barrier and APIWorkerInterface.manager
        """        
        if self.world_size > 1:
            MyManager.register("barrier", APIWorkerInterface.get_barrier)
            APIWorkerInterface.manager = MyManager(("127.0.0.1", SYNC_MANAGER_BASE_PORT + self.gpu_id), authkey=SYNC_MANAGER_AUTH_KEY)
            # multi GPU synchronization required
            if self.rank == 0:
                APIWorkerInterface.barrier = Barrier(world_size)
                APIWorkerInterface.manager.start()
            else:
                time.sleep(2)   # manager has to be started first to connect
                APIWorkerInterface.manager.connect()
                APIWorkerInterface.barrier = APIWorkerInterface.manager.barrier()


    def __make_worker_name(self):
        """Make a name for the worker based on gpu_id and gpu_name

        Returns:
            str: created name of worker
        """        
        worker_name = socket.gethostname()
        for id in range(self.world_size):
            if self.gpu_name:
                worker_name += f'_{self.gpu_name}_{id+self.gpu_id}'
            else:
                worker_name += f'_GPU{id+self.gpu_id}'
        return worker_name



    def job_request(self):
        """Worker requests a job from API Server on route /worker_job_request

        Returns:
            dict: job_data with worker [INPUT] parameters
        """        
        job_data = None
        if self.rank == 0:
            have_job = False
            counter = 0
            while not have_job:
                request = { "auth": self.worker_name, "job_type": self.job_type, "auth_key": self.auth_key }
                try:
                    response = self.__fetch('/worker_job_request', request)
                except requests.exceptions.ConnectionError:
                    APIWorkerInterface.check_periodically_if_server_online()
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
                        APIWorkerInterface.check_periodically_if_server_online()
                        if counter > 3:
                            exit("API server responded with warning: " + str(job_data.get('msg', "unknown")))
                    else:
                        exit("! API server responded with unknown cmd: " + str(cmd))

        if self.world_size > 1:
            # hold all GPU processes here until we have a new job                     
            APIWorkerInterface.barrier.wait()

        return job_data


    def send_job_results(self, job_data, results):
        """Process/convert job results and send it to API Server on route /worker_job_result

        Args:
            job_data (dict): worker [INPUT] parameters 
            results (dict): worker result parameters ('image', 'images' or 'text')

        Returns:
            Response: Response to the worker 
        """
        response = None
        if self.rank == 0:        
            for parameter in ['job_id', 'start_time', 'start_time_compute', 'auth']:
                results[parameter] = job_data.get(parameter)

            results = self.__convert_output_types_to_string_representation(job_data, results, True)
            try:
                response = self.__fetch('/worker_job_result', results)
            except requests.exceptions.ConnectionError:
                print('Connection to server lost')
                return
        return response


    def send_progress(self, job_data, progress, progress_data=None):
        """Process/convert job progress information and data while worker is computing and send it to API Server on route /worker_job_progress

        Args:
            job_data (dict): worker [INPUT] parameters 
            progress (int): current progress (f.i. percent or number of generated tokens)
            progress_data (dict, optional): dictionary with progress_images or text while worker is computing. Defaults to None.

        Returns:
            Response: Response to the worker
        """
        response = None
        if self.rank == 0:        
            payload = {'progress': progress, 'job_id': job_data['job_id']}
            payload['progress_data'] = self.__convert_output_types_to_string_representation(job_data, progress_data, False)
            self.progress_data_received = False
            response = self.__fetch_async('/worker_job_progress', payload)
        return response


    def __convert_output_types_to_string_representation(self, job_data, output, finished):
        """Process/convert job progress information and data while worker is computing and send it to API Server on route /worker_job_progress

        Args:
            job_data (dict): worker [INPUT] parameters 
            output (dict): current progress (f.i. percent or number of generated tokens)
            finished (bool): Set True if end result, False if progress_data

        Returns:
            dict: Response to the worker
        """

        
        if output:
            mode = 'output' if finished else 'progress'
            descriptions = job_data.get(f'{mode}_descriptions')
            for output_name, output_description in descriptions.items():
                if output_name in output:
                    output_type = output_description.get('type') 
                    if output_type == 'image':
                        output[output_name] = self.__convert_image_to_base64_string(
                            output[output_name], output_description.get('image_format', 'PNG'), job_data)
                    elif output_type == 'image_list':
                        output[output_name] = self.__convert_image_list_to_base64_string(
                            output[output_name], output_description.get('image_format', 'PNG'), job_data)
                if finished:
                    if output_name in job_data and output_name not in output:
                        output[output_name] = job_data[output_name]
        return output


    def __convert_image_to_base64_string(self, image, image_format, job_data):
        """Converts given PIL image to base64 string with given image_format and image metadata parsed from given job_data.

        Args:
            image (PIL): Python pillow image to be converted
            image_format (str): Image format. f.i. 'PNG', 'JPG'
            job_data (dict): Dictionary containing image metadata

        Returns:
            str: base64 string of image
        """        
        image_64 = ''
        with io.BytesIO() as buffer:
            if image_format == 'PNG':
                image.save(buffer, format=image_format, pnginfo=self.get_pnginfo_metadata(job_data))
            else:
                image.save(buffer, format=image_format)
            
            image_64 = f'data:image/{image_format};base64,' + base64.b64encode(buffer.getvalue()).decode('utf-8')
        return image_64


    def __convert_image_list_to_base64_string(self, list_images, image_format, job_data):
        """Converts given list of PIL images to base64 string with given image_format and image metadata parsed from given job_data.

        Args:
            list_images (list): List of python pillow images to be converted
            image_format (str): Image format. f.i. 'PNG', 'JPG'
            job_data (dict): Dictionary containing image metadata

        Returns:
            str: base64 string of images
        """        
        image_64 = ''
        for image in list_images:            
            image_64 += self.__convert_image_to_base64_string(image, image_format, job_data)
        return image_64


    def get_pnginfo_metadata(self, job_data):
        """Parsing image metadata from given job_data

        Args:
            job_data (dict): Dictionary containing image metadata

        Returns:
            PIL.PngImagePlugin.PngInfo: PngInfo Object with metadata for PNG images
        """        
        metadata = PngInfo()
        for parameter in self.image_metadata_params:
            metadata.add_text(parameter, str(job_data.get(parameter)))
        metadata.add_text('Comment', 'Generated with AIME ML API')
        
        return metadata

    @staticmethod
    def check_periodically_if_server_online(interval_seconds=2):
        """Checking periodically every <interval_seconds> if server is online by post request on route /worker_check_server_status

        Args:
            interval_seconds (int): Interval in seconds to update check if server is online


        Returns:
            bool: True if server is available again
        """        
        server_offline = True
        counter = 0
        while server_offline:
            try:
                response = requests.post(self.api_server + '/worker_check_server_status')
                server_offline = False
                print('\nServer back online')
                return True
            except requests.exceptions.ConnectionError:
                print(f'API Server {self.api_server} not available. Trying to reconnect...({counter})', end='\r')
                counter += 1
                time.sleep(interval_seconds)

    def __fetch(self, route, json):
        """Send post request on given route on API server with given arguments

        Args:
            route (str): Route on API server for post request
            json (dict): Arguments for post request

        Returns:
            Response: post request response
        """        
        return requests.post(self.api_server + route, json=json)



    def __fetch_async(self, route, json):
        """Send non-blocking post request on given route on API server with given arguments. After success request callback is called.

        Args:
            route (str): Route on API server for post request
            json (dict): Arguments for post request
        """        
        pool = Pool()
        pool.apply_async(self.__fetch, args=[route, json], callback=self.__request_finished_callback)


    def __request_finished_callback(self, result):
        """Is called when API server received progress information from worker. Sets progress_data_received = True 

        Args:
            result (Response): Response of self.__fetch()
        """   
        self.progress_data_received = True
        if self.progress_received_callback:
            self.progress_received_callback()


class ProgressCallback():
    def __init__(self, api_worker):
        self.api_worker = api_worker
        self.job_data = None

    def send_progress_to_api_server(self, progress, progress_data=None):
        self.api_worker.send_progress(self.job_data, progress, progress_data)

