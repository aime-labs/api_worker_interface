
import requests 
import os
import time
from datetime import datetime, timedelta
import socket
from multiprocessing import Barrier
from multiprocessing.managers import SyncManager
import io
import base64
from PIL.PngImagePlugin import PngInfo
from multiprocessing.dummy import Pool


SYNC_MANAGER_BASE_PORT  =  10042
SYNC_MANAGER_AUTH_KEY   = b"aime_api_worker"
SERVER_PARAMETERS = ['job_id', 'start_time', 'start_time_compute', 'auth']
DEFAULT_IMAGE_METADATA = [
    'prompt', 'negative_prompt', 'seed', 'base_steps', 'refine_steps', 'scale', 
    'aesthetic_score', 'negative_aesthetic_score', 'img2img_strength', 'base_sampler', 
    'refine_sampler', 'base_discretization', 'refine_discretization'
                          ]

class MyManager(SyncManager):
    pass


class APIWorkerInterface():
    f"""Interface for deep learning models to communicate with AIME-ML-API

    Args:
        api_server (str): Address of API Server. Example: 'http://api.aime.team'.
        job_type (str): Type of job . Example: "stable_diffusion_xl_txt2img".
        auth_key (str): key to authorize worker to connect with API Server.
        gpu_id (int, optional): ID of GPU the worker runs on. Defaults to 0.
        world_size (int, optional): Number of used GPUs the worker runs on. Defaults to 1.
        rank (int, optional): ID of current GPU if world_size > 1. Defaults to 0.
        gpu_name (str, optional): Name of GPU the worker runs on. Defaults to None.
        progress_received_callback (function, optional): Callback function with http response as argument, called when API server received all data from send_progress(..). Defaults to None.
        image_metadata_params (list, optional): Parameters to add as metadata to images (Currently only 'PNG'). Defaults to {DEFAULT_IMAGE_METADATA}.

    Example usage simple:
        
        api_worker = APIWorkerInterface()
        while True:
            job_data = api_worker.job_request('http://api.aime.team', 'llama2_chat', <auth_key>)
            output = do_deep_learning_worker_calculations(job_data, ...)
            api_worker.send_job_results(output)


    Example usage with progress:

        api_worker = APIWorkerInterface('http://api.aime.team', 'llama2_chat', <auth_key>)
        while True:
            job_data = api_worker.job_request()
            
            for step in deep_learning_worker_calculation:
                progress_in_percent = round(step*100/len(deep_learning_worker_calculation))
                progress_data = do_deep_learning_worker_calculation_step(job_data, ...)
                if api_worker.progress_data_received:
                    api_worker.send_progress(progress_in_percent, progress_data)
            output = get_result()
            api_worker.send_job_results(output)


    Example usage with callback:

        def progress_callback(api_worker, progress, progress_data):
            if api_worker.progress_data_received:
                api_worker.send_progress(progress, progress_data)


        api_worker = APIWorkerInterface('http://api.aime.team', 'llama2_chat', <auth_key>)
        callback = Callback(api_worker)
        
        while True:
            job_data = api_worker.job_request()
            output = do_deep_learning_worker_calculation(job_data, progress_callback, api_worker, ...)
            api_worker.send_progress(progress, progress_data)


    Example usage with callback class:

        class Callback():

            def __init__(self, api_worker):
                self.api_worker = api_worker


            def progress_callback(self, progress, progress_data):
                if self.api_worker.progress_data_received:
                    self.api_worker.send_progress(progress, progress_data)
            
            def result_callback(self, result):
                self.api_worker.send_job_results(result) 


        api_worker = APIWorkerInterface('http://api.aime.team', 'llama2_chat', <auth_key>)
        callback = Callback(api_worker)
        
        while True:
            job_data = api_worker.job_request()
            do_deep_learning_worker_calculation(job_data, callback.result_callback, callback.progress_callback, ...)
        
    """    
    def __init__(self, api_server, job_type, auth_key, gpu_id=0, world_size=1, rank=0, gpu_name=None, progress_received_callback=None, progress_error_callback=None, image_metadata_params=DEFAULT_IMAGE_METADATA):
        f"""Constructor
        Args:
            api_server (str): Address of API Server. Example: 'http://api.aime.team'.
            job_type (str): Type of job . Example: "stable_diffusion_xl_txt2img".
            auth_key (str): key to authorize worker to connect with API Server.
            gpu_id (int, optional): ID of GPU the worker runs on. Defaults to 0.
            world_size (int, optional): Number of used GPUs the worker runs on. Defaults to 1.
            rank (int, optional): ID of current GPU if world_size > 1. Defaults to 0.
            gpu_name (str, optional): Name of GPU the worker runs on. Defaults to None.
            progress_received_callback (function, optional): Callback function with http response as argument, called when API server received all data from send_progress(..). Defaults to None.
            image_metadata_params (list, optional): Parameters to add as metadata to images (Currently only 'PNG'). Defaults to {DEFAULT_IMAGE_METADATA}

        """        
        self.api_server = api_server
        self.job_type = job_type
        self.auth_key = auth_key
        self.gpu_id = gpu_id
        self.world_size = world_size
        self.rank = rank
        self.gpu_name = gpu_name
        self.progress_received_callback = progress_received_callback
        self.progress_error_callback = progress_error_callback
        self.image_metadata_params = image_metadata_params
        self.worker_name = self.__make_worker_name()
        self.manager, self.barrier = self.__init_manager_and_barrier()
        self.progress_data_received = True
        self.current_job_data = None
        self.worker_start = True
        self.async_check_server_connection()


    def job_request(self):
        """Worker requests a job from API Server on route /worker_job_request

        Returns:
            dict: job_data with worker [INPUT] parameters received from API server
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
                    self.check_periodically_if_server_online()
                    continue
                
                if response.status_code == 200:
                    job_data = response.json()   
                    cmd = job_data.get('cmd')
                    if cmd == 'job':
                        have_job = True
                    elif cmd == 'error':
                        exit("! API server responded with error: " + str(job_data.get('msg', "unknown")))
                    elif cmd == 'warning':
                        print("! API server responded with warning: " + str(job_data.get('msg', "unknown")))
                        counter += 1
                        self.check_periodically_if_server_online()
                        if counter > 3:
                            exit("API server responded with warning: " + str(job_data.get('msg', "unknown")))
                    else:
                        exit("! API server responded with unknown cmd: " + str(cmd))

        if self.world_size > 1:
            # hold all GPU processes here until we have a new job                     
            self.barrier.wait()
        self.current_job_data = job_data
        return job_data


    def send_job_results(self, results):
        """Process/convert job results and send it to API Server on route /worker_job_result

        Args:
            results (dict): worker [OUTPUT] result parameters (f.i. 'image', 'images' or 'text')
            Example results: {'images': [<PIL.Image.Image>, <PIL.Image.Image>, ...]}

        Returns:
            requests.models.Response: Http response from API server to the worker, 
            Example_response.json() : 
                API Server received data without problems:          {'cmd': 'ok'} 
                An error occured in API server:                     {'cmd': 'error', 'msg': <error message>} 
                API Server received data received with a warning:   {'cmd': 'warning', 'msg': <warning message>}
        """
        if self.rank == 0:        
            results = self.__prepare_output(results, True)
            try:
                return self.__fetch('/worker_job_result', results)
            except requests.exceptions.ConnectionError:
                print('Connection to server lost')
                return


    def send_progress(self, progress, progress_data=None):
        """Processes/converts job progress information and data and sends it to API Server on route /worker_job_progress asynchronously to main thread.
            When Api server received progress data, self.progress_data_received is set to True. Use progress_received_callback for response

        Args:
            progress (int): current progress (f.i. percent or number of generated tokens)
            progress_data (dict, optional): dictionary with progress_images or text while worker is computing. Defaults to None.
            Example progress data: {'progress_images': [<PIL.Image.Image>, <PIL.Image.Image>, ...]}
        """
        if self.rank == 0:        
            payload = {'progress': progress, 'job_id': self.current_job_data['job_id']}
            payload['progress_data'] = self.__prepare_output(progress_data, False)
            _ = self.__fetch_async('/worker_job_progress', payload)


    def __init_manager_and_barrier(self):
        """Register barrier in MyManager, initialize MyManager and assign them to APIWorkerInterface.barrier and APIWorkerInterface.manager
        """
        manager, barrier = None, None
        if self.world_size > 1:
            MyManager.register("barrier", self.__get_barrier)
            manager = MyManager(("127.0.0.1", SYNC_MANAGER_BASE_PORT + self.gpu_id), authkey=SYNC_MANAGER_AUTH_KEY)
            # multi GPU synchronization required
            if self.rank == 0:
                barrier = Barrier(world_size)
                manager.start()
            else:
                time.sleep(2)   # manager has to be started first to connect
                manager.connect()
                barrier = manager.barrier()    
        return manager, barrier


    def __make_worker_name(self):
        """Make a name for the worker based on worker hostname gpu name and gpu id: 

        Returns:
            str: name of the worker like <hostname>_<gpu_name>_<gpu_id> if gpu_name is given, , else <hostname>_GPU_<gpu_id>
        """        
        worker_name = socket.gethostname()
        for id in range(self.world_size):
            if self.gpu_name:
                worker_name += f'_{self.gpu_name}_{id+self.gpu_id}'
            else:
                worker_name += f'_GPU{id+self.gpu_id}'
        return worker_name


    def __convert_output_types_to_string_representation(self, output_data, output_name, output_description):
        """Converts parameters output data from type 'image' and 'image_list' to base64 strings. 

        Args:
            output_data (dict): worker [OUTPUT] parameter dictionary
            finished (bool): Set True fro sending end result, False for progress data

        Returns:
            dict: worker [OUTPUT] parameter dictionary with data converted to base64 string
        """
        if output_name in output_data:
            output_type = output_description.get('type') 
            if output_type == 'image':
                output_data[output_name] = self.__convert_image_to_base64_string(
                    output_data[output_name], output_description.get('image_format', 'PNG'))
            elif output_type == 'image_list':
                output_data[output_name] = self.__convert_image_list_to_base64_string(
                    output_data[output_name], output_description.get('image_format', 'PNG'))


    def __prepare_output(self, output_data, finished):
        f"""Adds {SERVER_PARAMETERS} to output_data. Converts parameters in output data from type 'image' and 'image_list' to base64 strings. 
        Adds [OUTPUT] parameters found in job_data[output_description/progress_description] to output_data

        Args:
            output_data (dict): worker [OUTPUT] parameter dictionary
            finished (bool): Set True fro sending end result, False for progress data

        Returns:
            dict: worker [OUTPUT] parameter dictionary with data converted to base64 string
        """
        if output_data:
            if finished:
                for parameter in SERVER_PARAMETERS:
                    output_data[parameter] = self.current_job_data.get(parameter)
            mode = 'output' if finished else 'progress'
            descriptions = self.current_job_data.get(f'{mode}_descriptions')
            for output_name, output_description in descriptions.items():
                self.__convert_output_types_to_string_representation(output_data, output_name, output_description)
                if finished:
                    if output_name in self.current_job_data and output_name not in output_data:
                        output_data[output_name] = self.current_job_data[output_name]
        return output_data


    def __convert_image_to_base64_string(self, image, image_format):
        """Converts given PIL image to base64 string with given image_format and image metadata parsed from current_job_data.

        Args:
            image (PIL.PngImagePlugin.PngImageFile): Python pillow image to be converted
            image_format (str): Image format. f.i. 'PNG', 'JPG'

        Returns:
            str: base64 string of image
        """        
        with io.BytesIO() as buffer:
            if image_format == 'PNG':
                image.save(buffer, format=image_format, pnginfo=self.get_pnginfo_metadata())
            else:
                image.save(buffer, format=image_format)
            
            image_64 = f'data:image/{image_format};base64,' + base64.b64encode(buffer.getvalue()).decode('utf-8')
        return image_64


    def __convert_image_list_to_base64_string(self, list_images, image_format):
        """Converts given list of PIL images to base64 string with given image_format and image metadata parsed from current_job_data.

        Args:
            list_images (list [PIL.PngImagePlugin.PngImageFile, ..]): List of python pillow images to be converted
            image_format (str): Image format. f.i. 'PNG', 'JPG'

        Returns:
            str: base64 string of images
        """        
        image_64 = ''.join(self.__convert_image_to_base64_string(image, image_format) for image in list_images)
        return image_64


    def get_pnginfo_metadata(self):
        """Parses image metadata from current_job_data.

        Returns:
            PIL.PngImagePlugin.PngInfo: PngInfo Object with metadata for PNG images
        """        
        metadata = PngInfo()
        for parameter_name in self.image_metadata_params:
            parameter = self.current_job_data.get(parameter_name)
            if parameter:
                metadata.add_text(parameter_name, str(parameter))
        metadata.add_text('Comment', 'Generated with AIME ML API')
        
        return metadata


    def async_check_server_connection(self):
        self.__fetch_async('/worker_check_server_status', {'auth_key': self.auth_key, 'job_type': self.job_type})


    def check_periodically_if_server_online(self, interval_seconds=1):
        """Checking periodically every <interval_seconds> if server is online by post request on route /worker_check_server_status

        Args:
            interval_seconds (int): Interval in seconds to update check if server is online

        Returns:
            bool: True if server is available again
        """        
        server_offline = True
        start_time = datetime.now()
        while server_offline:
            try:
                response = self.__fetch('/worker_check_server_status', {'auth_key': self.auth_key, 'job_type': self.job_type})
                server_offline = False
                print('\nServer back online')
                return True
            except requests.exceptions.ConnectionError:
                duration_being_offline = datetime.now() - start_time
                duration_being_offline = duration_being_offline - timedelta(microseconds=duration_being_offline.microseconds)
                print(f'Connection to API Server {self.api_server} offline for {duration_being_offline}. Trying to reconnect... ', end='\r')
                time.sleep(interval_seconds)


    def __print_server_status(self, response):
        """_summary_

        Args:
            online (bool): _description_
            response (requests.models.Response or requests.exceptions.ConnectionError or ): Expects response from API server. Also takes f API server connection is offline, 
        """        
        
        if type(response) is not requests.exceptions.ConnectionError:
            response_json = response.json()
            status = 'online'
            if response_json.get('msg'):
                message_str = f'But server responded with: {response_json.get("cmd")}: {response_json.get("msg")}'
            else:
                message_str = ''

        else:
            message_str = f'Error: {response}'
            status = 'offline'

        output_str = \
            '--------------------------------------------------------------\n\n' +\
            f'           API server {self.api_server} {status}\n' +\
            message_str +\
            '\n\n--------------------------------------------------------------'
        print(output_str)


    def __fetch(self, route, json=None):
        """Send post request on given route on API server with given arguments

        Args:
            route (str): Route on API server for post request
            json (dict, optional): Arguments for post request

        Returns:
            Response: post request response
        """        
        return requests.post(self.api_server + route, json=json)



    def __fetch_async(self, route, json=None):
        """Send non-blocking post request on given route on API server with given arguments. After success request callback is called.

        Args:
            route (str): Route on API server for post request, f.i. '/worker_job_progress'
            json (dict, optional): Arguments for post request
        """        
        pool = Pool()
        self.progress_data_received = False
        pool.apply_async(self.__fetch, args=[route, json], callback=self.__async_fetch_callback, error_callback=self.__async_fetch_error_callback)


    def __async_fetch_callback(self, response):
        """Is called when API server sent a response from __fetch_async. 
        Sets progress_data_received = True and calls progress_received_callback, if given to ApiWorkerInterface

        Args:
            response (requests.models.Response): Http response from API server, 
            Example_response.json() : 
                API Server received data without problems:          {'cmd': 'ok'} 
                An error occured in API server:                     {'cmd': 'error', 'msg': <error message>} 
                API Server received data received with a warning:   {'cmd': 'warning', 'msg': <warning message>}
        """
        self.progress_data_received = True     
        if self.worker_start:
            self.__print_server_status(response)
            self.worker_start = False
        elif self.progress_received_callback:
            self.progress_received_callback(response)


    def __async_fetch_error_callback(self, response):
        """Is called when API server received progress data from worker. 
        Sets progress_data_received = True and calls progress_received_callback, if given to ApiWorkerInterface

        Args:
            response (requests.exceptions.ConnectionError): Error , 
        """
        if self.worker_start:
            self.__print_server_status(response)
            self.worker_start = False
        else:
            if self.progress_received_error_callback:
                self.progress_received_error_callback(response)


    def __get_barrier(self):
        return self.barrier
