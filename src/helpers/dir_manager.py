import os
import json
import torch
import shutil
import csv
import numpy as np

from types import SimpleNamespace
from typing import Callable
from collections import namedtuple

from ..utils import load_json, download_hpc_model
from ..config import config

BASE_DIR = f'{config.base_dir}/trained_models'


class DirManager:
    """ Class which manages logs, models and config files """

    ### Methods for Initialisation of Object ###########################
    
    def __init__(self, exp_name:str=None, temp:bool=False):
        if temp:
            print("using temp directory")
            self.exp_name = 'temp'
            self.del_temp_dir()
        else:
            self.exp_name = exp_name

        self.make_dir()
        self.log = self.make_logger(file_name='log')
    
    def del_temp_dir(self):
        """deletes the temp, unsaved experiments directory"""
        if os.path.isdir(f'{BASE_DIR}/temp'): 
            shutil.rmtree(f'{BASE_DIR}/temp')        

    def make_dir(self):
        """makes experiments directory"""
        os.makedirs(self.path)
        os.mkdir(f'{self.path}/models')

    def make_logger(self, file_name:str)->Callable:
        """creates logging function which saves prints to txt file"""
        log_path = f'{self.path}/{file_name}.txt'
        open(log_path, 'a+').close()  
        
        def log(*x):
            print(*x)    
            with open(log_path, 'a') as f:
                for i in x:
                    f.write(str(i) + ' ')
                f.write('\n')
        return log
    
    ### Utility Methods ################################################
    
    @property
    def path(self):
        """returns base experiment path"""
        return f'{BASE_DIR}/{self.exp_name}'

    def save_args(self, name:str, args:namedtuple):
        """saves arguments into json format"""
        config_path = f'{self.path}/{name}.json'
        with open(config_path, 'x') as jsonFile:
            json.dump(args.__dict__, jsonFile, indent=4)

    ### Methods for Logging performance Set up Directories #############
    
    def reset_cls_logger(self):
        self.perf = np.zeros(3)
            
    def update_cls_logger(self, out):
        self.perf += [out.loss.item(), out.hits, out.num_preds]
        
    def print_perf(self, epoch:int, k:int, print_len:int, mode='train'):
        """returns and logs performance"""
        loss = f'{self.perf[0]/print_len:.3f}'
        acc  = f'{self.perf[1]/self.perf[2]:.3f}'
        if mode == 'train':
            self.update_curve(mode, epoch, float(loss), float(acc))
            self.log(f'{epoch:<3} {k:<5}   loss {loss}   acc {acc}')
        elif mode == 'dev':
            self.update_curve(mode, epoch, float(loss), float(acc))
            self.log(f'{epoch:<3} DEV   loss {loss}   acc {acc}')
        elif mode == 'test':
            self.log(f'{epoch:<3} TEST  loss {loss}   acc {acc}')
        self.reset_cls_logger()
        return loss, acc 
                              
    def update_curve(self, mode, *args):
        """ logs any passed arguments into a file"""
        with open(f'{self.path}/{mode}.csv', 'a+') as f:
            writer = csv.writer(f)
            writer.writerow(args)

    ### Methods for loading existing dir ##############################
    
    @classmethod
    def load_dir(cls, exp_name:str, hpc=False)->'DirManager':
        dir_manager = cls.__new__(cls)
        if hpc: 
            dir_manager.exp_name = 'hpc/'+exp_name
            download_hpc_model(exp_name)
        else:
            dir_manager.exp_name = exp_name
        
        dir_manager.log = print
        return dir_manager
    
    def load_args(self, name:str)->SimpleNamespace:
        args = load_json(f'{self.path}/{name}.json')
        return SimpleNamespace(**args)

    def load_curve(self, mode='train'):
        float_list = lambda x: [float(i) for i in x] 
        with open(f'{self.path}/{mode}.csv') as fp:
            reader = csv.reader(fp, delimiter=",", quotechar='"')
            data_read = [float_list(row) for row in reader]
        return tuple(zip(*data_read))
    
