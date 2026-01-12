"""
@FileName：real_time.py
@Description：
@Author：Ferry
@Time：2025 12/18/25 4:27 PM
@Copyright：©2024-2025 ShanghaiTech University-RIMLAB
"""



import numpy as np
import os,sys,glob,subprocess,yaml,shutil,time,copy,argparse
code_dir = os.path.dirname(os.path.realpath(__file__))
from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat
try:
    multiprocessing.set_start_method('spawn')
except:
    pass


def run_one_video(data_dir,cfg1,port):
    cfg = copy.deepcopy(cfg1)
    name = data_dir.split('/')[-1]

    cfg['data_dir'] = data_dir
    debug_dir = './results/{}/'.format(name)
    cfg['debug_dir'] = debug_dir
    os.system(f'mkdir -p {debug_dir}')



    cfg['LOG'] = 1
    cfg['port'] = port
    tmp_config_dir = '/tmp/config_{}.yml'.format(name)
    with open(tmp_config_dir,'w') as ff:
        yaml.dump(cfg,ff)

    cmd = f'{code_dir}/../build/realtime_interface {tmp_config_dir}'
    print(cmd)
    try:
        subprocess.call(cmd,shell=True)
    except:
        pass




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', type=str, default='/home/ferry/data/Code2/Research/Inhand_Activate/BundleTrack/YCBInEOAT/inhand_object2')
    # parser.add_argument('--port', type=int, default=5555)

    parser.add_argument('--data_dir', type=str, default='/home/ferry/data/Code2/Research/Inhand_Activate/BundleTrack/YCBInEOAT/mustard0')
    parser.add_argument('--port', type=int, default=5555)



    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        raise RuntimeError(f"Make sure data_dir={data_dir} exists")

    code_dir = os.path.dirname(os.path.realpath(__file__))
    config_dir = f'{code_dir}/../config_ycbineoat.yml'
    with open(config_dir,'r') as ff:
        cfg = yaml.safe_load(ff)

    run_one_video(args.data_dir,cfg,args.port)