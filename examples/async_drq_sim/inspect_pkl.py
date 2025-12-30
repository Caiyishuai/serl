import pickle as pkl
import numpy as np

file_path = "examples/async_drq_sim/success_trajs_100.pkl"

with open(file_path, "rb") as f:
    trajs = pkl.load(f)

print(f"Number of items: {len(trajs)}")
if len(trajs) > 0:
    item = trajs[0]
    print(f"Item keys: {item.keys()}")
    
    if 'infos' in item:
        infos = item['infos']
        print(f"Infos type: {type(infos)}")
        if isinstance(infos, list) and len(infos) > 0:
            print(f"Infos length: {len(infos)}")
            first_info = infos[0]
            print(f"First info: {first_info}")
