import pickle
import itertools
from src.constant import *
import os

transformations =  ['frost', 'brightness']
vision_tasks = ['D', 'I']
req_type = ['cp', 'pp']
objects = ['person', 'bus', 'bird']

name_dict = {'L': 'l', 'C|L': 'cl', 'D': 'd', 'S|C,L': 'scl', 'I': 's'}
pickle_dir = 'metric_results'

for t in transformations:
    for obj in objects:
        if t == 'frost' and obj == 'bird':
            continue
        if t == 'brightness' and obj == 'bus':
            continue
        for vt in vision_tasks:
            if vt == 'D':
                human_thld = human_thld_D[t][obj]
            else:
                human_thld = human_thld_I[t][obj]
            
            for v in human_thld.keys():
                for req in human_thld[v]:
                    print('---' + t+ ' ' + obj + ' ' + vt + ' ' + v + ' ' + req + '-------')

                    thlds = human_thld[v][req]
                    num_thlds = len(thlds.keys())
                    
                    for model_index in range(13):
                        if vision_tasks == 'I' and model_index >= 10:
                            continue
                        print('Model ' + str(model_index))
                        PR_results = []
                        mAP_results = []
                        for th in thlds:
                            filename = vt + '_' + obj + '_' +t + '_'+ str(th) + '_bootstrap.pickle'
                            #print(filename)
                            if not os.path.exists(pickle_dir + '/'+filename):
                                continue
                            #print(filename)
                            with open(pickle_dir + '/'+filename, 'rb') as f:
                                results = pickle.load(f)
                            name_to_check = req + '_' + name_dict[v]+'_results'
                            
                            all_results = list(itertools.chain.from_iterable([results[model_index][i][name_to_check] for i in range(50) if i in results[model_index] ]))# ()
                            only_numbers = [x[1] for x in all_results if x[0] == obj if x[1] is not None]
                            only_numbers = [x if x < 1 and x > -1 else 0 for x in only_numbers]
                            if len(only_numbers) > 0:
                                PR_results.append(sum(only_numbers)/len(only_numbers))

                            if v == 'I' or v == 'D':
                                mAP_name = req + '_' + name_dict[v]+'_mAP'
                                all_results = list(itertools.chain.from_iterable([results[model_index][i][mAP_name] for i in range(50) if i in results[model_index] ]))# ()

                                only_numbers = [x[1]-x[2]  for x in all_results if x[0] == obj if x[1] is not None]
                                only_numbers = [x if x < 1 and x > -1 else 0 for x in only_numbers]
                                #print(only_numbers)
                                #exit()
                                if len(only_numbers) > 0:
                                    mAP_results.append(sum(only_numbers)/len(only_numbers))

                        if len(PR_results) > 0:
                            print('PRs for ' + name_to_check)
                            print(max(PR_results))
                        if len(mAP_results) > 0:
                            print('mAPs for ' + mAP_name)
