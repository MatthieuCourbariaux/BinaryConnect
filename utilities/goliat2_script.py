import os
import time

# gpu
nb_gpu = 4
delay_between_lauches = 5*60 # 5 minutes
time_per_job = 7.5*60*60 # CIFAR10 9h in float32 with 1.3 more neurones, partial sum =1, no range updates, 200 epochs
gpu_id = 0                    

# hyper parameters lists
comp_NOB = [9,11]
up_NOB = [11,13]
NOIB = [8] 
dynamic_range = [1]

for i in comp_NOB:
    for j in up_NOB:
        for k in NOIB:
            for l in dynamic_range:

                command = "THEANO_FLAGS='device=gpu"+str(gpu_id)+"' python main.py "+str(i)+" "+str(j)+" "+str(k)+" "+str(l)+" &> "+str(i)+"_"+str(j)+"_"+str(k)+"_"+str(l)+".txt &"
                os.system(command)
                print command

                if gpu_id == nb_gpu - 1:

                    gpu_id = 0
                    time.sleep(time_per_job)
                    print " "

                else:
                
                    gpu_id = gpu_id+1
                    time.sleep(delay_between_lauches)
                    
                    