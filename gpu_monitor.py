import psutil
import datetime
from subprocess import Popen, PIPE
import matplotlib.pyplot as plt
import time
import numpy as np

def convertTime(t):
  tt = t.split(":")
  tHour = float(tt[0])
  tMin = float(tt[1])
  tSec = float(tt[2])
  tOutput = 3600 * tHour + 60 * tMin + tSec
  return tOutput

# # ========================= CPU =========================
# cpu_percent = psutil.cpu_percent()
# virtual_memory = psutil.virtual_memory()
# memory_percent = virtual_memory[2]
# memory_total = virtual_memory[0]

# print("[CPU] memory_total = %s, memory_percent = %s, cpu_percent = %s" % (memory_total, memory_percent, cpu_percent))

# # ========================= GPU =========================
# p = Popen(["nvidia-smi" ,"--query-gpu=utilization.gpu,temperature.gpu,fan.speed,power.draw,power.limit", "--format=csv,noheader,nounits"], stdout=PIPE)
# pout, _ = p.communicate()

# tmp = str(pout.rstrip()).split(', ')
# gpu_utilization = tmp[0]
# gpu_temporature = tmp[1]
# gpu_fanspeed = tmp[2]
# gpu_fanpower_cur = tmp[3]
# gpu_fanpower_max = tmp[4]

# print("[GPU] gpu_utilization = %s%%, gpu_temporature = %sC, fan_speed = %s%% with %sW/%sW" % (gpu_utilization, gpu_temporature, gpu_fanspeed, gpu_fanpower_cur, gpu_fanpower_max))

f = open('gpu_log.txt', 'w')

time_array = []

cpu_dict = dict()
cpu_dict['cpu_percent'] = []
cpu_dict['memory_percent'] = []
cpu_dict['memory_total'] = []

gpu_dict = dict()
gpu_dict['gpu_utilization'] = []
gpu_dict['gpu_temporature'] = []
gpu_dict['gpu_fanspeed'] = []
gpu_dict['gpu_fanpower_cur'] = []
gpu_dict['gpu_fanpower_max'] = []


try:
  while True:
    cur_time = datetime.datetime.now()
    tt = convertTime(str(cur_time).split()[1])
    time_array.append(tt)

    cpu_percent = psutil.cpu_percent()
    virtual_memory = psutil.virtual_memory()
    memory_percent = virtual_memory[2]
    memory_total = virtual_memory[0]
    cpu_dict['cpu_percent'].append(float(cpu_percent))
    cpu_dict['memory_percent'].append(float(memory_percent))
    cpu_dict['memory_total'].append(float(memory_total))

    p = Popen(["nvidia-smi" ,"--query-gpu=utilization.gpu,temperature.gpu,fan.speed,power.draw,power.limit", "--format=csv,noheader,nounits"], stdout=PIPE)
    pout, _ = p.communicate()
    # print(pout)
    # print(type(pout))
    tmp = pout.decode("utf-8").rstrip().split('\n')[0].split(', ')
    gpu_utilization = tmp[0]
    gpu_temporature = tmp[1]
    gpu_fanspeed = tmp[2]
    gpu_fanpower_cur = tmp[3]
    gpu_fanpower_max = tmp[4]
    gpu_dict['gpu_utilization'].append(float(gpu_utilization))
    gpu_dict['gpu_temporature'].append(float(gpu_temporature))
    gpu_dict['gpu_fanspeed'].append(float(gpu_fanspeed))
    gpu_dict['gpu_fanpower_cur'].append(float(gpu_fanpower_cur))
    gpu_dict['gpu_fanpower_max'].append(float(gpu_fanpower_max))

    # f.write('%s-%s-%s-%s-%s-%s-%s\n' % (str(tt), cpu_percent, memory_percent, gpu_utilization, gpu_temporature, gpu_fanspeed, gpu_fanpower_cur))
    f.write('%s\n' % gpu_utilization)

    time.sleep(0.1)

except KeyboardInterrupt:   
    pass 
    
print("Done")
