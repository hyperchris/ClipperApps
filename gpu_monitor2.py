import sys
import time

import pynvml

def getGpuUtilization(pynvml_handle):
  return float(pynvml.nvmlDeviceGetUtilizationRates(pynvml_handle).gpu)

gpu_id = 0
if (len(sys.argv) > 1):
  gpu_id = int(sys.argv[1])

pynvml.nvmlInit()
pynvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

for i in range(1000):
  gpu_utilization = getGpuUtilization(pynvml_handle)
  print(gpu_utilization)
  time.sleep(0.1)
