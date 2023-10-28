from IPython.display import clear_output
import time 
import os
i = None
while i != 'n':
    clear_output(wait=True)
    os.system("nvidia-smi")
    os.system("free")
    print('Do you want to refresh? y/n')
    i = input()
    time.sleep(0.5)
    pass