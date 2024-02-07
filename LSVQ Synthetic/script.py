import os
import time

start = time.time()

#for i in range(1,3):
#    os.system('python resize2.py --len 0.1 --batch '+ str(i))
#    duration = (time.time() -start)/3600
#    print(duration)

# rate = 0.1, total video scenes = # of videos x rate      
for i in range(1,26):

    os.system('python mpeg1.py --len 0.1 --batch ' + str(i))
    os.system('python mpeg2.py --len 0.1 --batch '+ str(i))
    os.system('python mpeg3.py --len 0.1 --batch '+ str(i))
    os.system('python h264_1.py --len 0.1 --batch '+ str(i))
    os.system('python h264_2.py --len 0.1 --batch '+ str(i))
    os.system('python h264_3.py --len 0.1 --batch '+ str(i))
    os.system('python resize1.py --len 0.1 --batch '+ str(i))
    os.system('python resize2.py --len 0.1 --batch '+ str(i))
    os.system('python resize3.py --len 0.1 --batch '+ str(i))
    os.system('python original.py --len 0.1 --batch '+ str(i))
    os.system('python inter1.py --len 0.1 --batch ' + str(i))
    os.system('python inter2.py --len 0.1 --batch '+ str(i))
    os.system('python inter3.py --len 0.1 --batch '+ str(i))
    
    duration = (time.time() -start)/3600
    print(duration)

