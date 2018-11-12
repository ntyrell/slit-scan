import io
import time
import threading
import picamera
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from queue import Queue
from matplotlib.animation import FuncAnimation

n = 90
qs = [Queue(maxsize=n), Queue(maxsize=n)]
q_index = 0
q_flag = 0

class ImageProcessor(threading.Thread):
    def __init__(self, owner):
        super(ImageProcessor, self).__init__()
        self.stream = io.BytesIO()
        self.event = threading.Event()
        self.terminated = False
        self.owner = owner
        self.start()
        self.frame_count = -1;

    def set_frame_count(self, f):
        self.frame_count = f
    
    def run(self):
        global q_flag
        global q_index
        global qs
        # This method runs in a separate thread
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.event.wait(1):
                try:
                    self.stream.seek(0)
                    # Read the image and do some processing on it
                    im = Image.open(self.stream)
                    data = np.asarray(im)
                    if qs[q_index].full():
                        q_index = ~q_index
                        q_flag = 1
                        # there may be conflicts between threads setting the queue flag
                        # consider putting a timestamp on the queue and using it to order the frames during update
                    qs[q_index].put((self.frame_count,data[:,320]))
                    # print('frame count: '+ str(self.frame_count))
                    #...
                    #...
                    # Set done to True if you want the script to terminate
                    # at some point
                    #self.owner.done=True
                finally:
                    # Reset the stream and event
                    self.stream.seek(0)
                    self.stream.truncate()
                    self.event.clear()
                    # Return ourselves to the available pool
                    with self.owner.lock:
                        self.owner.pool.append(self)

n_threads = 8
frame_count = -1
drop_count = -1

class ProcessOutput(object):
    def __init__(self):
        self.done = False
        # Construct a pool of 8 image processors along with a lock
        # to control access between threads
        self.lock = threading.Lock()
        self.pool = [ImageProcessor(self) for i in range(n_threads)]
        self.processor = None

    def write(self, buf):
        global frame_count
        global drop_count
        if buf.startswith(b'\xff\xd8'):
            frame_count += 1
            # New frame; set the current processor going and grab
            # a spare one
            if self.processor:
                self.processor.set_frame_count(frame_count)
                self.processor.event.set()
            else:
                drop_count += 1
                print('NO PROCESSORS AVAILABLE; dropped frame: ' 
                      + str(frame_count)
                      + '; drop count: ' + str(drop_count))
            with self.lock:
                if self.pool:
                    self.processor = self.pool.pop()
                else:
                    # No processor's available, we'll have to skip
                    # this frame; you may want to print a warning
                    # here to see whether you hit this case
                    self.processor = None       
        if self.processor:
            self.processor.stream.write(buf)

    def flush(self):
        # When told to flush (this indicates end of recording), shut
        # down in an orderly fashion. First, add the current processor
        # back to the pool
        if self.processor:
            with self.lock:
                self.pool.append(self.processor)
                self.processor = None
        # Now, empty the pool, joining each thread as we go
        while True:
            with self.lock:
                try:
                    proc = self.pool.pop()
                except IndexError:
                    pass # pool is empty
            proc.terminated = True
            proc.join()

data = np.zeros([480,640,3],dtype=np.uint8)
ax1 = plt.subplot(111)
im1=ax1.imshow(data)

n2 = int(n/2)

dic = dict()
dic_prev = dict()
i_prev = -1
index = -1

def update(i):
    global data
    global q_flag
    global dic
    global dic_prev
    global index
    global i_prev
    if q_flag:
        q_flag = 0
        dic = dict()
        count = 0
        
        # could I put the data into a dictionary directly without the queue? can it handle threading?
        while not qs[~q_index].empty():
            q_data = qs[~q_index].get()
            dic[q_data[0]] = q_data[1]
            if count == n/2:
                index = q_data[0]
            count += 1
        
        print('index: ' + str(index))
        
        big_dic = {**dic, **dic_prev}
        dn = 0
        data_temp = np.zeros([480,0,3],dtype=np.uint8)
        # sort data_temp and data_temp_prev according to frame counts 
        for f,s in sorted(big_dic.items()):
            if f > i_prev and f <= index:
                data_temp = np.insert(data_temp,0,s,axis=1)
                dn += 1
        i_prev = index
        dic_prev = dic
        data= np.roll(data,dn,axis=1)
        data[:,0:dn]= data_temp
        im1.set_data(data)

fps = 60

with picamera.PiCamera(resolution='VGA', framerate=fps,sensor_mode=7) as camera:
    output = ProcessOutput()
    camera.start_recording(output, format='mjpeg')
    # need to sample the animation faster than the queue filling rate
    ani=FuncAnimation(plt.gcf(), update, interval=1000/fps*n/4)
    plt.show()

