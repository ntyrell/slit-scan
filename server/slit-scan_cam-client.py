import io
import time
import threading
import picamera
from PIL import Image
import numpy as np
from Queue import Queue
import cv2
import argparse
from socket import *

# create UDP socket (that's what DGRAM is for)
clientSocket = socket(AF_INET, SOCK_DGRAM)
clientSocket.setblocking(0)
addr = ("127.0.0.1", 12000) # this is local, but could be on network

# initiate the parse
parser = argparse.ArgumentParser()
parser.add_argument("-O", "--output", help="save output as video file", action="store_true")
parser.add_argument("-v", "--video", help="display output with OpenCV", action="store_true")
parser.add_argument("-s", "--stream", help="stream camera output over udp socket", action="store_true")

# read arguments from the command line
args = parser.parse_args()

fps = 60 # camera frames per second
fps_a = 10 # desired update rate of animation, must be an integer divisor of camera fps

n = 50 # buffer size
qs = [Queue(maxsize=n), Queue(maxsize=n)]
q_index = 0

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
                        t1 = threading.Thread(target=make_frames)
                        t1.start() # start the frame ordering thread..
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

n_threads = 16
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

n2 = int(n/2)
dic = dict()
dic_prev = dict()
i_prev = -1
index = -1
qf = Queue(maxsize = int(n / (fps/fps_a) + 10))
def make_frames():
    global data
    global dic
    global dic_prev
    global index
    global i_prev
    dic = dict()
    count = 0
        
    # could I put the data into a dictionary directly without the queue? can it handle threading?
    while not qs[~q_index].empty():
        q_data = qs[~q_index].get()
        dic[q_data[0]] = q_data[1]
        if count == n/2:
            index = q_data[0]
        count += 1
       
    #print('index: ' + str(index))
        
    #big_dic = {**dic, **dic_prev} # doesn't work with python 2
    big_dic = dic.copy()
    big_dic.update(dic_prev)
    dn = 0
    data_temp = np.zeros([480,0,3],dtype=np.uint8)
    # sort data_temp and data_temp_prev according to frame counts 
    for f,s in sorted(big_dic.items()):
        if f > i_prev and f <= index:
            data_temp = np.insert(data_temp,0,s,axis=1)
            dn += 1
            if dn % int(fps/fps_a) == 0:
                if not qf.full():
                    qf.put(data_temp)
                else:
	            	print('FRAME QUEUE FILLED!')

                data_temp = np.zeros([480,0,3],dtype=np.uint8) 
                if index-dn < 3:
					index = f
					break
    i_prev = index
    dic_prev = dic	

data = np.zeros([480,850,3],dtype=np.uint8)

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50] # 95 is too high, 50 works

with picamera.PiCamera(resolution='VGA', framerate=fps,sensor_mode=7) as camera:
    output = ProcessOutput()
    camera.start_recording(output, format='mjpeg')
    s = 0
    f = 0
    
    if args.output:
		fourcc = cv2.cv.CV_FOURCC('D','I','V','X')
		out = cv2.VideoWriter('output.avi',fourcc,20.0,(850,480))
    while True:
     if not qf.empty():
        print('frame queue size: ' + str(qf.qsize()))
        
        data_temp = qf.get()

        dn = data_temp.shape[1]
        data= np.roll(data,dn,axis=1)
        data[:,0:dn]= data_temp
        if args.output: out.write(data)
        if args.video: cv2.imshow('slit-scan',data)
        # send data as packet over server. maybe encode as jpeg
        if args.stream:
			# encode the output frame in JPEG format
			#print('start encoding!')
			(flag, encodedImage) = cv2.imencode(".jpg",data[:,:,::-1], encode_param)
			#print('finish encoding!')
			#ensure the frame was successfully encoded
			#if flag: print('image not encoded!')
			message = bytearray(encodedImage)# send as jpeg since the raw frame size is almost 1mb
			#print('start sending packet!')
			clientSocket.sendto(message, addr) # this will block once the send buffer fills up.... seems to be happening
			#print('finish sending packet!')
        f = time.time()
        wait_time = 1000/fps_a - int(1000*(f-s))
        if wait_time < 1 or wait_time > 1000/fps_a:
			wait_time = 1
        
        c = cv2.waitKey(wait_time)
        if c == ord('q'):
			break
        s = time.time()
    if args.output: out.release()
    cv2.destroyAllWindows()        
     		
        
