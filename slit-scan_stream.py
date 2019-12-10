import io
import time
import threading
import picamera
from PIL import Image
import numpy as np
from Queue import Queue
import cv2
import argparse
from flask import Response
from flask import Flask
from flask import render_template


class SlitScanParams(object):
    def __init__(self):
        self.fps = 60
        self.fps_a = 15
        self.n = 30 # buffer size
        self.qs = [Queue(maxsize=self.n), Queue(maxsize=self.n)]
        self.q_index = 0
        self.n_threads = 16
        self.dic = dict()
        self.dic_prev = dict()
        self.i_prev = -1
        self.index = -1
        self.frame_count = -1
        self.drop_count = -1
        self.qf = Queue(maxsize = int(self.n / (self.fps/self.fps_a) + 10))
        self.data = np.zeros([480,640,3],dtype=np.uint8)
        self.lock = threading.Lock()
        
p = SlitScanParams()
app = Flask(__name__)

        
class ImageProcessor(threading.Thread):
    def __init__(self, owner, params):
        super(ImageProcessor, self).__init__()
        self.stream = io.BytesIO()
        self.event = threading.Event()
        self.terminated = False
        self.owner = owner
        self.start()
        self.frame_count = -1;
        self.params = params # SlitScanParams object

    def set_frame_count(self, f):
        self.frame_count = f
    
    def run(self):
        # This method runs in a separate thread
        while not self.terminated:
            # Wait for an image to be written to the stream
            if self.event.wait(1):
                try:
                    self.stream.seek(0)
                    # Read the image and do some processing on it
                    im = Image.open(self.stream)
                    data = np.asarray(im)
                    if self.params.qs[self.params.q_index].full():
                        self.params.q_index = ~self.params.q_index
                        t1 = threading.Thread(target=make_frames, args=(self.params,))
                        t1.start() # start the frame ordering thread..
                        # there may be conflicts between threads setting the queue flag
                        # consider putting a timestamp on the queue and using it to order the frames during update
                    self.params.qs[self.params.q_index].put((self.frame_count,data[:,320]))
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


class ProcessOutput(object):
    def __init__(self, params):
        self.done = False
        # Construct a pool of 8 image processors along with a lock
        # to control access between threads
        self.lock = threading.Lock()
        self.pool = [ImageProcessor(self,params) for i in range(params.n_threads)]
        self.processor = None
        self.params = params # SlitScanParams object
        
    def write(self, buf):
        if buf.startswith(b'\xff\xd8'):
            self.params.frame_count += 1
            # New frame; set the current processor going and grab
            # a spare one
            if self.processor:
                self.processor.set_frame_count(self.params.frame_count)
                self.processor.event.set()
            else:
                self.params.drop_count += 1
                print('NO PROCESSORS AVAILABLE; dropped frame: ' 
                      + str(self.params.frame_count)
                      + '; drop count: ' + str(self.params.drop_count))
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

            
def make_frames(params):
    params.dic = dict()
    count = 0
        
    # could I put the data into a dictionary directly without the queue? can it handle threading?
    while not params.qs[~params.q_index].empty():
        q_data = params.qs[~params.q_index].get()
        params.dic[q_data[0]] = q_data[1]
        if count == params.n/2:
            params.index = q_data[0]
        count += 1
       
    #print('index: ' + str(index))
        
    #big_dic = {**dic, **dic_prev} # doesn't work with python 2
    params.big_dic = params.dic.copy()
    params.big_dic.update(params.dic_prev)
    dn = 0
    data_temp = np.zeros([480,0,3],dtype=np.uint8)
    # sort data_temp and data_temp_prev according to frame counts 
    for f,s in sorted(params.big_dic.items()):
        if f > params.i_prev and f <= params.index:
            data_temp = np.insert(data_temp,0,s,axis=1)
            dn += 1
            if dn % int(params.fps/params.fps_a) == 0:
                if not params.qf.full():
                    params.qf.put(data_temp)
                else:
	            	print('FRAME QUEUE FILLED!')

                data_temp = np.zeros([480,0,3],dtype=np.uint8) 
                if params.index-dn < 3:
		    params.index = f
		    break
    params.i_prev = params.index
    params.dic_prev = params.dic
    

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def generate_output_frame(output_flag, params):
	# add an argument that toggles video saving, instead of global variable
	with picamera.PiCamera(resolution='VGA', framerate=params.fps,sensor_mode=7) as camera:
		output = ProcessOutput(params)
		camera.start_recording(output, format='mjpeg')
		s = 0
		f = 0
		if output_flag:
			fourcc = cv2.cv.CV_FOURCC('D','I','V','X')
			out = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))
		while True:
		 if not params.qf.empty():
			#print('frame queue size: ' + str(qf.qsize()))
			
			data_temp = params.qf.get()

			dn = data_temp.shape[1]
			with params.lock:
				params.data= np.roll(data,dn,axis=1)
				params.data[:,0:dn]= data_temp
			if output_flag: out.write(params.data)
			f = time.time()
			wait_time = 1000/params.fps_a - int(1000*(f-s))
			if wait_time < 1 or wait_time > 1000/params.fps_a:
				wait_time = 1
			
			c = cv2.waitKey(wait_time)
			if c == ord('q'):
				break
			s = time.time()
		if output_flag: out.release()    
    
def generate():
	while True:
		with p.lock:
			# encode the output frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg",p.data)
			#ensure the frame was successfully encoded
			if not flag: continue
		
		# yield the output frame in the byte format
		yield(bytes(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n'))

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(), 
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# do the main thread check thing, and then pass the agruments into the generate frame thread or something
# maybe make camera fps, animation fps, buffer size, and number of threads command line arguments too, will have to be smart about queue variable..


# check to see if this is the main thread of execution
if __name__ == '__main__':
	# initiate the parser
	parser = argparse.ArgumentParser()
	parser.add_argument("-O", "--output", help="save output as video file", action="store_true")

	# read arguments from the command line
	args = parser.parse_args()


	t = threading.Thread(target=generate_output_frame, args=(
		args.output,p))
	t.daemon = True
	t.start() 	
	
	# start the flask app
	app.run(host='192.168.29.198', port = '8000', debug=True,
			threaded=True, use_reloader=False)
        
