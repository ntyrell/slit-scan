import io
import time
import threading
import numpy as np
import cv2
import argparse
from flask import Response
from flask import Flask
from flask import render_template
from socket import *

# create UDP socket
serverSocket = socket(AF_INET, SOCK_DGRAM)
serverSocket.bind(('', 12000))

lock = threading.Lock()
app = Flask(__name__)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")


#def get_frame():
	# get the frame from the camera pi, which will be sent over udp socket
	# save to global variable called data, or something
	# might be better to send the frame already encoded as a jpeg?
	# this thead will have to execute separate from generate?

def generate():
	global lock
	while True:
		with lock:
			encodedImage,client = serverSocket.recvfrom(2**19) # receive encoded image as bytes... not sure how many bytes
		
		# yield the output frame in the byte format
		yield(bytes(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + encodedImage + b'\r\n'))

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


	#t = threading.Thread(target=generate_output_frame, args=(
	#	args.output,))
	#t.daemon = True
	#t.start() 	
	
	# start the flask app
	app.run(host='192.168.4.1', port = '8000', debug=True,
			threaded=True, use_reloader=False)
        
