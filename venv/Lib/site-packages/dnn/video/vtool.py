import cv2
from rs4 import attrdict

def capture (path, interval = 10):
    count = 0
    vidcap = cv2.VideoCapture (path)
    latest = 0
    frames = []
    while (vidcap.isOpened ()):
        ret, image = vidcap.read ()
        frame_no =  int (vidcap.get(1))
        if latest == frame_no:
            break
        latest = frame_no         
        if frame_no % interval == 0:
            #print('Saved frame number : ' + str(int(vidcap.get(1))))
            frames.append (image)
            #cv2.imwrite("./images/frame%d.jpg" % count, image)
            #print('Saved frame%d.jpg' % count)
            count += 1
    vidcap.release ()
    return frames

def get_info (path):
    vidcap = cv2.VideoCapture (path)
    d = attrdict.AttrDict ()
    d.frame_rate = vidcap.get (cv2.CAP_PROP_FPS)
    d.frame_width = vidcap.get (cv2.CAP_PROP_FRAME_WIDTH)
    d.frame_height = vidcap.get (cv2.CAP_PROP_FRAME_HEIGHT)
    d.video_length = vidcap.get (cv2.CAP_PROP_FRAME_COUNT)  / d.frame_rate    
    d.format = vidcap.get (cv2.CAP_PROP_FORMAT)    
    d.codec = vidcap.get (cv2.CAP_PROP_FOURCC )
    return d
