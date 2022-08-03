import cv2
import imutils
import numpy as np
import argparse
#frame
def detect(frame):
    bounding_box_cordinates, weights =  HOGCV.detectMultiScale(frame, winStride = (1, 1), padding = (0, 0), scale = 1.1)
    #The win-stride is the step size in the x and y direction of our sliding window.
    #The padding switch controls the amount of pixels is padded with prior to HOG feature vector extraction and SVM classification.
    #To control the scale of the image pyramid (allowing us to detect people in images at multiple scales), we can use the --scale argument.

    
    person = 1
    for x,y,w,h in bounding_box_cordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        #Draw the bounding boxes on the image
        cv2.putText(frame, f'person {person}', (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        person += 1
    
    cv2.putText(frame, 'Status : Detecting ', (20,30), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0), 1)
    cv2.putText(frame, f'Total Persons : {person-1}', (20,50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0), 1)
    cv2.imshow('output', frame)
    #Display the resulting image

    return frame

    return frame

def detectByPathVideo(path, writer):

    video = cv2.VideoCapture(path)
    check, frame = video.read()
    if check == False:
        print('Video Not Found. Please Enter a Valid Path (Full path of Video Should be Provided).')
        return

    print('Detecting people...')
    while video.isOpened():
        #check is True if reading was successful 
        check, frame =  video.read()

        if check:
            frame = imutils.resize(frame , width=min(800,frame.shape[1]))
            #Resizing the frame to have a maximum width of 800 pixels
            frame = detect(frame)
            
            if writer is not None:
                writer.write(frame)
            
            key = cv2.waitKey(1)
            #allows users to display a window for given milliseconds or until any key is pressed.
            if key== ord('q'):
                break
        else:
            break
    video.release()
    #release method of the writer to ensure that the output video file pointer is released.
    cv2.destroyAllWindows()
    #destroys all windows


def detectByPathImage(path, output_path):
    image = cv2.imread(path)
    #loads an image from the specified file

    image = imutils.resize(image, width = min(400, image.shape[1])) 

    result_image = detect(image)

    if output_path is not None:
        cv2.imwrite(output_path, result_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def humanDetector(args):
    image_path = args["image"]
    video_path = args['video']

    writer = None

    if video_path is not None:
        print('[INFO] Opening Video from path.')
        detectByPathVideo(video_path, writer)
    elif image_path is not None:
        print('[INFO] Opening Image from path.')
        detectByPathImage(image_path, args['output'])

# construct the argument parse and parsing the command line arguments for flexibility and reuseability
#{'video': None, 'image': '1_233.jpg', 'camera': False, 'output': None}
def argsParser():
    arg_parse = argparse.ArgumentParser() #creating the parser
    arg_parse.add_argument("-v", "--video", default=None, help="path to Video File ")# Add an argument
    arg_parse.add_argument("-i", "--image", default=None, help="path to Image File ")# Add an argument
    arg_parse.add_argument("-o", "--output", type=str, help="path to optional output video file")# Add an argument
    args = vars(arg_parse.parse_args())# Parse the argument
    print(args)
    return args

#main module of the system
if __name__ == "__main__":
    HOGCV = cv2.HOGDescriptor()   #initialize the HOG descriptor
    HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    #Daimler Pedestrian Detection Benchmark Dataset
    #sets the Support Vector Machine detector to be the default human detector
#cv2.HOGDescriptor_getDefaultPeopleDetector()
#Returns coefficients of the classifier trained for people detection (for 64x128 windows).

    args = argsParser() #for parsing the command line arguments
    humanDetector(args) #the output dictionary arge is passed to human detector method.

