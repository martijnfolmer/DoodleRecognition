import cv2
import numpy as np

from tensorflow import lite

"""
    This class can be used to take an input image which consists of a white paper with some doodles on it, and
    turn this into a prompt that can be used as input for a generative image model.
    
    In order to run this, first models have to be trained using the TrainDoodleRecognition.py file. It uses a bagging
    method with these models, meaning that multiple models can vote on the classification of a model.
    
    Author :        Martijn Folmer 
    Date created :  26-03-2023

"""


class RunDoodleRecognition:
    def __init__(self, array_with_tflitePaths, category_list):
        # initialize the tflites and get the index-to-category list we saved during training
        self.all_interpreters = self.initialize_all_interpreters(array_with_tflitePaths)
        self.cat_list = category_list

    def initialize_all_interpreters(self, path_to_tlfites):
        """
        Initialize the machine learning model (saved as .tflite) and return all information needed for running

        :param path_to_tlfites: An array with all the paths to the trained .tflite models
        :return: an array containing all of the interpreters (for invocation), the input/output details of the model
        and the input size of the image it expects
        """
        all_interpreters = []
        for tflitePath in path_to_tlfites:
            interpreter, input_details, output_details, input_size = self.read_interpreter(tflitePath)
            all_interpreters.append([interpreter, input_details, output_details, input_size])
        return all_interpreters

    def read_interpreter(self, path_to_interpreter):
        """
        Initializes a single .tflite model and returns its information

        :param path_to_interpreter: The path to where we saved the .tflite model
        :return: the interpreter (for invocation), the input/output details of the model
        and the input size of the image it expects
        """
        # initialize the interpreter
        interpreter = lite.Interpreter(model_path=path_to_interpreter)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test the model on random input data.
        input_shape = input_details[0]['shape']

        return interpreter, input_details, output_details, input_shape

    def run_interpreter(self, _img, _interpreter, _input_shape, _input_details, _output_details):
        """
        Running a single image through an interpreter, and returning the output

        :param _img: The image we wish to run through the interpreter
        :param _interpreter: The interpreter that was initialized on one of our .tflite files
        :param _input_shape: The input size of the image that the model expects
        :param _input_details: All information about how to input the image into the interpreter
        :param _output_details: All information about how to retreive the results from the interpreter
        :return: The index of the category that was detected
        """
        # preprocess the image for input, turn it single channel grayscale
        _img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)

        # Threshold the image to make it more clear
        _, _img = cv2.threshold(_img, 127, 255, cv2.THRESH_BINARY)

        # Turn it into the right shape and type
        _img = cv2.resize(_img, (_input_shape[1], _input_shape[2]))
        _img = np.reshape(_img, _input_shape)
        _img = _img.astype(dtype=np.float32)

        # run the interpreter
        _interpreter.set_tensor(_input_details[0]['index'], _img)
        _interpreter.invoke()

        # postprocess our output
        outp_cat = _interpreter.get_tensor(_output_details[0]['index'])[0]
        outp_cat = np.argmax(outp_cat, axis=0)

        # return the category index
        return outp_cat


    def findAllIndividualDoodles(self, img):

        """
        This function is used to find all of the doodles in an image

        :param img: The image, which is usually a white sheet with some doodles on it
        :return: An array with the sepparate doodles as well as an array with all bounding boxes of those doodles
        """

        # Turn it into grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply thresholding
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        bitnot = cv2.bitwise_not(thresh, thresh)

        # Dilation (this must be varied based on the size of the image)
        for i in range(10):
            bitnot = cv2.dilate(bitnot, (3, 3))

        # Find contours
        contours, _ = cv2.findContours(bitnot, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Crop the doodles and pass them on along with the boundingboxes
        all_doodles = []    # all doodles sepparated from eachother
        all_bb = []         # all boundingboxes
        for i, contour in enumerate(contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Crop doodle and append to the all_doodles list
            all_doodles.append(img[y:y + h, x:x + w])
            all_bb.append([x, y, x+w, y+h])     # append the bounding boxes [x1, y1, x2, y2]

        return all_doodles, all_bb

    def turnDoodleSquare(self, doodle):
        """
        The interpreters expect a square image, so we must pad the sides of the doodles to make them square

        :param doodle: The image of the doodle we cropped out
        :return: A square version of the doodle
        """

        # Get the dimensions of the image
        height, width, _ = doodle.shape

        # Determine the maximum dimension
        max_dim = max(width, height)

        # Create a new image with white color
        padded_image = np.ones((max_dim, max_dim, 3), dtype=np.uint8) * 255

        # Paste the original image at the center of the new image
        x_offset = (max_dim - width) // 2
        y_offset = (max_dim - height) // 2
        padded_image[y_offset:y_offset + height, x_offset:x_offset + width] = np.array(doodle)

        return padded_image

    def padOutsideImg(self, doodle, frac_padding):
        """
        If the doodle is too close to the edges, we can use this function to create a padding on the outside of it

        :param doodle: The image we wish to pad (assumed to be square)
        :param frac_padding: The size of the padding (=frac * width of image)
        :return: A padded doodle
        """
        height, width, _ = doodle.shape
        padding_size = int(width * frac_padding)

        # Create a new array with white padding
        padded_img = np.full((height + 2 * padding_size, width + 2 * padding_size, 3), 255, dtype=np.uint8)

        # Copy the original image into the padded array
        padded_img[padding_size:-padding_size, padding_size:-padding_size, :] = doodle
        return padded_img

    def runRecognitionDoodle(self, doodle):
        """
        Run the classification models on a single instance of a doodle

        :param doodle: The image of the doodle we want to run classification on
        :return: The index of the category we think this doodle is, and the name of the category
        """
        # turn doodle square and pad outside a little
        doodle = self.turnDoodleSquare(doodle)

        # run on the interpreters (bagging method, have multiple interpreters vote on it)
        output_all = []
        for interpreter_c, input_details_c, output_details_c, input_shape_c in self.all_interpreters:
            output_category = self.run_interpreter(doodle, _interpreter=interpreter_c, _input_shape=input_shape_c,
                                              _input_details=input_details_c, _output_details=output_details_c)
            output_all.append(output_category)

        # Use bagging to find the best output
        counts = np.bincount(np.asarray(output_all))
        output_avg = np.argmax(counts)
        return output_avg, self.cat_list[output_avg]

    def runImage(self, img):
        """
        Used to find all the doodles in an image, and classify them

        :param img: The image we want to search for doodles on
        :return: all indexes of categories, names of categories and bounding boxes of the detected doodles
        """
        all_doodles, all_bb = self.findAllIndividualDoodles(img)
        # Run the recognition on each of the doodles.

        all_cat, all_name = [], []
        for doodle in all_doodles:
            cat_cur, name_cur = self.runRecognitionDoodle(doodle)
            all_cat.append(cat_cur)
            all_name.append(name_cur)

        return all_cat, all_name, all_bb

    def calculate_1D_IOU(self, x1, x2, x3, x4):
        """
        Used to find out how much two lines overlap, when looking at only 1 sort of coordinate (x or y or z) This is
        used to determine whether two images are on the same horizontal or vertical line

        :param x1: The x-or-y coordinate of one end of Line 1
        :param x2: The x-or-y coordinate of the other end of Line 1
        :param x3: The x-or-y coordinate of one end of Line 2
        :param x4: The x-or-y coordinate of the other end of Line 2
        :return: The Intersection over union.
        """
        len1 = x2-x1
        len2 = x4-x3
        overlap = max(0, min(x4,x2) - max(x1,x3))

        try:
            ious = overlap/(len1+len2-overlap)
            return ious
        except:
            return 0

    def turnDetectedNamesToPrompt(self, all_bb, all_names):
        """
        Turn all of the detected doodle names into a prompt we can feed into an image generator.

        :param all_bb: All the bounding boxes in which we detected doodles
        :param all_names: All the names belonging to those doodles
        :return: The text prompt
        """

        # special case : 0 or 1 doodle detected
        if len(all_bb) == 0:
            return ""
        if len(all_bb) == 1:
            prompt = "a " + all_names[0]
            return prompt

        # If we have multiple doodles, find out which bounding boxes are on the same line
        same_line_order = []
        for name, bb in zip(all_names, all_bb):
            if len(same_line_order)==0:
                same_line_order.append([bb, name])
            else:
                # find out how much they overlap
                bby1 = bb[1]
                bby2 = bb[3]
                same_line = False
                for i_bbch, bbch in enumerate(same_line_order):
                    bby3, bby4 = bbch[0][1], bbch[0][3]
                    IOU_1D = self.calculate_1D_IOU(bby1,bby2,bby3,bby4)
                    if IOU_1D > 0.5:  # more than 50 % overlap, so it is on the same horizontal line
                        bbch.extend([bb, name])
                        same_line_order[i_bbch] = bbch
                        same_line = True
                        break
                if not same_line: # it isn't on any of the same horizontal lines, so we append a new line
                    same_line_order.append([bb, name])

        # get the lines, and make the prompt
        prompt = ""
        for i_line, line in enumerate(same_line_order):

            for i in range(0, len(line),2):
                prompt += "a "+line[i+1]
                if i<len(line)-2:
                    prompt += " and "
                else:
                    prompt += " "

            if i_line<len(same_line_order)-1:

                y1 = line[0][1]
                y2 = same_line_order[i_line+1][0][1]
                if y1 > y2:
                    prompt += "below "
                else:
                    prompt += "above "

        return prompt


if __name__ == "__main__":

    # example usage.

    # The list of tflites we use for our Bagging method
    paths_to_tflites = ['trained_tflites/final_model_100x100_1.tflite', 'trained_tflites/final_model_100x100_2.tflite',
                        'trained_tflites/final_model_100x100_3.tflite']
    # The category list we use to translate the output of the model (an index) to a name of the category
    list_with_categories = np.load('category_to_name.npy')
    # Initialize the class
    RDR = RunDoodleRecognition(paths_to_tflites, list_with_categories)

    # Read our test image, run it, and turn it into a prompt
    path_to_test_img = 'test_image.png'
    img = cv2.imread(path_to_test_img)
    all_cat, all_names, all_bb = RDR.runImage(img)
    prompt = RDR.turnDetectedNamesToPrompt(all_bb, all_names)

    # visualise bb, names and prompts
    print(f"\nThe prompt is : \n    {prompt}")
    for bb, name, in zip(all_bb, all_names):
        x1, y1, x2, y2 = bb
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        img = cv2.putText(img, str(name), (x1+10, y1+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # show the images
    cv2.imshow("test_image", img)
    cv2.waitKey(-1)


