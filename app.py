# import flask
from flask import Flask, request, send_file, render_template, jsonify
import cv2
import numpy as np
# import matplotlib.pyplot as plt

import keras
from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
from urllib.request import urlopen
import os.path
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/prediction_iris', methods=['POST', 'GET'])
def prediction_iris():
    if request.method == 'POST':
        loaded_model = load('Iris.DecisionTree.joblib')
        classes = ["Setosa", "Versicolor", "Virginica"]
        X_new = ([[request.form['sep_len'],
                   request.form['sep_wid'],
                   request.form['pet_len'],
                   request.form['pet_wid']]])
        y_new = loaded_model.predict(X_new)
        return render_template('result.html', results=classes[y_new[0]])


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return image

def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return image

@app.route('/getmsg/', methods=['GET'])
def respond():
    # Retrieve the name from url parameter
    name = request.args.get("name", None)

    # For debugging
    print(f"got name {name}")

    response = {}

    # Check if user sent a name at all
    if not name:
        response["ERROR"] = "no name found, please send a name."
    # Check if the user entered a number not a name
    elif str(name).isdigit():
        response["ERROR"] = "name can't be numeric."
    # Now the user entered a valid name
    else:
        response["MESSAGE"] = f"Welcome {name} to our awesome platform!!"

    # Return the response in json format
    return jsonify(response)


# xoay anh
def Image_Alignment(file_chuan,img):
    # Convert to grayscale.
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(file_chuan, cv2.COLOR_BGR2GRAY)
    height, width = img2.shape

    # Create ORB detector with 5000 features.
    orb_detector = cv2.ORB_create(7400)

    # (which is not reqiured in this case).
    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # Hamming distance as measurement mode.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match the two sets of descriptors.
    matches = matcher.match(d1, d2)

    # Sort matches on the basis of their Hamming distance.
    matches.sort(key=lambda x: x.distance)

    # Take the top 90 % matches forward.
    matches = matches[:int(len(matches) * 90)]
    no_of_matches = len(matches)

    # Define empty matrices of shape no_of_matches * 2.
    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Find the homography matrix.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # colored image wrt the reference image.
    transformed_img = cv2.warpPerspective(img, homography, (width, height))

    return (transformed_img)

def row_line(im_th):
    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    return im_floodfill_inv
    
@app.route('/data', methods=['POST'])
def detect(transformed_img):
    # load model
    with CustomObjectScope({'relu6': keras.layers.ReLU(6.), 'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
        model = load_model("./models/weights.02.h5")

    img_resize = cv2.resize(transformed_img, (1654, 1166), interpolation=cv2.INTER_AREA)
    img_convert = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

    # Set values below 128 to 255.
    th, im_th = cv2.threshold(img_convert, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Đảo ngược ảnh
    img_bin = 255 - row_line(im_th)

    # Độ dài của kernel
    kernel_length = np.array(img_bin).shape[1] // 200

    # A verticle kernel of (1 X kernel_0length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))

    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

    # Tìm các đường thẳng nằm dọc
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=1)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=1)

    # Tìm các đường thẳng nằm ngang
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=1)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=1)

    # Tham số để quyết định số lượng hình ảnh kết hợp tạp thành ô vuông
    alpha = 0.5
    beta = 1 - alpha

    # Kết hợp tạo các ô vuông
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0)
    img_final_bin = cv2.erode(img_final_bin, kernel, iterations=1)

    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # cv2.imshow("anh_1", verticle_kernel)
    # cv2.imshow("anh_2", verticle_lines_img)
    # cv2.imshow("anh_3", horizontal_lines_img)
    # cv2.imshow("anh_4", img_final_bin)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    # # cv2.imwrite("img_final_bin.jpg", img_final_bin)
    # #
    # Áp dụng phương pháp tìm biên sẽ phát hiện được các hình vuông và kết hợp lại để tạo hình chữ nhật lớn
    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sắp xếp các hình chữ nhật theo thứ tự từ trên xuống
    def sort_contours(cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0
        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = False
        # handle if we are sorting against the y-coordinate rather than

        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        # construct the list of bounding boxes and sort them from top to

        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]

        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)

    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

    idx = 0
    tmp_anh = []
    tmp_ket_qua = []
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)
        # print(w)
        # print(h)
        # print("het !!!!")
        if (200 < w < 700 and 47 < h < 73 or w > 800 and 47 < h < 73):
            # if (200 < w and  h < 90):
            idx += 1
            new_img = img_resize[y:y + h, x:x + w]
            new_img_convert = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
            tmp_anh.append(new_img_convert)

    nguong_pixel = 128
    for h in range(len(tmp_anh)):
        sum_index = 0
        tmp_sum = []
        for i in tmp_anh[h]:
            for j in i:
                if (j < nguong_pixel):
                    sum_index += 1
                else:
                    continue
        tmp_sum.append(sum_index)
        map_name = ["Họ tên: ", "Số zalo: ", "Số điện thoai: ", "Tên hiển thị trên Facebook: ", "Email: ",
                    "Học sinh THPT lớp: ", "Sinh viên cao đẳng/đại học trường: ", "Khác: ", "Ngành_Khác: "]
        
        for k in tmp_sum:
            # print(k)
            if (k < 10000):
                ngat_dong = "hết một thông tin!!!"
                tmp_ket_qua.append(ngat_dong)
                tmp_ket_qua.append(str(map_name[h]))
                img_resize = cv2.resize(tmp_anh[h], (900, 100), interpolation=cv2.INTER_AREA)

                # Threshold.
                th, im_th = cv2.threshold(img_resize, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                image = np.invert(im_th)

                # Invert the image
                img_bin = 255 - image

                # Defining a kernel length
                kernel_length = np.array(img_bin).shape[1] // 150
                # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
                verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
                # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
                hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
                # A kernel of (3 X 3) ones.
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

                # Morphological operation to detect vertical lines from an image
                img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=1)
                verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=1)

                # Morphological operation to detect horizontal lines from an image
                img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=1)
                horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=1)

                # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
                alpha = 0.5
                beta = 1 - alpha

                # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
                img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0)
                img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=1)

                (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255,
                                                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                # cv2.imshow("anh_1",verticle_lines_img)
                # cv2.imshow("anh_2",horizontal_lines_img)
                # cv2.imshow("anh_3",img_final_bin)
                # cv2.waitKey()
                # cv2.destroyAllWindows()

                # Find contours for image, which will detect all the boxes
                contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # Sort all the contours by top to bottom.
                def sort_contours(cnts, method="left-to-right"):
                    # initialize the reverse flag and sort index
                    reverse = False
                    i = 0
                    # handle if we need to sort in reverse
                    if method == "right-to-left" or method == "bottom-to-top":
                        reverse = False
                    # handle if we are sorting against the y-coordinate rather than

                    # the x-coordinate of the bounding box
                    if method == "top-to-bottom" or method == "bottom-to-top":
                        i = 1
                    # construct the list of bounding boxes and sort them from top to

                    # bottom
                    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

                    (cnts, boundingBoxes) = zip(
                        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

                    # return the list of sorted contours and bounding boxes
                    return (cnts, boundingBoxes)

                (contours, boundingBoxes) = sort_contours(contours, method="left-to-right")

                idx = 0
                for c in contours:
                    # Returns the location and width, height for every contour
                    x, y, w, h = cv2.boundingRect(c)
                    # print(w)
                    # print(h)
                    # print("het!!!!")
                    if (24 < w < 110 and 66 < h < 90):
                        idx += 1
                        # img_cut = img_resize[y:y+h,x:x+w]
                        new_img = img_resize[y + 4:(y - 1) + h, x + 1:(x - 2) + w]

                        # plt.imshow(new_img)
                        # plt.show()
                        # y+3 dich chuyen len:(y-5) dich chuyen xuong
                        # x-1 di chuyen qua phai0
                        img_color = cv2.cvtColor(new_img, 1)
                        img_resize_lan_1 = cv2.resize(img_color, (128, 128),
                                                      interpolation=cv2.INTER_AREA)  # chuyển kích thước ảnh về 128x128
                        cropped_image = img_resize_lan_1[10:128, 10:128]

                        img_resize_lan_2 = cv2.resize(cropped_image, (128, 128),
                                                      interpolation=cv2.INTER_AREA)  # chuyển kích thước ảnh về 128x128
                        img = cv2.medianBlur(img_resize_lan_2, 9)

                        thresh = 200
                        # (thresh, img_bin) = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                        img_agray = cv2.threshold(img, thresh, maxval=255, type=cv2.THRESH_BINARY_INV)[
                            1]  # chuyển ảnh về dạng trắng đen
                        image = np.invert(img_agray)
                        img_color = 255 - image
                        (thresh, img_bin) = cv2.threshold(img_color, 128, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)

                        img_color = cv2.cvtColor(img_bin, cv2.COLOR_RGBA2BGR)

                        names = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
                                 10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I",
                                 19: "J",
                                 20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R", 28: "S",
                                 29: "T", 30: "U",
                                 31: "V", 32: 'W', 33: "X", 34: "Y", 35: "Z"}

                        # dự đoán
                        prediction = model.predict(img_color.reshape(-1, 128, 128, 3))

                        # # # in ra mãng các giá trị dự đoán
                        # print(prediction)
                        #
                        # # lấy phần tử có giá trị lớn nhất
                        predict_img = np.argmax(prediction, axis=-1)
                        tmp_ket_qua.append(names.get(predict_img[0]))
                        
            else:
                continue
                
        
    
    return (tmp_ket_qua)
    
@app.route('/data', methods=['GET'])  
def write_img(tmp_ket_qua):

    with open('data/text.txt',"w+",encoding='UTF-8') as f:
        couter = 1
        f.write("\r")
        f.write("anh: %s " % (str(couter)))
        for i in tmp_ket_qua:

            if (str(i) == "hết một thông tin!!!"):
                f.write("\r")
            else:
                f.write("%s" % (str(i)))
                    
            couter += 1
    
    return 0

# http://localhost:5000/prediction?url=http://trolyao.cusc.vn/image/m.jpg


# @app.route('/prediction')
@app.route('/prediction', methods=['GET'])
def predition():
    # url = 'http://trolyao.cusc.vn/image/m.jpg'
    url = ''
    if request.method == 'GET':
        url = request.args.get("url", None)

    # img = cv2.imread('1.jpg', cv2.IMREAD_COLOR) # đọc ảnh
    # img = url_to_image("http://trolyao.cusc.vn/image/m.jpg");
    # img = url_to_image("http://trolyao.cusc.vn/image/i.jpg");
    # img = url_to_image("http://trolyao.cusc.vn/image/g.jpg");
    img = url_to_image(url);

    return jsonify(detect(img))


# return jsonify(keras.__version__)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    file_chuan = request.files['image_chuan']
    # Save file
    # filename = 'static/' + file.filename
    # file.save(filename)

    # Read image
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), 1)
    file_chuan = cv2.imdecode(np.fromstring(file_chuan.read(), np.uint8), 1)
    return render_template('result.html', results=detect(Image_Alignment(file_chuan,img)))

@app.route('/download')
def download_file():
	path = "data/text.txt"
 
	return send_file(path, as_attachment=True,conditional=True)

if __name__ == '__main__':

    #app.run(debug=True, host="0.0.0.0", port='5000')
    app.run(threaded=False)
    #print(predition(img))
