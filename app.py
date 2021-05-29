from flask import Flask, request, send_file, render_template, json
import requests
import cv2
import numpy as np
import webbrowser
# from flask_ngrok import run_with_ngrok
import tensorflow
from keras.models import load_model
from pyimagesearch import imutils


app = Flask(__name__, static_url_path='/static')
# run_with_ngrok(app)
MODEL_PATH = 'models/your_model.h5'
tensorflow.keras.backend.clear_session()

# load model
model = load_model("./models/weights.36.h5")
print('Model loaded. Start serving...')


# Hàm xoay anh
def Image_Alignment(file_chuan, img):
    # chuyển ảnh về ảnh mức xám
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img2 = file_chuan
    height, width = img2.shape

    # khởi tạo 7400 điểm ngẩu nhiên
    orb_detector = cv2.ORB_create(7400)

    kp1, d1 = orb_detector.detectAndCompute(img1, None)
    kp2, d2 = orb_detector.detectAndCompute(img2, None)

    # tính khoảng cách hamming giữa 2 điểm ảnh.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Tính khoảng cách giũa 2 điểm
    matches = matcher.match(d1, d2)

    # sắp xếp dựa trên khoảng cách Hamming.
    matches.sort(key=lambda x: x.distance)

    # lấy ra tốp những điểm tốt nhất
    matches = matches[:int(len(matches) * 90)]
    no_of_matches = len(matches)

    p1 = np.zeros((no_of_matches, 2))
    p2 = np.zeros((no_of_matches, 2))

    for i in range(len(matches)):
        p1[i, :] = kp1[matches[i].queryIdx].pt
        p2[i, :] = kp2[matches[i].trainIdx].pt

    # Tìm ma trận homography.
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

    # vẽ lại ảnh cần xoay theo tọa độ của ảnh chuẩn.
    transformed_img = cv2.warpPerspective(img, homography, (width, height))

    return (transformed_img)


def detect(transformed_img, model):

    RESCALED_HEIGHT = 1166
    RESCALED_WIDTH = 1654
    img_resize = imutils.resize(transformed_img, height=int(RESCALED_HEIGHT), width=int(RESCALED_WIDTH))
    # Chuyển ảnh về ảnh xám
    img_convert = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)

    # phân ngưỡng nhị phân
    th, img_bin = cv2.threshold(img_convert, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Độ lớn của kernel
    kernel_length = np.array(img_bin).shape[1] // 150  # 1654/150 = 11

    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))

    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (1 X 1) ones.
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

    # Tìm các hình chữ nhật lớn.
    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sắp xếp các hình chữ nhật theo thứ tự từ trên xuống
    def sort_contours(cnts, method="left-to-right"):

        reverse = False
        i = 0

        if method == "right-to-left" or method == "bottom-to-top":
            reverse = False

        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1

        # danh sách các contours được sắp xếp từ trên xuống
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]

        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

        # danh sách các contours
        return (cnts, boundingBoxes)

    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")


    tmp_anh = []
    for c in contours:
        # kết quả giá trị w và h sau khi tìm Contours.
        x, y, w, h = cv2.boundingRect(c)
        min_w1 = 200
        max_w1 = 700
        max_w2 = 800
        min_h1 = 47
        max_h1 = 73
        # điều kiện cắt
        if (min_w1 < w < max_w1 and min_h1 < h < max_h1 or w > max_w2 and min_h1 < h < max_h1):
            new_img = img_resize[y:y + h, x:x + w]
            new_img_convert = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
            tmp_anh.append(new_img_convert)
            # idx += 1

    nguong_pixel = 128
    tmp_ket_qua = []
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

        map_name = ["Ten_sv: ", "Zalo_sv: ", "Sdt_sv: ", "Facebook_sv: ", "Email_sv: ",
                    "Trinh_do: ", "truong: ", "TT_khac_sv: ", "Nganh_khac_sv: "]

        for k in tmp_sum:
            # print(k)
            if (k < 10000):
                ngat_dong = "hết một thông tin!!!"
                tmp_ket_qua.append(ngat_dong)
                tmp_ket_qua.append(str(map_name[h]))
                img_resize = cv2.resize(tmp_anh[h], (900, 100), interpolation=cv2.INTER_AREA)

                # Phân ngưỡng nhị phân
                th, im_th = cv2.threshold(img_resize, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                # Đảo ngược giá trị điểm ảnh
                image = np.invert(im_th)

                # dao nguoc gia tri anh
                img_bin = 255 - image

                # Độ lớn của kernel
                kernel_length = np.array(img_bin).shape[1] // 150

                verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))

                hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
                # A kernel of (1 X 1) ones.
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

                # Kết hợp tạo các ô vuông.
                img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0)
                img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=1)

                (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                # Tìm các hình chữ nhật lớn.
                contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # Sắp xếp các hình chữ nhật theo thứ tự từ trái sang phải
                def sort_contours(cnts, method="left-to-right"):
                    reverse = False
                    i = 0

                    if method == "right-to-left" or method == "bottom-to-top":
                        reverse = False

                    if method == "top-to-bottom" or method == "bottom-to-top":
                        i = 1

                    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

                    (cnts, boundingBoxes) = zip(
                        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))

                    return (cnts, boundingBoxes)

                (contours, boundingBoxes) = sort_contours(contours, method="left-to-right")


                # Điều kiện của Contours
                for c in contours:
                    x, y, w, h = cv2.boundingRect(c)
                    min_w1 = 24
                    max_w1 = 45
                    min_w2 = 45
                    max_w2 = 110
                    min_h1 = 72
                    max_h1 = 90
                    if (min_w1 < w < max_w1 and min_h1 < h < max_h1 or min_w2 < w < max_w2 and min_h1 < h < max_h1):
                        # Ảnh dùng nhận dạng
                        new_img = img_resize[y:y + h, x:x + w]

                        # chuyển kích thước ảnh về 128x128
                        img_resize_lan_1 = cv2.resize(new_img, (128, 128),
                                                      interpolation=cv2.INTER_AREA)

                        img_color = cv2.cvtColor(img_resize_lan_1, cv2.COLOR_GRAY2RGB)

                        # chuẩn hóa dữ liệu trước khi dự đoán
                        normalized_image_array = (img_color.astype(np.float32) / 255)

                        names = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                                 10: 'A', 11: 'Â',
                                 12: 'Ầ', 13: 'Ậ', 14: 'Ẩ', 15: 'Ấ', 16: 'Ẫ', 17: 'À', 18: 'Ạ', 19: 'Ả', 20: 'Á',
                                 21: 'Ă', 22: 'Ằ',
                                 23: 'Ặ', 24: 'Ẳ', 25: 'Ắ', 26: 'Ẵ', 27: 'Ã', 28: 'B', 29: 'C', 30: 'D', 31: 'Đ',
                                 32: 'E', 33: 'Ê',
                                 34: 'Ề', 35: 'Ệ', 36: 'Ể', 37: 'Ế', 38: 'Ễ', 39: 'È', 40: 'Ẹ', 41: 'Ẻ', 42: 'É',
                                 43: 'Ẽ', 44: 'F',
                                 45: 'G', 46: 'H', 47: 'I', 48: 'Ì', 49: 'Ị', 50: 'Ỉ', 51: 'Í', 52: 'Ĩ', 53: 'J',
                                 54: 'K', 55: '@',
                                 56: '.', 57: '_', 58: 'L', 59: 'M', 60: 'N', 61: 'O', 62: 'Ò', 63: 'Ọ', 64: 'Ô',
                                 65: 'Ồ', 66: 'Ộ',
                                 67: 'Ố', 68: 'Ố', 69: 'Ỗ', 70: 'Ỏ', 71: 'Ó', 72: 'Ơ', 73: 'Ờ', 74: 'Ợ', 75: 'Ở',
                                 76: 'Ớ', 77: 'Ỡ',
                                 78: 'Õ', 79: 'P', 80: 'Q', 81: 'R', 82: 'S', 83: 'T', 84: 'U', 85: 'Ù', 86: 'Ụ',
                                 87: 'Ủ', 88: 'Ú',
                                 89: 'Ư', 90: 'Ừ', 91: 'Ự', 92: 'Ử', 93: 'Ứ', 94: 'Ữ', 95: 'Ũ', 96: 'V', 97: 'W',
                                 98: 'X', 99: 'Y',
                                 100: 'Ý', 101: 'Z', 102: 'a', 103: 'â', 104: 'ầ', 105: 'ậ', 106: 'ẩ', 107: 'ấ',
                                 108: 'ẫ', 109: 'à',
                                 110: 'ạ', 111: 'ả', 112: 'á', 113: 'ă', 114: 'ằ', 115: 'ặ', 116: 'ẳ', 117: 'ắ',
                                 118: 'ẵ', 119: 'ã',
                                 120: 'b', 121: 'd', 122: 'đ', 123: 'e', 124: 'ê', 125: 'ề', 126: 'ệ', 127: 'ể',
                                 128: 'ế', 129: 'ễ',
                                 130: 'è', 131: 'ẹ', 132: 'ẻ', 133: 'é', 134: 'ẽ', 135: 'f', 136: 'g', 137: 'h',
                                 138: 'i', 139: 'ì',
                                 140: 'ị', 141: 'ỉ', 142: 'í', 143: 'ĩ', 144: 'j', 145: 'k', 146: 'l', 147: 'm',
                                 148: 'n', 149: 'q',
                                 150: 'r', 151: 's', 152: 't', 153: 'u', 154: 'ù', 155: 'ụ', 156: 'ủ', 157: 'ú',
                                 158: 'ư', 159: 'ừ',
                                 160: 'ự', 161: 'ử', 162: 'ứ', 163: 'ữ', 164: 'ũ', 165: 'v', 166: 'x', 167: 'y',
                                 }
                        # dự đoán
                        prediction = model.predict(normalized_image_array.reshape(-1, 128, 128, 3))

                        # lấy phần tử có giá trị lớn nhất
                        predict_img = np.argmax(prediction, axis=-1)

                        # mảng kết quả
                        tmp_ket_qua.append(names.get(predict_img[0]))

            else:
                continue

    return (tmp_ket_qua)


@app.route('/')
@app.route('/home', methods=['get'])
def home():
    response_mua = requests.get('http://localhost/php_framework/public/api/get_api_mua')
    response_mua = response_mua.json()

    response_tinh = requests.get('http://localhost/php_framework/public/api/get_api_tinh')
    response_tinh = response_tinh.json()

    return render_template('home.html', data=response_tinh, len=len(response_tinh)
                           , data_mua=response_mua)


@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        files = request.files.getlist('image')
        nam_nhap = request.form['ma_mua']
        ma_nv = request.form['ma_nv']
        ma_tinh = request.form['Tinh_TP']
        ma_tp_qh = request.form['Quan_huyen']
        ma_truong = request.form['Truong_hoc']
        data_from = {
            'id_mua': str(nam_nhap),
            'Ma_nv': str(ma_nv),
            'id_tinh_tp': str(ma_tinh),
            'id_tp_quanhuyen': str(ma_tp_qh),
            'nganh_yeu_thich': 'Lập trình viên Quốc tế - APTECH',
            'id_truong': str(ma_truong),
            'nhu_cau_hoc': 'Quốc tế + Cao đẳng (Kỹ sư thực hành - thời gian đào tạo: 2,5 năm)'
        }

        File_chuan = cv2.imread('data/anh_chuan.jpg')
        File_chuan = cv2.cvtColor(File_chuan, cv2.COLOR_BGR2GRAY)
        couter = 1
        for file in files:
            with open('data/text.txt', "w+", encoding='UTF-8') as f:
                # Read image
                img = cv2.imdecode(np.fromstring(file.read(), np.uint8), 1)
                xoay_anh = Image_Alignment(File_chuan, img)
                preds = detect(xoay_anh, model)

                f.write("anh:%s" % (str(couter)))
                for i in preds:
                    if (str(i) == "hết một thông tin!!!"):
                        f.write("\r")
                    else:
                        f.write("%s" % (str(i)))
                couter += 1
            f.close()

            filepath = 'data/text.txt'
            f = open(filepath, encoding="UTF-8")
            data_json = {}
            for line in f:
                k, v = line.strip().split(':')
                data_json[k.strip()] = v.strip()

            f.close()

            if 'anh' in data_json:
                del data_json['anh']

            if 'truong' in data_json:
                del data_json['truong']

            data_convert = json.dumps(data_json, sort_keys=False)

            payload = merge(data_from, data_json)

            print(payload)

            headers = {}
            files = {}

            url = "http://localhost/php_framework/public/api/post_api_nhan_vien"

            response = requests.request("POST", url, headers=headers, data=payload, files=files)

            response = response.json()

    return render_template('result.html', results=data_convert, data=response)


@app.route('/download', methods=['POST', 'GET'])
def download_file():
    path = 'data/text.txt'
    return send_file(path, as_attachment=True, conditional=True)


if __name__ == '__main__':
    webbrowser.open_new('http://127.0.0.1:5000/')
    # app.run(debug=True, host="0.0.0.0", port='8050')
    app.run()
