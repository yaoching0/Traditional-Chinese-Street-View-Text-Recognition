from torchvision import transforms
import random
import numpy
from PIL import Image

#模擬不同距離拍攝character的情景
class distance_simulator(object):
    
    def __call__(self, img):  
        width, height = img.size

        #隨機縮小為原本的1/2、1/3、1/4、1/5和1/6之尺寸(或保持原樣)
        reduce_ratio=random.choice([2,3,4,5,6]+[0]*3)
        #print(f'reduce_ratio {reduce_ratio}')
        
        #0輸出原本的img
        if reduce_ratio==0:
            return img

        else:
            out_width=round(width/reduce_ratio)
            out_height=round(height/reduce_ratio)

            #任一一邊長度小於15則放棄增強
            if out_width<15 or out_height<15:
                return img
            
            return transforms.Resize([out_height,out_width])(img)

#自定義一個專為中文字的透視變換的transform
class chinese_perTransforms(object):
        
    #PIL透視變換用
    def find_coeffs(self,source_coords, target_coords):
        matrix = []
        for s, t in zip(source_coords, target_coords):
            matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
            matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
        A = numpy.matrix(matrix, dtype=numpy.float)
        B = numpy.array(source_coords).reshape(8)
        res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
        return numpy.array(res).reshape(8)
    
    def __call__(self, img):  
        width, height = img.size
        #梯形、平行四邊形、不變
        #[左上，右上，右下，左下]

        shape=random.choice([[(0, 0), (width-1, 0), (int(5*(width-1)/6), height-1), (int((width-1)/6), height-1)],
        [(int((width-1)/6), 0), (int(5*(width-1)/6), 0),  ((width-1), height-1),(0, height-1)],  
        [(0, int((height-1)/6)), ((width-1), 0), ((width-1), height-1), (0, int(5*(height-1)/6))],
        [(0, 0), ((width-1), int((height-1)/6)), ((width-1),int(5*(height-1)/6)), (0, height-1)],
        #左高右低平行四邊形
        [(0, 0), ((width-1), int((height-1)/5)),(width-1, height-1), (0, int(4*(height-1)/5))],
        [(int((width-1)/5), 0), (int(4*(width-1)/5), int((height-1)/5)),(int(4*(width-1)/5), (height-1)), (int((width-1)/5), int(4*(height-1)/5))],
        [(int((width-1)/3), 0), (int(2*(width-1)/3), int((height-1)/3)),(int(2*(width-1)/3), (height-1)), (int((width-1)/3), int(2*(height-1)/3))],
        #右高左低平行四邊形
        [(0, int((height-1)/5)), ((width-1), 0),(width-1, int(4*(height-1)/5)), (0, height-1)],    
        [(int((width-1)/5), int((height-1)/5)), (int(4*(width-1)/5), 0),(int(4*(width-1)/5), int(4*(height-1)/5)), (int((width-1)/5), (height-1))],
        [(int((width-1)/3), int((height-1)/3)), (int(2*(width-1)/3), 0),(int(2*(width-1)/3), int(2*(height-1)/3)), (int((width-1)/3), (height-1))],
        #原始形狀
        [(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)],
        [(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)],
        #[(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)],
        ])

        try:
            coeffs = self.find_coeffs([(0, 0), (width-1, 0), (width-1, height-1), (0, height-1)],shape)
            img=img.transform(img.size, Image.PERSPECTIVE, coeffs,
                        Image.BICUBIC)
        except:
            img.show()
            return img
        else:
            return img

#自定義一個旋轉固定角度的transform
class rotTransforms(object):
    def __call__(self, img):  
        angle = random.choice(list(range(-45,45,3))+[0]*25)
        img = transforms.functional.rotate(img, angle)
        return img