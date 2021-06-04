import cv2

import program.data
import time

xxx = program.data.Data()


# start = time.time()
# xxx.import_data()
# end = time.time()
# print(end - start)
#
#
# start = time.time()
# xxx.dump_data()
# end = time.time()
# print(end - start)



start = time.time()
xxx.load_data()
end = time.time()
print(end - start)


cv2.startWindowThread()
image = xxx.images[-1]
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (100,100))
cv2.imshow("aa", image)
cv2.waitKey()
