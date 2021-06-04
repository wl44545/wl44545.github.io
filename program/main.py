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
#


start = time.time()
xxx.load_data()
end = time.time()
print(end - start)
