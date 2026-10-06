import time
begin_time = time.time()
for i in range(100):
    time.sleep(0.01)
end_time = time.time()
print( end_time - begin_time )
