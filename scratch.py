import numpy as np

start_time = 1657360961816342.0
end_time = 1657364596855842.0

duration_ns = end_time - start_time
duration_min = duration_ns / (1e6 * 60)

print("Time duration in minutes:", duration_min)