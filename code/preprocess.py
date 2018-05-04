import numpy as np 

class FramesDiff(object): 

    def __init__(self, frame_size, scale=1.): 
        self.frame_size = frame_size
        self.prev_frame = np.zeros(self.frame_size)
        self.scale = float(scale)
        pass 

    def Reset(self): 
        self.prev_frame = np.zeros(self.frame_size)

    def __call__(self, new_input):
        out = new_input - self.prev_frame
        out = out.flatten()[:,np.newaxis] / self.scale

        self.prev_frame = new_input

        return out 