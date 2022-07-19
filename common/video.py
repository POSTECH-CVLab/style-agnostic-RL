import imageio
import os
import numpy as np


class VideoRecorder(object):
    def __init__(self, dir_name, height=256, width=256, camera_id=0, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env, mode='rgb_array'):
        if self.enabled:
            try:
                frame = env.render(
                    mode=mode,
                    height=self.height,
                    width=self.width,
                    camera_id=self.camera_id
                )
            except:
                frame = env.render(
                    mode=mode
                )
    
            self.frames.append(frame)

    def render(self, img):
        frame = img.copy()
        frame = np.transpose(frame, (1, 2, 0))
        self.frames.append((frame * 255).astype(np.uint8))

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
