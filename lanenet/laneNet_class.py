import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

# Use explicit package-relative imports to avoid clashing with the outer `lanenet` package
from lanenet.LaneNet import lanenet as lanenet_model
from lanenet.LaneNet import parse_config_utils


class LaneNet(object):
    def __init__(self):
        self.cfg = parse_config_utils.lanenet_cfg
        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
        self.net = lanenet_model.LaneNet(phase='test', cfg=self.cfg)
        self.binary_seg_ret, self.instance_seg_ret = self.net.inference(input_tensor=self.input_tensor, name='LaneNet')

        self.weights_path = self._resolve_weights_path()

        # Set sess configuration
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = self.cfg.GPU.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = self.cfg.GPU.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'

        self.sess = tf.Session(config=sess_config)



        # define moving average version of the learned variables for eval
        with tf.variable_scope(name_or_scope='moving_avg'):
            variable_averages = tf.train.ExponentialMovingAverage(
                self.cfg.SOLVER.MOVING_AVE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()

        self.saver = tf.train.Saver(variables_to_restore)
        self.saver.restore(sess=self.sess, save_path=self.weights_path)

        # # Use tf.train.Checkpoint for restoration
        # checkpoint = tf.train.Checkpoint(variables_to_restore=variables_to_restore)
        # checkpoint.restore(self.weights_path)
        #
        # # Restore checkpoint with expect_partial()
        # try:
        #     checkpoint.restore(self.weights_path).expect_partial()
        #     print("Checkpoint partially restored with `expect_partial()`.")
        # except Exception as e:
        #     print(f"Error restoring checkpoint: {e}")

        print("LaneNet Model Initilaized")

    def _resolve_weights_path(self):
        """Pick the first available checkpoint path to avoid missing file errors."""
        checkpoint_file = os.path.join(ROOT_PATH, "LaneNet", "weights", "new", "checkpoint")
        candidates = []
        if os.path.isfile(checkpoint_file):
            with open(checkpoint_file, "r") as f:
                line = f.readline()
                if "model_checkpoint_path" in line:
                    # model_checkpoint_path: "path"
                    ckpt_path = line.split(":", 1)[1].strip().strip('"').strip()
                    candidates.append(ckpt_path)

        # Fallback to repo-local weights if present
        candidates.append(os.path.join(ROOT_PATH, "LaneNet", "weights", "new", "tusimple_lanenet.ckpt"))

        for base in candidates:
            if os.path.exists(base + ".index") or os.path.exists(base + ".meta") or os.path.exists(base):
                return base

        raise FileNotFoundError(
            "LaneNet checkpoint not found. Checked: {}".format(", ".join(candidates))
        )

    @staticmethod
    def preProcessing(image):
        image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
        image = image / 127.5 - 1.0
        return image


    def predict(self,image):
        src_image = self.preProcessing(image)
        
        with self.sess.as_default():
            self.binary_seg_image, self.instance_seg_image = self.sess.run(
                [self.binary_seg_ret, self.instance_seg_ret],
                feed_dict={self.input_tensor: [src_image]}
            )
            rgb = self.instance_seg_image[0].astype(np.uint8)
            bw = self.binary_seg_image[0].astype(np.uint8)
            res = cv2.bitwise_and(rgb,rgb,mask=bw)

            lanes_rgb,center_xy = self.postProcess(res)
            return lanes_rgb,center_xy

    def postProcess(self,image):
        src_img = cv2.cvtColor(image,cv2.COLOR_RGBA2RGB)
    
        red_mask = (src_img[:,:,2]>200).astype(np.uint8)
        src_img = cv2.bitwise_and(src_img,src_img,mask=1-red_mask)
        
        #Right Lanes
        green_mask = (src_img[:,:,1]>200).astype(np.uint8)
        green_area = cv2.bitwise_and(src_img,src_img,mask=green_mask)

        #Left Lanes
        blue_mask = (src_img[:,:,0]>200).astype(np.uint8)
        blue_area = cv2.bitwise_and(src_img,src_img,mask=blue_mask)

        lanes_rgb = cv2.addWeighted(green_area,1,blue_area,1,0)

        img_center_point,center_xy = self.window_search(green_mask,blue_mask)
        lanes_rgb = cv2.addWeighted(lanes_rgb,1,img_center_point,1,0)

        return lanes_rgb,center_xy

    @staticmethod
    def window_search(righ_lane, left_lane):
        center_coordinates =[]
        out = np.zeros(righ_lane.shape,np.uint8)
        out = cv2.merge((out,out,out))

        mid_point = int(righ_lane.shape[1]/2)

        nwindows = 9
        h = righ_lane.shape[0]
        vp = int(h/2)
        window_height = int(vp/nwindows)

        r_lane = righ_lane[vp:,:].copy()
        r_lane = cv2.erode(r_lane,np.ones((3,3)))

        l_lane = left_lane[vp:,:]
        l_lane = cv2.erode(l_lane,np.ones((3,3)))
        
        for window in range(nwindows):
            win_y_low = vp - (window+1)*window_height
            win_y_high = vp - window*window_height
            win_y_center = win_y_low + int((win_y_high-win_y_low)/2)

            r_row = r_lane[win_y_low:win_y_high,:]
            l_row = l_lane[win_y_low:win_y_high,:]

            histogram = np.sum(r_row, axis=0)
            r_point = np.argmax(histogram)

            histogram = np.sum(l_row, axis=0)
            l_point = np.argmax(histogram)

            if(l_point != 0) and (r_point != 0):
                rd = r_point-mid_point
                ld = mid_point-l_point
                if(abs(rd-ld)<100):
                    center = l_point + int((r_point-l_point)/2)
                    out = cv2.circle(out,(center,vp+win_y_center),2,(0,0,255),-1)
                    center_coordinates.append((center,vp+win_y_center))
        return out,center_coordinates
