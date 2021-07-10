import tensorflow as tf
import time
import random
import glob
from model import *
from loss import *
from datagenerator import *
from scipy.interpolate import interpn
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

img_height = 128
img_width = 144
img_depth = 112
img_layer = 1
label_layer = 5
vol_size = (img_height, img_width, img_depth)

to_train = True
to_test = False
to_restore = True
batch_size = 1
flag = 1
output_path = "./check/"
check_dir = "./check/TDRegSemi"

ngf = 32
ndf = 64

train_dir = "LPB40/stage2"

train_vol_names = glob.glob(train_dir + '/*.mat')


class CycleRES():
    def model_setup(self, model, is_training, z):

        self.flow_A = tf.placeholder(tf.float32, [batch_size, img_height, img_width, img_depth, img_layer], name="flow_A")
        self.input_A = tf.placeholder(tf.float32, [batch_size, img_height, img_width, img_depth, img_layer], name="mov_A")
        self.input_A_org = tf.placeholder(tf.float32, [batch_size, img_height, img_width, img_depth, img_layer], name="mov_A_org")
        self.input_B = tf.placeholder(tf.float32, [batch_size, img_height, img_width, img_depth, img_layer], name="fix_B")
        self.label_A = tf.placeholder(tf.float32, [batch_size, img_height, img_width, img_depth, label_layer], name="lable_A")
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.label_B = tf.placeholder(tf.float32, [batch_size, img_height, img_width, img_depth, label_layer],
                                      name="lable_B")
        self.label_A_org = tf.placeholder(tf.float32, [batch_size, img_height, img_width, img_depth, img_layer],
                                          name="lable_A_org")

        self.num_fake_inputs = 0

        self.lr = tf.placeholder(tf.float32, shape=[], name="lr")
        nf_enc = [16, 32, 32, 32]
        if (model == 'vm1'):
            nf_dec = [32, 32, 32, 32, 8, 8, 3]
        else:
            nf_dec = [32, 32, 32, 32, 32, 16, 16, 3]

        with tf.variable_scope("Model1") as scope:
            self.output_A = unet(self.input_A, train=is_training, name="att_A")
            scope.reuse_variables()
            self.output_B = unet(self.input_B, train=is_training, name="att_A")
        self.feature_A = self.output_A * self.input_A
        self.feature_B = self.output_B * self.input_B
        with tf.variable_scope("Model2") as scope:
            self.warp_AtoB, self.flow_AtoB, self.warp_AtoB_org,self.warp_gt_AtoB= build_generator_unet(self.feature_A, self.label_A,
                                                                                       self.input_A_org,
                                                                                       self.feature_B,
                                                                                       nf_enc,
                                                                                       nf_dec,
                                                                                       train=is_training,
                                                                                       stride_z=z, name="g_A")
        with tf.variable_scope("Model1") as scope:
            scope.reuse_variables()
            self.output_AtoB = unet(self.warp_AtoB_org, train=is_training, name="att_A")

    def loss_calc(self):
        seg_loss_AtoB = 10 * crossEntropyLoss_multi(self.warp_gt_AtoB, self.output_AtoB)
        seg_loss_AtoB = tf.Print(seg_loss_AtoB, [seg_loss_AtoB], 'seg_loss_AtoB')

        seg_loss_AtoB1 = 5 * crossEntropyLoss_multi(self.label_B, self.output_AtoB)
        seg_loss_AtoB1 = tf.Print(seg_loss_AtoB1, [seg_loss_AtoB1], 'seg_loss_AtoB1')

        seg_loss_B = 10 * crossEntropyLoss_multi(self.label_B, self.output_B)
        seg_loss_B = tf.Print(seg_loss_B, [seg_loss_B], 'seg_loss_B')
        joint_loss_segB = seg_loss_B+seg_loss_AtoB1
        joint_loss_segB = tf.Print(joint_loss_segB, [joint_loss_segB], 'joint_loss_segB')

        seg_loss_A = 10 * crossEntropyLoss_multi(self.label_A, self.output_A)
        seg_loss_A = tf.Print(seg_loss_A, [seg_loss_A], 'seg_loss_A')
        joint_loss_segA = seg_loss_A+seg_loss_AtoB
        joint_loss_segA = tf.Print(joint_loss_segA, [joint_loss_segA], 'joint_loss_segA')

        joint_loss_segAB =seg_loss_A+ seg_loss_B+seg_loss_AtoB
        joint_loss_segAB = tf.Print(joint_loss_segAB, [joint_loss_segAB], 'joint_loss_segAB')

        self.label_B1 = self.label_B[:, :, :, :, 1]
        self.label_B1 = self.label_B1[:, :, :, :, tf.newaxis]
        self.label_B2 = self.label_B[:, :, :, :, 2]
        self.label_B2 = self.label_B2[:, :, :, :, tf.newaxis]
        self.label_B3 = self.label_B[:, :, :, :, 3]
        self.label_B3 = self.label_B3[:, :, :, :, tf.newaxis]
        self.label_B4 = self.label_B[:, :, :, :, 4]
        self.label_B4 = self.label_B4[:, :, :, :, tf.newaxis]

        reg_loss_img1 = cc3D(self.input_B * self.label_B1, self.warp_AtoB[:, :, :, :, 1])
        reg_loss_img2 = cc3D(self.input_B * self.label_B2, self.warp_AtoB[:, :, :, :, 2])
        reg_loss_img3 = cc3D(self.input_B * self.label_B3, self.warp_AtoB[:, :, :, :, 3])
        reg_loss_img4 = cc3D(self.input_B * self.label_B4, self.warp_AtoB[:, :, :, :, 4])

        reg_loss_img_a = reg_loss_img1 + reg_loss_img2 + reg_loss_img3 + reg_loss_img4
        reg_loss_img_a = tf.Print(reg_loss_img_a, [reg_loss_img_a], 'reg_loss_img_a')
        gdloss = gradientLoss(self.flow_AtoB, 'l2')
        gdloss = tf.Print(gdloss, [gdloss], 'gdloss')
        joint_loss2 = reg_loss_img_a + gdloss
        joint_loss2 = tf.Print(joint_loss2, [joint_loss2], 'joint_loss2')

        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)
        self.model_vars = tf.trainable_variables()

        S_A_vars = [var for var in self.model_vars if 'att_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.joint_trainer_SEGA = optimizer.minimize(joint_loss_segA, var_list=S_A_vars)
            self.joint_trainer_SEGB = optimizer.minimize(joint_loss_segB, var_list=S_A_vars)
            self.joint_trainer_SEGAB = optimizer.minimize(joint_loss_segAB, var_list=S_A_vars)
            self.joint_trainer_REGA = optimizer.minimize(joint_loss2, var_list=g_A_vars)
        for var in self.model_vars: print(var.name)
        self.joint_loss_sum_SEGA = tf.summary.scalar("joint_loss_segA", joint_loss_segA)
        self.joint_loss_sum_SEGB = tf.summary.scalar("joint_loss_segB", joint_loss_segB)

        self.joint_loss_sum_SEGAB = tf.summary.scalar("joint_loss_segAB", joint_loss_segAB)
        self.joint_loss_sum_REGA = tf.summary.scalar("joint_loss2", joint_loss2)

    def Get_Ja(self, displacement):
        D_y = (displacement[:, 1:, :-1, :-1, :] - displacement[:, :-1, :-1, :-1, :])
        D_x = (displacement[:, :-1, 1:, :-1, :] - displacement[:, :-1, :-1, :-1, :])
        D_z = (displacement[:, :-1, :-1, 1:, :] - displacement[:, :-1, :-1, :-1, :])
        D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
        D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])
        D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])
        return D1 - D2 + D3

    def train(self):
        self.model_setup(model='vm2', is_training=to_train, z=2)

        # Loss function calculations
        self.loss_calc()

        # Initializing the global variables
        init = tf.global_variables_initializer()
        saver_all = tf.train.Saver()

        with tf.Session() as sess:

            sess.run(init)
            if to_restore:
                chkpt_fname = os.path.join(check_dir, 'model-149')
                saver_all.restore(sess, chkpt_fname)
            writer = tf.summary.FileWriter("./check/EX_att")

            if not os.path.exists(check_dir):
                os.makedirs(check_dir)

            # Training Loop
            for epoch in range(sess.run(self.global_step), 300):
                print("In the epoch ", epoch)
                saver_all.save(sess, os.path.join(check_dir, "model"), global_step=epoch)

                # Dealing with the learning rate as per the epoch number
                curr_lr = 0.0001
                random.shuffle(train_vol_names)
                train_example_gen = example_gen_3D_brain_ag(train_vol_names, bs=batch_size, data_aug=True)
                num_images = len(train_vol_names)
                iters = num_images//batch_size
                for ptr in range(0, iters):
                    print("Epoch: [%2d] [%4d] time: %4.4fs"
                        % (epoch, ptr, time.time()))

                    data = train_example_gen.__next__()
                    self.mov_input = data[0]
                    self.fix_input = data[1]
                    self.mov_label_org = data[5]
                    flag = data[4]
                    if flag == 1:
                        _, summary_str4 = sess.run([self.joint_trainer_SEGAB, self.joint_loss_sum_SEGAB],
                                                   feed_dict={self.label_B: data[2],
                                                              self.label_A: data[3],
                                                              self.label_A_org: self.mov_label_org,
                                                              self.input_A_org: self.mov_input,
                                                              self.input_A: self.mov_input,
                                                              self.input_B: self.fix_input,
                                                              self.lr: curr_lr})
                        _, summary_str5 = sess.run([self.joint_trainer_REGA, self.joint_loss_sum_REGA],
                                                   feed_dict={self.label_B: data[2],
                                                              self.label_A: data[3],
                                                              self.input_A_org: self.mov_input,
                                                              self.input_A: self.mov_input,
                                                              self.input_B: self.fix_input,
                                                              self.lr: curr_lr})
                    if flag == 2:
                        _, summary_str4 = sess.run([self.joint_trainer_SEGB, self.joint_loss_sum_SEGB],
                                                   feed_dict={self.label_B: data[2],
                                                              self.label_A: data[3],
                                                              self.input_A_org: self.mov_input,
                                                              self.input_A: self.mov_input,
                                                              self.input_B: self.fix_input,
                                                              self.lr: curr_lr})
                        _, summary_str5 = sess.run([self.joint_trainer_REGA, self.joint_loss_sum_REGA],
                                                   feed_dict={self.label_B: data[2],
                                                              self.label_A: data[3],
                                                              self.input_A_org: self.mov_input,
                                                              self.input_A: self.mov_input,
                                                              self.input_B: self.fix_input,
                                                              self.lr: curr_lr})
                    if flag == 3:
                        _, summary_str4 = sess.run([self.joint_trainer_SEGA, self.joint_loss_sum_SEGA],
                                                   feed_dict={self.label_B: data[2],
                                                              self.label_A: data[3],
                                                              self.label_A_org: self.mov_label_org,
                                                              self.input_A_org: self.mov_input,
                                                              self.input_A: self.mov_input,
                                                              self.input_B: self.fix_input,
                                                              self.lr: curr_lr})


                sess.run(tf.assign(self.global_step, epoch + 1))

            writer.add_graph(sess.graph)

    def nearest(self, flow,img):
        xx = np.arange(vol_size[1])
        yy = np.arange(vol_size[0])
        zz = np.arange(vol_size[2])
        a = np.array(np.meshgrid(xx, yy, zz))
        grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)  # 用网格点表示，交换坐标轴

        sample = flow[0, :, :, :, :] + grid
        sample = np.stack((sample[:, :, :, 1], sample[:, :, :, 0], sample[:, :, :, 2]), 3)  # 沿着新轴建立数组连接的序列

        warp_gt = interpn((yy, xx, zz), img, sample, method='linear',
                              bounds_error=False,
                              fill_value=0)
        return warp_gt, img

    def test(self):
        print("Testing the results")
        test_dir = "LPB40/test"
        test_dir_2 ="LPB40/test"
        filelist = os.listdir(test_dir)
        filelist2 = os.listdir(test_dir_2)
        num_images = len(filelist)

        self.model_setup(model='vm2', is_training=to_train, z=2)

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            chkpt_fname = os.path.join(check_dir, 'model-149')
            saver.restore(sess, chkpt_fname)

            savepath = 'results_MM_stage2'
            if not os.path.exists(savepath):
                os.makedirs(savepath)
            count = []
            for i in range(0, num_images):
                print(i)
                vol_name = filelist[i]
                vol_name = test_dir + '/' + vol_name
                mov_data = sio.loadmat(vol_name)['mov_img']
                mov_data = limit(mov_data)
                mov_data = standardization(mov_data)
                mov_data = mov_data[np.newaxis, :, :, :, np.newaxis]
                mov_data_org = mov_data
                self.mov_input = mov_data

                fixed_vol = sio.loadmat(vol_name)['fixed_img']
                fixed_data = fixed_vol
                fixed_data = limit(fixed_data)
                fixed_data = standardization(fixed_data)
                fixed_data = fixed_data[np.newaxis, :, :, :, np.newaxis]
                fixed_data_org = fixed_data
                self.fix_input = fixed_data

                fixed_label_org = sio.loadmat(test_dir_2 + '/' + filelist2[i])['fixed_label']
                fixed_label_org = fixed_label_org.astype('int16')
                d = np.where(fixed_label_org > 4)
                fixed_label_org[d] = 0

                mov_label_org = sio.loadmat(test_dir_2 + '/' + filelist2[i])['mov_label']
                mov_label_org = mov_label_org.astype('float32')
                d = np.where(mov_label_org > 4)
                mov_label_org[d] = 0
                mov_label_org = mov_label_org[np.newaxis, :, :, :, np.newaxis]

                y, flow, warp_gt, predict_mov, predict_fix = sess.run(
                    [self.warp_AtoB, self.flow_AtoB, self.warp_gt_AtoB, self.output_A, self.output_B], feed_dict={
                        self.label_A_org: mov_label_org,
                        self.input_A_org: mov_data_org,
                        self.input_A: self.mov_input,
                        self.input_B: self.fix_input})
                warp_gt = np.squeeze(warp_gt)
                warp_mask = np.where(warp_gt > 0, 1, 0)
                y = np.squeeze(y)
                temp = flow[0, :, :, :, :]
                temp = sitk.GetImageFromArray(temp)
                ja1 = sitk.DisplacementFieldJacobianDeterminant(temp)
                ja1 = sitk.GetArrayFromImage(ja1)
                count = len(np.where(ja1 <= 0)[0])
                a = ja1[warp_mask > 0]
                b = a[a <= 0]
                count1 = len(b)

                mov_label_org = np.squeeze(mov_label_org)
                mov_data_org = np.squeeze(mov_data_org)
                fixed_data_org = np.squeeze(fixed_data_org)
                warp_gt = np.squeeze(warp_gt)
                predict_mov = np.squeeze(predict_mov)
                predict_fix = np.squeeze(predict_fix)
                y1 = np.argmax(predict_mov, axis=3)
                y1 = y1.astype('int16')
                y2 = np.argmax(predict_fix, axis=3)
                y2 = y2.astype('int16')
                img = sio.loadmat('grid_brain.mat')['grid']
                warp_grid, warp_reg = self.nearest(flow, img)

                y = np.squeeze(y)
                self.fix_input = np.squeeze(self.fix_input)
                self.mov_input = np.squeeze(self.mov_input)
                ydict = {'fixed': fixed_data_org, 'mov': mov_data_org, 'warp_gt': warp_gt, 'warp_src': y,
                         'label': fixed_label_org, 'mov_label': mov_label_org, 'predict_mov': y1, 'predict_fix': y2,
                         'ja': count, 'ja1': count1, 'warp_grid': warp_grid, 'warp_reg': warp_reg}
                savename = savepath + '/' + vol_name.split('/')[-1].split('.')[0] + '.mat'
                sio.savemat(savename, ydict)


def main():
    model = CycleRES()
    if to_train:
        model.train()
    elif to_test:
        model.test()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()