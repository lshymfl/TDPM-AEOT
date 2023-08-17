import copy
import json
import os
import warnings
from absl import app, flags
import scipy.io as sio
import torch
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms
from tqdm import trange, tqdm
import time
import math
import numpy as np
from modelsDM.diffusionHQ import GaussianDiffusionTrainer, GaussianDiffusionSampler, truncatedDiffusionStep, TruncatedDiffusionSampler
from modelsDM.modelHQ import UNet
from score.both import get_inception_and_fid_score
from torchvision.datasets import ImageFolder
from modelsAE.HQ256 import autoencoder
from utils.P_loader import P_loader


#megengine.dtr.eviction_threshold = '4GB'
#megengine.dtr.enable()


FLAGS = flags.FLAGS  
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('sample', False, help='load ckpt.pt and evaluate FID and IS')
flags.DEFINE_bool('fid', False, help='load ckpt.pt and calculate the fid for each ckpt')
flags.DEFINE_bool('middle_truncated', False, help='selected steps of 0~truncated_T')
flags.DEFINE_bool('interpolation', False, help='interpolation of latent space')
# UNet
flags.DEFINE_integer('ch', 64, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 1, 2, 4, 4, 4], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [2], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.0, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value') ###
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_integer('truncated_T', 150, help='total diffusion steps')############################### --eval modified
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 1e-5, help='target learning rate')
flags.DEFINE_float('grad_clip', 0.1, help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 500000, help='total training steps')
flags.DEFINE_integer('img_size', 256, help='image size')
flags.DEFINE_integer('warmup', 100000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 16, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
flags.DEFINE_bool('resume_training', False, help='resume training') 
# Logging & Sampling
flags.DEFINE_string('logdir', './DMAE_CelebAHQ-100/DiffusionAEOT', help='log directory')############################### --eval modified
flags.DEFINE_integer('sample_size', 25, "sampling size of images")
flags.DEFINE_integer('sample_step', 10000, help='frequency of sampling')
flags.DEFINE_bool('netsample', False, help='use net_model to reverse sample')
flags.DEFINE_bool('emasample', True, help='use ema_model to reverse sample')
# Evaluation
flags.DEFINE_integer('save_step', 10000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 10000, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_batch_images', 4, help='the number of generated images for evaluation')############################### --eval modified
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
# traindiffusionAE_OT
flags.DEFINE_string('data_root_train', '6357-code/datasets/testdata', help='root of train data')
flags.DEFINE_string('AE_OT_root', './CelebAHQ-2/', help='features of AE')
flags.DEFINE_integer('dim_z', 1200, help='dimension of feature in latent space')
flags.DEFINE_integer('dim_c', 3, help='input image number of channels')
flags.DEFINE_integer('dim_f', 32, help='number of features in first layer of AE')
flags.DEFINE_integer('sampling_ckpt_id', 120000, help='selected the best ckpt for sampling')############################### --eval modified
flags.DEFINE_bool('trainDiffusion', False, help='Training diffusion model need disrupting the latent codes of AE')
flags.DEFINE_string('fid_cache', './stats/celeba_256.train.npz', help='FID cache')
flags.DEFINE_integer('begin_ckptid', 100, help='selected the start ckptid for sampling')
flags.DEFINE_integer('end_ckptid', 100, help='selected the end ckptid for sampling')
flags.DEFINE_integer('step_id', 100, help='selected the end ckptid for sampling')
 
flags.DEFINE_integer('middle_truncated_T', 100, help='Select any steps between 0 and truncated_T')############################### --eval modified 
 
####
device = torch.device('cuda:0')

def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x, y,_   in iter(dataloader): 
            yield x


def warmup_lr(step):
    return math.pow(10.0, int(step/FLAGS.warmup)) #min(step, FLAGS.warmup) / FLAGS.warmup  #math.pow(0.1, int(step/FLAGS.warmup)) #

def evaluate(sampler,DiffusionStep,  model, features, AE):
    model.eval()
    start = time.time()
    feature_indices = torch.randperm(features.size(0))
    features = features[feature_indices]
    with torch.no_grad():
        images = []   
        for i in trange(0, min(FLAGS.num_batch_images, int(features.shape[0]/FLAGS.sample_size)), 1, desc='generating images'): #range(int(FLAGS.num_batch_images)):
            z = features.view(features.shape[0],-1,1,1).cuda()
            z1 = z[i*FLAGS.sample_size:(i+1)*FLAGS.sample_size,:,:,:]
            x_T = AE.decoder(z1)
            x_T = DiffusionStep(x_T, FLAGS.truncated_T)
            #print(x_T.shape)
            batch_images = sampler(x_T.to(device)).cpu()
            images.append(batch_images)
        images = torch.cat(images, dim=0).numpy()

    print(images.shape)  
    end = time.time()
    print("generated images %.4f seconds." % (end - start))
    model.train()
    #np.savez("./gen_celebahq-{}.npz".format(FLAGS.truncated_T), arr_0=images)
    generate_images = min(FLAGS.num_batch_images, int(features.shape[0]/FLAGS.sample_size))*FLAGS.sample_size
    (IS, IS_std), FID = get_inception_and_fid_score(images, FLAGS.fid_cache, num_images=generate_images,
                        use_torch=FLAGS.fid_use_torch, verbose=True)
 
    return (IS, IS_std), FID, images


def train(features,DiffusionStep, data_root_train, AE):
   
    
    img_transform = transforms.Compose([ 
                #transforms.RandomHorizontalFlip(),  
                transforms.ToTensor(),
            ])
    dataset = P_loader(root=data_root_train, transform=img_transform)  
    #dataset = ImageFolder(root=data_root_train, transform=img_transform)    
      
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

    # model setup
    net_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.truncated_T).to(device)
    net_sampler = GaussianDiffusionSampler(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.truncated_T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.truncated_T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
     
    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)

    # log setup
    os.makedirs(os.path.join(FLAGS.logdir, 'sample'), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.logdir, 'ckpt'), exist_ok=True)
    
    ########### AE feature #######################
    
    z = features.view(features.size(0),-1,1,1).cuda()
    z_batch = z[0:FLAGS.sample_size,:,:,:]
    with torch.no_grad():
        x_T = AE.decoder(z_batch)
    #save_image(x_T, os.path.join(FLAGS.logdir,  'rec.png' ),nrow=6)
    x_T = DiffusionStep(x_T, FLAGS.truncated_T)
    ###########################
   
    writer = SummaryWriter(FLAGS.logdir)
    writer.flush()
    # backup all arguments
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))
    ################### resume training ###########################################
    if FLAGS.resume_training:
        print(FLAGS.sampling_ckpt_id)
        ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt', f'ckpt_{FLAGS.sampling_ckpt_id}.pt'))  ###########added
        net_model.load_state_dict(ckpt['net_model'])
        ema_model.load_state_dict(ckpt['ema_model'])
        sched.load_state_dict(ckpt['sched']) 
        optim.load_state_dict(ckpt['optim']) 
        step = ckpt['step']
   ########################################################################### 

    # start training
    with trange(FLAGS.total_steps, dynamic_ncols=True) as pbar:
        for step in pbar:
            # train
            optim.zero_grad()
            x_0 = next(datalooper).to(device)
            #print(x_0.shape) 
            loss = trainer(x_0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                net_model.parameters(), FLAGS.grad_clip)
            optim.step()
            sched.step()
            #print("lr of epoch", step, "=>", sched.get_lr())
            ema(net_model, ema_model, FLAGS.ema_decay)

            # log
            writer.add_scalar('loss', loss, step)
            pbar.set_postfix(loss='%.4f' % loss)

            # sample
            if FLAGS.sample_step > 0 and (step+1) % FLAGS.sample_step == 0:
                net_model.eval()
                with torch.no_grad():
                    x_0 = ema_sampler(x_T)
                save_image(
                    x_0, os.path.join(
                        FLAGS.logdir, 'sample', '%d.png' % (step+1)),
                    nrow=5)
                net_model.train()

            # save
            if FLAGS.save_step > 0 and (step+1) % FLAGS.save_step == 0:
                ckpt = {
                    'net_model': net_model.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'sched': sched.state_dict(),
                    'optim': optim.state_dict(),
                    'step': step,
                    'x_T': x_T,
                }
                torch.save(ckpt, os.path.join(FLAGS.logdir, 'ckpt', 'ckpt_{}.pt'.format(step+1)))

            # evaluate
            if FLAGS.eval_step > 0 and (step+1) % FLAGS.eval_step == 0:  ##FLAGS.eval_step > 0 and 
                #net_IS, net_FID, _ = evaluate(net_sampler,DiffusionStep, net_model, features, AE)
                ema_IS, ema_FID, _ = evaluate(ema_sampler,DiffusionStep, ema_model, features, AE)
                metrics = {
                    #'IS': net_IS[0],
                    #'IS_std': net_IS[1],
                    #'FID': net_FID,
                    'IS_EMA': ema_IS[0],
                    'IS_std_EMA': ema_IS[1],
                    'FID_EMA': ema_FID
                }
                pbar.write(
                    "%d/%d " % (step, FLAGS.total_steps) +
                    ", ".join('%s:%.3f' % (k, v) for k, v in metrics.items()))
                for name, value in metrics.items():
                    writer.add_scalar(name, value, step)
                writer.flush()
                with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                    metrics['step'] = step + 1
                    f.write(json.dumps(metrics) + "\n")
    writer.close()


def eval(features, DiffusionStep, AE):
    # model setup
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.truncated_T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt', f'ckpt_{FLAGS.sampling_ckpt_id}.pt'))  ###########added
    print(FLAGS.sampling_ckpt_id)
    if FLAGS.netsample:
        model.load_state_dict(ckpt['net_model'])
        (IS, IS_std), FID, samples = evaluate(sampler, DiffusionStep, model, features, AE)
        print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
        save_image(torch.tensor(samples[:25]),os.path.join(FLAGS.logdir, 'samples.png'),nrow=5) 
    
    if FLAGS.emasample:
        model.load_state_dict(ckpt['ema_model'])
        (IS, IS_std), FID, samples = evaluate(sampler,DiffusionStep,  model, features, AE)
        print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
        save_image(torch.tensor(samples[:16]),os.path.join(FLAGS.logdir, 'samples_ema.png'),nrow=4)
#############################################################################################################        

def interpolation(DiffusionStep, features, AE):
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.truncated_T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    
    print(FLAGS.sampling_ckpt_id)
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt', f'ckpt_{FLAGS.sampling_ckpt_id}.pt'))  ###########added
    model.load_state_dict(ckpt['ema_model'])
    
    nums = 64
    z_all = torch.empty([nums, 1200])
    z1 = features[0].unsqueeze(0)
    z2 = features[100].unsqueeze(0)
    for i in range(nums):
        z_all[i] = (1-i/nums)*z1 + i/nums*z2
    z_gen = z_all.view(z_all.shape[0],-1,1,1).cuda()
    with torch.no_grad():
        x_T = AE.decoder(z_gen)
        x_T = DiffusionStep(x_T, FLAGS.truncated_T)
        #print(x_T.shape)
        batch_images = sampler(x_T.to(device)).cpu()
    print(batch_images.shape)  
    save_image(batch_images,os.path.join(FLAGS.logdir, 'gen_interpolation.jpg'),nrow=8)

#############################################################################################################
def calStepFid(features, DiffusionStep, AE):
    # model setup
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.truncated_T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    
    
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    #fids = {}
    for ckptid in tqdm(range(FLAGS.begin_ckptid, FLAGS.end_ckptid + 1, FLAGS.step_id),
                              desc="processing ckpt"):
        ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt', f'ckpt_{ckptid}.pt'),)

        model.load_state_dict(ckpt['ema_model'])
        (IS, IS_std), FID, samples = evaluate(sampler,DiffusionStep,  model, features, AE)
        #fids[ckptid] = FID
        print("ckpt: {}, fid: {}".format(ckptid, FID))

##################################################################################################added
def truncatedsample(features, DiffusionStep, middle_truncated_T, AE):
    ## model setup
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout)
    Truncatedsampler = TruncatedDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.middle_truncated_T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    #TruncatedStep = truncatedDiffusionStep(FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device) 
    # load model and evaluate
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt', f'ckpt_{FLAGS.sampling_ckpt_id}.pt'))  ###########added
    model.load_state_dict(ckpt['ema_model'])
    result_root_path = './DMAE_CelebAHQ-150'
    gen_imgs_path = os.path.join(result_root_path, 'gen_imgs_{}'.format(middle_truncated_T))
    if not os.path.exists(gen_imgs_path):
        os.mkdir(gen_imgs_path)
    
    start = time.time()
    with torch.no_grad():
        images = []   
        for i in trange(0, min(FLAGS.num_batch_images, int(features.shape[0]/FLAGS.sample_size)), 1, desc='generating images'): #range(int(FLAGS.num_batch_images)):
            z = features.view(features.shape[0],-1,1,1).cuda()
            z1 = z[i*FLAGS.sample_size:(i+1)*FLAGS.sample_size,:,:,:]
            x_T = AE.decoder(z1)
            x_T = DiffusionStep(x_T, middle_truncated_T)
            batch_images = Truncatedsampler(x_T.to(device)).cpu()
            for k in range(FLAGS.sample_size):
                y_gen = batch_images[k,:,:,:]
                save_image(y_gen.cpu(), os.path.join(gen_imgs_path, 'img_{0:07d}_gen.jpg'.format(k+i*FLAGS.sample_size)))

            images.append(batch_images)
        images = torch.cat(images, dim=0).numpy()
    print(images.shape)  
    end = time.time()
    print("generated images %.4f seconds." % (end - start))
    model.train()
    generate_images = min(FLAGS.num_batch_images, int(features.shape[0]/FLAGS.sample_size))*FLAGS.sample_size
    (IS, IS_std), FID = get_inception_and_fid_score(images, FLAGS.fid_cache, num_images=generate_images,
                        use_torch=FLAGS.fid_use_torch, verbose=True)
    
    return (IS, IS_std), FID, images
     

#############################################################################################################

def main(argv):
   
    ###  modified reconstructed images of AE  ###
    data_root_train = FLAGS.data_root_train
    ae_model_path = os.path.join(FLAGS.AE_OT_root, 'ae_models')
    ae_feature = os.path.join(FLAGS.AE_OT_root, 'ae_features.pt')
    gen_feature = os.path.join(FLAGS.AE_OT_root, 'gen_features.mat')          
    AE = autoencoder(FLAGS.dim_z, FLAGS.dim_c, FLAGS.dim_f).to(device)
    ### truncated step ###
    DiffusionStep = truncatedDiffusionStep(FLAGS.beta_1, FLAGS.beta_T, FLAGS.T).to(device)

    for file in os.listdir(ae_model_path): 
        print(file)
        AE.load_state_dict(torch.load(os.path.join(ae_model_path, file)))
    if FLAGS.trainDiffusion:
        print(f'training DiffusionAE')
        features = torch.load(ae_feature)
        feature_indices = torch.randperm(features.size(0))
        features = features[feature_indices]

    else:
        print(f'Reversed sampling generated images')
        feature_dict = sio.loadmat(gen_feature)
        features = feature_dict['features']
        features = torch.from_numpy(features)
        feature_indices = torch.randperm(features.size(0))
        features = features[feature_indices]
        print(features.shape)
    ###------------------------------------------------##
     # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train(features,DiffusionStep, data_root_train, AE)
    if FLAGS.sample:
        eval(features, DiffusionStep, AE)
    if FLAGS.fid:
        calStepFid(features, DiffusionStep, AE) 
    if FLAGS.middle_truncated:
        middle_truncated_T = FLAGS.middle_truncated_T
        (IS, IS_std), FID, samples = truncatedsample(features, DiffusionStep, middle_truncated_T, AE)
        print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
        save_image(torch.tensor(samples[:25]),os.path.join(FLAGS.logdir, 'truncated_samples.png'),nrow=5) 
        #np.savez("./gen_celebahq-{}.npz".format(middle_truncated_T), arr_0=samples)
    if FLAGS.interpolation:
        interpolation(DiffusionStep,features, AE)
    if not FLAGS.train and not FLAGS.sample and not FLAGS.fid and not FLAGS.middle_truncated and not FLAGS.interpolation:
        print('Add --train and/or --sample to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)
