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
from modelsDM.diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler, truncatedDiffusionStep
from modelsDM.model import UNet
from score.both import get_inception_and_fid_score
import numpy as np
from modelsAE.AE_celeba import autoencoder
from utils.P_loader import P_loader

FLAGS = flags.FLAGS  
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('sample', False, help='load ckpt.pt and evaluate FID and IS')
flags.DEFINE_bool('fid', False, help='load ckpt.pt and calculate the fid for each ckpt')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 4, 4], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [2], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value') ###
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_integer('truncated_T', 100, help='total diffusion steps') ############################ --eval modified 
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 1e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1e-3, help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 300000, help='total training steps')
flags.DEFINE_integer('img_size', 64, help='image size')
flags.DEFINE_integer('warmup', 20000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 64, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
flags.DEFINE_bool('resume_training', False, help='resume training') 
# Logging & Sampling
flags.DEFINE_string('logdir', './DMAE_celeba-2-100/DiffusionAEOT', help='log directory') ############################## --eval modified 
flags.DEFINE_integer('sample_size', 100, "sampling size of images")
flags.DEFINE_integer('sample_step', 5000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 5000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 5000, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_batch_images', 10, help='the number of generated images for evaluation') ############################### --eval modified 
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
# traindiffusionAE_OT
flags.DEFINE_string('data_root_train', '/home/ubuntu/workspace/data/CelebA_64/train', help='root of train data')
flags.DEFINE_string('AE_OT_root', './celeba-2/', help='features of AE')
flags.DEFINE_integer('dim_z', 1024, help='dimension of feature in latent space')
flags.DEFINE_integer('dim_c', 3, help='input image number of channels')
flags.DEFINE_integer('dim_f', 48, help='number of features in first layer of AE')
flags.DEFINE_integer('sampling_ckpt_id', 120000, help='selected the best ckpt for sampling') ############################### --eval modified 
flags.DEFINE_bool('trainDiffusion', False, help='Training diffusion model need disrupting the latent codes of AE')
flags.DEFINE_string('fid_cache', './stats/celeba64.train.npz', help='FID cache')
flags.DEFINE_integer('begin_ckptid', 100, help='selected the start ckptid for sampling')
flags.DEFINE_integer('end_ckptid', 100, help='selected the end ckptid for sampling')
flags.DEFINE_integer('step_id', 100, help='selected the end ckptid for sampling')
 
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
        for x, y, _ in iter(dataloader): 
            yield x


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup

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

            batch_images = sampler(x_T.to(device)).cpu()
            images.append(batch_images)
        images = torch.cat(images, dim=0).numpy()
    
    np.savez("./gen_celeba-{}.npz".format(FLAGS.truncated_T), arr_0=images)
    print(images.shape)  
    end = time.time()
    print("generated images %.4f seconds." % (end - start))
    model.train()
    generate_images = min(FLAGS.num_batch_images, int(features.shape[0]/FLAGS.sample_size))*FLAGS.sample_size
    (IS, IS_std), FID = get_inception_and_fid_score(images, FLAGS.fid_cache, num_images=generate_images,
                        use_torch=FLAGS.fid_use_torch, verbose=True)
 
    return (IS, IS_std), FID, images


def train(features,DiffusionStep, data_root_train, AE):
   
    # dataset
    img_transform = transforms.Compose([ 
                transforms.RandomHorizontalFlip(),  
                transforms.ToTensor(),
            ])
    dataset = P_loader(root=data_root_train, transform=img_transform)    
    
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
    feature_indices = torch.randperm(features.size(0))
    features = features[feature_indices]
    z = features.view(features.size(0),-1,1,1).cuda()
    z_batch = z[0:FLAGS.sample_size,:,:,:]
    with torch.no_grad():
        x_T = AE.decoder(z_batch)
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
                    nrow=10)
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
                net_IS, net_FID, _ = evaluate(net_sampler,DiffusionStep, net_model, features, AE)
                ema_IS, ema_FID, _ = evaluate(ema_sampler,DiffusionStep, ema_model, features, AE)
                metrics = {
                    'IS': net_IS[0],
                    'IS_std': net_IS[1],
                    'FID': net_FID,
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
    print(FLAGS.sampling_ckpt_id)
    ckpt = torch.load(os.path.join(FLAGS.logdir, 'ckpt', f'ckpt_{FLAGS.sampling_ckpt_id}.pt'))  ###########added
    model.load_state_dict(ckpt['net_model'])
    '''(IS, IS_std), FID, samples = evaluate(sampler, DiffusionStep, model, features, AE)
    print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:100]),
        os.path.join(FLAGS.logdir, 'samples.png'),
        nrow=10)'''

    model.load_state_dict(ckpt['ema_model'])
    (IS, IS_std), FID, samples = evaluate(sampler,DiffusionStep,  model, features, AE)
    print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:100]),
        os.path.join(FLAGS.logdir, 'samples_ema.png'),
        nrow=10)
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

    else:
        print(f'Reversed sampling generated images')
        feature_dict = sio.loadmat(gen_feature)
        features = feature_dict['features']
        features = torch.from_numpy(features)
    ###------------------------------------------------##
     # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        train(features,DiffusionStep, data_root_train, AE)
    if FLAGS.sample:
        eval(features, DiffusionStep, AE)
    if FLAGS.fid:
        calStepFid(features, DiffusionStep, AE) 
    if not FLAGS.train and not FLAGS.sample and not FLAGS.fid:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)
