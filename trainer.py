"""
modified from: Multimodal Unsupervised Image-to-Image Translation
https://github.com/NVlabs/MUNIT
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)..
"""
from networks import MsImageDis, MaskGen, MaskGenOld
from utils import weights_init, get_model_list, get_scheduler
import torch
import torch.nn as nn
import os


class MUSIC_Trainer(nn.Module):
    def __init__(self, hyperparameters):
        super(MUSIC_Trainer, self).__init__()
        lr = hyperparameters['lr']
        old_flag = hyperparameters['old_flag']
        # Initiate the networks
        if old_flag == 1:
            self.gen = MaskGenOld(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
            self.style_dim = hyperparameters['gen']['style_dim']
            self.s_a = torch.randn(8, self.style_dim, 1, 1).cuda()
            self.s_b = torch.randn(8, self.style_dim, 1, 1).cuda()
        else:
            self.gen = MaskGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # auto-encoder for domain a
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for domain a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        try:
            enhance = hyperparameters['enhance']
        except KeyError:
            enhance = None

        self.enhance = enhance

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())
        gen_params = list(self.gen.parameters())

        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def masking(self, mask, img):
        if self.enhance:
            mask = mask ** self.enhance
        img = img * 0.5 + 0.5
        masked_image = mask * img
        masked_image = (masked_image - 0.5) * 2
        return masked_image

    def scaled_sum(self, input_1, input_2):
        input_1 = input_1 * 0.5 + 0.5
        input_2 = input_2 * 0.5 + 0.5
        sum_output = input_1 + input_2
        sum_output = torch.clamp(sum_output, 0, 1) # added at 3.yaml
        sum_output = (sum_output - 0.5) * 2
        return sum_output

    def scaled_sub(self, input_1, input_2):
        input_1 = input_1 * 0.5 + 0.5
        input_2 = input_2 * 0.5 + 0.5
        sub_output = input_1 - input_2
        sub_output = torch.clamp(sub_output, 0, 1) # added at 3.yaml
        sub_output = (sub_output - 0.5) * 2
        return sub_output

    def forward(self, x_b):
        self.eval()
        x_ba_mask = self.gen.decode(self.gen.encode(x_b))
        x_ba = self.masking(x_ba_mask, x_b)
        self.train()
        return x_ba

    def gen_update(self, x_a, x_b, hyperparameters):
        self.gen_opt.zero_grad()

        # encode-decode
        x_ba_mask = self.gen.decode(self.gen.encode(x_b))
        x_aa_mask = self.gen.decode(self.gen.encode(x_a))
        x_ba = self.masking(x_ba_mask, x_b)
        x_aa = self.masking(x_aa_mask, x_a)
        # encode again
        x_t = self.scaled_sub(x_b, x_ba)
        x_b_new = self.scaled_sum(x_t, x_a)
        x_b_new_mask = self.gen.decode(self.gen.encode(x_b_new))
        x_ba_new = self.masking(x_b_new_mask, x_b_new)
        x_t_new = self.scaled_sub(x_b_new, x_ba_new)
        # decode twice
        x_baa_mask = self.gen.decode(self.gen.encode(x_ba))
        x_baa = self.masking(x_baa_mask, x_ba)

        # reconstruction loss
        self.loss_gen_recon_x_aa = self.recon_criterion(x_aa, x_a)
        self.loss_gen_recon_x_t = self.recon_criterion(x_t_new, x_t)
        self.loss_gen_recon_x_ba_new = self.recon_criterion(x_ba_new, x_a)
        self.loss_gen_recon_x_baa = self.recon_criterion(x_baa, x_ba)

        # GAN loss
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_b_new)

        # total loss
        self.loss_gen_total = hyperparameters['gan_w_a'] * self.loss_gen_adv_a + \
                              hyperparameters['gan_w_b'] * self.loss_gen_adv_b + \
                              hyperparameters['a2a_w'] * self.loss_gen_recon_x_aa + \
                              hyperparameters['x_t_w'] * self.loss_gen_recon_x_t + \
                              hyperparameters['recon_w'] * self.loss_gen_recon_x_ba_new + \
                              hyperparameters['DTN_w'] * self.loss_gen_recon_x_baa
        self.loss_gen_total.backward()
        self.gen_opt.step()

    def sample(self, x_a, x_b):
        self.eval()

        x_ba, x_aa, x_ba_new = [], [], []
        x_t, x_t_new = [], []
        x_baa, x_b_new = [], []
        for i in range(x_a.size(0)):
            x_ba.append(self.masking(self.gen.decode(self.gen.encode(x_b[i].unsqueeze(0))), x_b[i].unsqueeze(0)))
            x_aa.append(self.masking(self.gen.decode(self.gen.encode(x_a[i].unsqueeze(0))), x_a[i].unsqueeze(0)))
            x_t.append(self.scaled_sub(x_b[i], x_ba[i]))
            x_b_new.append(self.scaled_sum(x_t[i], x_a[i].unsqueeze(0)))
            x_ba_new.append(self.masking(self.gen.decode(self.gen.encode(x_b_new[i])), x_b_new[i]))
            x_t_new.append(self.scaled_sub(x_b_new[i], x_ba_new[i]))
            x_baa.append(self.masking(self.gen.decode(self.gen.encode(x_ba[i])), x_ba[i]))

        x_ba, x_aa, x_ba_new = torch.cat(x_ba), torch.cat(x_aa), torch.cat(x_ba_new)
        x_t, x_t_new = torch.cat(x_t), torch.cat(x_t_new)
        x_baa, x_b_new = torch.cat(x_baa), torch.cat(x_b_new)

        self.train()
        return x_b, x_ba, x_baa, x_a, x_ba_new, x_aa, x_t, x_t_new, x_b_new

    def dis_update(self, x_a, x_b, hyperparameters):
        self.dis_opt.zero_grad()

        x_ba = self.masking(self.gen.decode(self.gen.encode(x_b)), x_b)
        x_t = self.scaled_sub(x_b, x_ba)
        x_b_new = self.scaled_sum(x_t, x_a)

        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_b_new.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w_a'] * self.loss_dis_a + hyperparameters['gan_w_b'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()

    def update_learning_rate(self):
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        self.gen.load_state_dict(state_dict['gen'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'gen': self.gen.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)