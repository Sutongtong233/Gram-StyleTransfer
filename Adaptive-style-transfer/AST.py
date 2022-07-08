from torch import nn
import torch
import torch.nn.functional as F
from networks import *

class AST(nn.modules):
    def __init__(self, options):
        super(AST, self).__init__()
        # build models
        self.encoder = encoder(options)
        self.decoder = decoder(options)
        self.dicriminator = discriminator(options)
        self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.mse = nn.MSELoss(reduction='mean')
        # optimizers and variables
        dis_params = list(self.discriminator.parameters())
        gen_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad], lr=options.lr, betas=(0.5, 0.999), weight_decay=0.0001, amsgrad=True)
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad], lr=options.lr, betas=(0.5, 0.999), weight_decay=0.0001, amsgrad=True)
        # loss initialization
        self.gener_loss = torch.tensor(0.)
        self.discr_loss = torch.tensor(0.)
        # TODO: weight init
    def forward():
        return
    def update(self, options, batch_art, batch_content, whether_update_generator):
        self.dis_opt.zero_grad()
        self.gen_opt.zero_grad()
        batch_output = self.decoder(self.encoder(batch_content))
        batch_output_preds = self.discriminator(batch_output)
        batch_art_preds = self.discriminator(batch_art)
        batch_content_preds = self.discriminator(batch_content)
        if whether_update_generator: # TODO: here alpha
            g_acc = self.gener_update(options, batch_content, batch_output, batch_output_preds)
        d_acc = self.dis_update(options, batch_art_preds, batch_content_preds, batch_output_preds)
        self.gen_opt.step()
        self.dis_opt.step()
        return d_acc

    def gener_update(self, options, batch_content, batch_output, batch_output_preds):
        # calculate three parts of loss
        loss_D = self.loss(torch.ones_like(batch_output_preds['pred_7']), batch_output_preds['pred_7']) # TODO: multi layers? scalar output
        loss_C = self.mse(self.encoder(batch_output), self.encoder(batch_content))
        loss_T = self.mse(F.avg_pool2d(batch_output,kernel_size=10,stride=1), F.avg_pool2d(batch_content,kernel_size=10,stride=1))  # Transform==average pool
        self.gener_loss = options.D_weight * loss_D + options.T_weight * loss_T + options.C_loss * loss_C
        self.gener_loss.backward(retain_graph=True)
        return self.gener_loss
    def discr_update(self, options, batch_art_preds, batch_content_preds, batch_output_preds):
        # calculate only one part of loss Loss_D, with two terms
        loss_D_art = self.loss(torch.ones_like(batch_art_preds['pred_7']), batch_art_preds['pred_7']) # real: D(y)->1
        loss_D_gen = self.loss(torch.zeros_like(batch_output_preds['pred_7']), batch_output_preds['pred_7']) # fake: D->0
        loss_D_content = self.loss(torch.zeros_like(batch_content_preds['pred_7']), batch_content_preds['pred_7']) # TODO: not mentioned
        self.discr_loss = options.discr_weight * (loss_D_art+loss_D_gen+loss_D_content)
        self.discr_loss.backward(retain_graph=True)
        return self.discr_loss
    def sample(self, test):
        self.eval()
        with torch.no_grad():
            y = self.decoder(self.encoder(test))
        self.train()
        return test, y






        