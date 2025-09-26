import pytorch_lightning as pl
import torch
from torch.optim import lr_scheduler, optimizer

import utils
from models import helper


class VPRModel(pl.LightningModule):
    """This is the main model for Visual Place Recognition
    we use Pytorch Lightning for modularity purposes.

    Args:
        pl (_type_): _description_
    """

    def __init__(self,
                 # ---- Backbone
                 backbone_arch='resnet50',
                 backbone_config={},
                 pretrained=True,
                 layers_to_freeze=1,
                 layers_to_crop=[],

                 # ---- Aggregator
                 agg_arch='ConvAP',  # CosPlace, NetVLAD, GeM
                 agg_config={},

                 # ---- Train hyperparameters
                 lr=0.003,
                 optimizer='sgd',
                 weight_decay=1e-3,
                 momentum=0.9,
                 lr_sched='linear',
                 lr_sched_args={
                     'start_factor': 1,
                     'end_factor': 0.2,
                     'total_iters': 4000,
                 },

                 # ----- Loss
                 loss_name='MultiSimilarityLoss',
                 miner_name='MultiSimilarityMiner',
                 miner_margin=0.1,
                 faiss_gpu=False
                 ):
        super().__init__()

        # Backbone
        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop
        self.backbone_config = backbone_config

        # Aggregator
        self.agg_arch = agg_arch
        self.agg_config = agg_config

        # Train hyperparameters
        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.lr_sched = lr_sched
        self.lr_sched_args = lr_sched_args

        # Loss
        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin

        self.save_hyperparameters()  # write hyperparams into a file

        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = []  # we will keep track of the % of trivial pairs/triplets at the loss level

        # for dataset test
        self.test_pred_feat = []
        self.test_dataset = None

        self.faiss_gpu = faiss_gpu

        # ----------------------------------
        # get the backbone and the aggregator
        self.backbone = helper.get_backbone(backbone_arch, backbone_config, pretrained, layers_to_freeze,
                                            layers_to_crop)
        self.aggregator = helper.get_aggregator(agg_arch, agg_config)

    # the forward pass of the lightning model
    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregator(x)
        return x

    # configure the optimizer
    def configure_optimizers(self):
        # optimizer
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.lr,
                                        weight_decay=self.weight_decay,
                                        momentum=self.momentum)
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added to "configure_optimizers()"')

        # scheduler
        if self.lr_sched.lower() == 'multistep':
            scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_sched_args['milestones'],
                                                 gamma=self.lr_sched_args['gamma'])
        elif self.lr_sched.lower() == 'cosine':
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, self.lr_sched_args['T_max'])
        elif self.lr_sched.lower() == 'linear':
            scheduler = lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.lr_sched_args['start_factor'],
                end_factor=self.lr_sched_args['end_factor'],
                total_iters=self.lr_sched_args['total_iters']
            )
        return [optimizer], [scheduler]

    # # configure the optizer step, takes into account the warmup stage
    # def optimizer_step(self, epoch, batch_idx,
    #                    optimizer, optimizer_idx, optimizer_closure,
    #                    on_tpu, using_native_amp, using_lbfgs):
    #     # warm up lr
    #     optimizer.step(closure=optimizer_closure)
    #     self.lr_schedulers().step()

    #  The loss function call (this method will be called at each training iteration)
    def loss_function(self, descriptors, labels):
        # we mine the pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)

            # calculate the % of trivial pairs/triplets
            # which do not contribute in the loss value
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)

        else:  # no online mining
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple:
                # somes losses do the online mining inside (they don't need a miner objet),
                # so they return the loss and the batch accuracy
                # for example, if you are developping a new loss function, you might be better
                # doing the online mining strategy inside the forward function of the loss class,
                # and return a tuple containing the loss value and the batch_accuracy (the % of valid pairs or triplets)
                loss, batch_acc = loss

        # keep accuracy of every batch and later reset it at epoch start
        self.batch_acc.append(batch_acc)
        # log it
        self.log('b_acc', sum(self.batch_acc) /
                 len(self.batch_acc), prog_bar=True, logger=True)
        return loss

    # This is the training step that's executed at each iteration
    def training_step(self, batch, batch_idx):
        places, labels = batch

        # Note that GSVCities yields places (each containing N images)
        # which means the dataloader will return a batch containing BS places
        BS, N, ch, h, w = places.shape

        # reshape places and labels
        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)

        # Feed forward the batch to the model
        descriptors = self(images)  # Here we are calling the method forward that we defined above
        loss = self.loss_function(descriptors, labels)  # Call the loss_function we defined above

        self.log('loss', loss.item(), logger=True)
        return {'loss': loss}

    def on_train_epoch_end(self):
        # we empty the batch_acc list for next epoch
        self.batch_acc = []
        torch.cuda.empty_cache()

    # For validation, we will also iterate step by step over the validation set
    # this is the way Pytorch Lghtning is made. All about modularity, folks.
    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        # calculate descriptors
        descriptors = self(places)
        return descriptors.detach().cpu()

    def validation_epoch_end(self, val_step_outputs):
        """this return descriptors in their order
        depending on how the validation dataset is implemented
        for this project (MSLS val, Pittburg val), it is always references then queries
        [R1, R2, ..., Rn, Q1, Q2, ...]
        """
        dm = self.trainer.datamodule
        # The following line is a hack: if we have only one validation set, then
        # we need to put the outputs in a list (Pytorch Lightning does not do it presently)
        if len(dm.val_datasets) == 1:  # we need to put the outputs in a list
            val_step_outputs = [val_step_outputs]

        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)
            if 'pitts' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.dbStruct.numDb
                num_queries = len(val_dataset) - num_references
                positives = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                # split to ref and queries
                num_references = val_dataset.num_references
                num_queries = len(val_dataset) - num_references
                positives = val_dataset.pIdx
            elif 'vpair' in val_set_name:
                num_references = val_dataset.numDb
                num_queries = val_dataset.numQ
                positives = val_dataset.getPositives()
            elif 'aerialvl' in val_set_name:
                num_references = val_dataset.numDb
                num_queries = val_dataset.numQ
                positives = val_dataset.getPositives()
            elif 'denseuav' in val_set_name:
                num_references = val_dataset.numDb
                num_queries = val_dataset.numQ
                positives = val_dataset.getPositives()
            elif 'aerocities_val' in val_set_name:
                num_references = val_dataset.numDb
                num_queries = val_dataset.numQ
                positives = val_dataset.getPositives()
            elif 'quicksearch' in val_set_name:
                num_references = val_dataset.numDb
                num_queries = val_dataset.numQ
                positives = val_dataset.getPositives()
            elif 'aerocities_he_val' in val_set_name:
                num_references = val_dataset.numDb
                num_queries = val_dataset.numQ
                positives = val_dataset.getPositives()
            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplemented

            r_list = feats[: num_references]
            q_list = feats[num_references:]
            preds_dict = utils.get_validation_recalls(r_list=r_list,
                                                      q_list=q_list,
                                                      k_values=[1, 5, 10, 15, 20, 50, 100],
                                                      gt=positives,
                                                      print_results=True,
                                                      dataset_name=val_set_name,
                                                      faiss_gpu=self.faiss_gpu
                                                      )

            self.log(f'R1', preds_dict[1], prog_bar=False, logger=True)
            self.log(f'R5', preds_dict[5], prog_bar=False, logger=True)
            self.log(f'R10', preds_dict[10], prog_bar=False, logger=True)
            del r_list, q_list, feats, num_references, positives
        print('\n\n')

    def test_step(self, batch, batch_idx):
        x_hat = self(batch[0])  # test dataset return img_tensor and index
        self.test_pred_feat.append(x_hat.detach().cpu())

    def on_test_epoch_start(self):
        print('\n\n')
        self.test_dataset = self.trainer.test_dataloaders.dataset
        print('test dataset:', self.test_dataset.dataset_info)
        print('\n\n')

    def on_test_epoch_end(self):
        test_dataset = self.test_dataset

        feats = torch.cat(self.test_pred_feat, dim=0)
        num_references = test_dataset.numDb
        num_queries = test_dataset.numQ
        positives = test_dataset.getPositives()

        r_list = feats[: num_references]
        q_list = feats[num_references:]

        preds_dict = utils.get_validation_recalls(r_list=r_list,
                                                  q_list=q_list,
                                                  k_values=[1, 5, 10, 15, 20, 50, 100],
                                                  gt=positives,
                                                  eval_dataset=test_dataset,
                                                  print_results=True,
                                                  dataset_name=test_dataset.dataset_info['dataset_name'],
                                                  faiss_gpu=self.faiss_gpu,
                                                  save_retrieval=False
                                                  )

        self.log(f'R1', preds_dict[1], prog_bar=False, logger=True)
        self.log(f'R5', preds_dict[5], prog_bar=False, logger=True)
        self.log(f'R10', preds_dict[10], prog_bar=False, logger=True)


        # delete the processed result
        del r_list, q_list, feats, num_references, positives
        print('\n\n')
