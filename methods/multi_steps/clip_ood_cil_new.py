import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import optim
from argparse import ArgumentParser

from backbone.my_clip.model import clip_ood_cil
from methods.multi_steps.finetune_il import Finetune_IL
from utils.toolkit import count_parameters, tensor2numpy
import math
from transformers import CLIPProcessor
import json
from utils.data_manager import DataManager
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp import GradScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, average_precision_score, roc_auc_score
from utils.toolkit import accuracy, cal_bwf, mean_class_recall
import os
import re
os.environ["TOKENIZERS_PARALLELISM"] = "false"

EPSILON = 1e-8

def add_special_args(parser:ArgumentParser) -> ArgumentParser: 
    parser.add_argument("--m_desp", type=bool, default=None, help="Whether to use multiple text descriptions")
    parser.add_argument("--lambd",  type=int, default=None, help='hyperparameter of image to image similarity') 
    parser.add_argument("--test_bs",  type=int, default=128, help='test batch size') 
    parser.add_argument("--is_OOD_test", type=bool, default=None, help="Whether to do OOD testing")
    parser.add_argument("--T",  type=float, default=None, help='Temperature')
    parser.add_argument("--ood_text",  type=str, default=None, help='ood text') 
class CLIP_OOD_CIL_new(Finetune_IL):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self._m_desp = config.m_desp
        self._lambda = config.lambd
        self._is_openset_test = config.is_OOD_test
        self._T = config.T
        self._test_bs = config.test_bs
        self._init_cls = config.init_cls
        self._increment = config.increment
        self._ood_text = config.ood_text
        self._til_record = []
        self._cil_record = []
        self._tid_record = []
        self.task_metric_curve = []
        self.cnn_metric_curve = []
        if self._is_openset_test:
            self._AUC_record = []
            self._FPR95_record = []

        if config.backbone == "clip_vit_b_16_224":
            self._pretrained_weights_path = "pretrain_weights/clip_vit_base_patch16"
        ## 
        self._clip_process = CLIPProcessor.from_pretrained(self._pretrained_weights_path)
        if self._m_desp:
            if config.dataset == 'cifar100_i2t'or config.dataset == 'cifar100_i2t_few_shot':
                desp_json = 'datasets/cifar100_prompts_base.json'
            elif config.dataset == 'imagenetr_i2t':
                desp_json = 'datasets/I2T_Imagenet_r.json'
            elif config.dataset == 'imagenet100_i2t' or config.dataset == 'imagenet100_i2t_new':
                desp_json = 'datasets/imagenet100.json'
            elif config.dataset == 'mini_imagenet100_i2t':
                desp_json = 'datasets/mini_imagenet.json'
            elif config.dataset == 'skin8_i2t':
                # desp_json = 'datasets/skin8_prompt.json'
                desp_json = 'datasets/skin8_prompt_M.json'
            elif config.dataset == 'skin40_i2t':
                # desp_json = 'datasets/skin8_prompt.json'
                desp_json = 'datasets/skin40_M_prompt.json'
        else:
            if config.dataset == 'cifar100_i2t'or config.dataset == 'cifar100_i2t_few_shot':
                desp_json = 'datasets/cifar100_one_prompt.json'
            elif config.dataset == 'imagenetr_i2t':
                desp_json = 'datasets/I2T_imagenet_r_one_prompt.json'
            elif config.dataset == 'skin40_i2t':
                desp_json = 'datasets/skin40_one_prompt.json'
            
        id_class_desp = []
        # with open("datasets/cifar100_prompts_full.json") as f:
        with open(desp_json) as f:
            id_texts = json.load(f)
        # load description
        for i in range(len(id_texts[list(id_texts.keys())[0]])):
            id_class_desp.append([id_texts[label][i] for label in list(id_texts.keys())])

        self._id_text_embeddings = {}
        # tokenizer
        for i in range(len(id_class_desp)):
            self._id_text_embeddings.update({i : self._clip_process.tokenizer(id_class_desp[i],return_tensors='pt', padding=True)})
         
            # OOD_dataset = 'cifar100_i2t_ood'
            # self._data_manager_OOD = DataManager(logger, OOD_dataset, config.img_size, config.split_dataset, 
            #                                     config.shuffle, config.seed, config.init_cls, config.increment, config.use_valid)    


        self._logger.info('Applying CLIP_OOD_CIL (a class incremental method, test with {})'.format(self._incre_type))


    def prepare_model(self):
        self._network = clip_ood_cil(self._logger, self._pretrained_weights_path)
        
        
        self._network.test_mode()
        self._network = self._network.cuda()

    def prepare_task_data(self, data_manager_ID):
        self._cur_task += 1
        self._cur_classes = data_manager_ID.get_task_size(self._cur_task)
        print("self._known_classes", self._known_classes)
        print("self._cur_classes", self._cur_classes)
        self._total_classes = self._known_classes + self._cur_classes
        

        
        self._train_dataset_ID = data_manager_ID.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                source='train', mode='train')
        self._train_dataset_ID_prototype = data_manager_ID.get_dataset(indices=np.arange(self._known_classes, self._total_classes),
                source='train', mode='test')

            
        self._test_dataset = data_manager_ID.get_dataset(indices=np.arange(0, self._total_classes), source='test', mode='test')
        self._test_dataset_fc = data_manager_ID.get_dataset(indices=np.arange(self._known_classes, self._total_classes), source='test', mode='test')
        self._openset_test_dataset = data_manager_ID.get_openset_dataset(known_indices=np.arange(0, self._total_classes), source='test', mode='test')

        self._cur_task_test_samples_num = len(self._test_dataset)

        self._logger.info('Train dataset of ID size: {}'.format(len(self._train_dataset_ID)))
        self._logger.info('Test dataset size: {}'.format(len(self._test_dataset)))
        self._logger.info('Test dataset of current task size: {}'.format(len(self._test_dataset_fc)))

        self._train_loader_ID = DataLoader(self._train_dataset_ID, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers)
        self._train_loader_prototype = DataLoader(self._train_dataset_ID_prototype, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

        self._test_loader = DataLoader(self._test_dataset, batch_size=self._test_bs, shuffle=False, num_workers=self._num_workers)
        self._test_fc_loader = DataLoader(self._test_dataset_fc, batch_size=self._test_bs, shuffle=False, num_workers=self._num_workers)

        self._iters_per_epoch_lora =  math.ceil(len(self._train_dataset_ID)*1.0/self._batch_size)

        self._openset_test_loader = DataLoader(self._openset_test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=self._num_workers)

        self._order = torch.tensor(data_manager_ID._class_order)
        # cal text features
        self._network.eval()
        
        
        with torch.no_grad():
            for i in range(len(self._id_text_embeddings)):
                tmp_text_features = self._network.text_features(self._id_text_embeddings[i]["input_ids"].cuda(), self._id_text_embeddings[i]["attention_mask"].cuda())
                tmp_text_features /= tmp_text_features.norm(p=2, dim=-1, keepdim=True)
                tmp_text_features = tmp_text_features.unsqueeze(0)
                if i==0:
                    text_features = tmp_text_features
                else:
                    text_features = torch.cat((text_features, tmp_text_features), dim=0)

        self._id_text_features = text_features

        task_begin = sum(self._increment_steps[:self._cur_task])
        task_end = task_begin + self._increment_steps[self._cur_task]
        cur_classes = self._order[task_begin : task_end]

        # refer to: FD-Align: Feature Discrimination Alignment for Fine-tuning Pre-Trained Models in Few-Shot Learning
        cur_task_id_text_features = self._id_text_features[:,cur_classes,:].mean(dim=0)

        if self._ood_text:
            # ood text
            self._task_text_features = cur_task_id_text_features / cur_task_id_text_features.norm(p=2, dim=-1, keepdim=True)
            cur_task_ood_text_features = self._id_text_features[:,cur_classes,:].mean(dim=1)
            cur_text_features = torch.concat((cur_task_id_text_features,cur_task_ood_text_features), dim=0)

        else:
            cur_text_features = cur_task_id_text_features


        self._text_features = cur_text_features / cur_text_features.norm(p=2, dim=-1, keepdim=True)

       

    def incremental_train(self):

        # train lora
        self._logger.info("Training current task-special lora with data of current task!")
        self._network = self._network.cuda()
        self._network.train_lora_mode()
        self._logger.info('Trainable params: {}'.format(count_parameters(self._network, True)))
        
        optimizer = self._get_optimizer( self._network.parameters(), self._config, False)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self._epochs*self._iters_per_epoch_lora)
        if self._ood_text:
            self._network = self._train_model(self._network, self._train_loader_ID, optimizer, scheduler,  test_loader=self._test_fc_loader,
                                          text_features = self._text_features, task_text_features = self._task_text_features, task_id=self._cur_task, epochs=self._epochs, note='_')
        else:
            self._network = self._train_model(self._network, self._train_loader_ID, optimizer, scheduler,  test_loader=self._test_fc_loader,
                                          text_features = self._text_features,  task_id=self._cur_task, epochs=self._epochs, note='_')
        
        self._network.test_mode()
        
        # cal prototype
        self._cal_image_prototype(self._network, self._train_loader_prototype, task_id=self._cur_task)

        # save lora and prototype
        self._save_checkpoint('seed{}_task{}_checkpoint.pkl'.format(self._seed, self._cur_task),
                self._network.cpu())

  
    def _train_model(self, model, train_loader, optimizer, scheduler,  test_loader=None, text_features=None, task_text_features = None, task_id=None, epochs=100, note=''):
        task_begin = sum(self._increment_steps[:task_id])
        task_end = task_begin + self._increment_steps[task_id]
        if note != '':
            note += '_'
        self._scaler = GradScaler()
        for epoch in range(epochs):
            model, train_losses = self._epoch_train(model, train_loader, optimizer, scheduler, 
                                                    text_features=text_features, task_begin=task_begin, task_end=task_end, task_id=task_id)
            
            info = ('Task {}, Epoch {}/{} => '.format(task_id, epoch+1, epochs) + ('{} {:.3f}, '* int(len(train_losses)/2)).format(*train_losses))
             
            self._logger.info(info)
        if task_text_features==None:
            task_text_features = text_features
        test_acc = self._epoch_test(model, test_loader,  text_features=task_text_features, task_begin=task_begin, task_end=task_end, task_id=task_id)
        
        info = info + 'test_acc {:.3f}, '.format(test_acc) 
        self._logger.info(info)

        self._til_record.append(test_acc)
        self._logger.info("TIL: {} curve of all task is [\t".format(self._eval_metric) + ("{:2.2f}\t"*len(self._til_record)).format(*self._til_record) + ']')
        return model

    def _epoch_train(self, model, train_loader,  optimizer, scheduler, text_features=None, task_begin=None, task_end=None, task_id=None):
        losses = 0
        clip_losses = 0.
        correct = 0.
        total = 0
        
        model.train()

        for _, inputs, targets in train_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            targets = targets - task_begin
            with autocast():
                image_embeds = model(inputs)
              
            image_features = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            logits_per_image = torch.matmul(image_features.float(), text_features.t()) * logit_scale 
            clip_loss = self._clip_loss(logits_per_image, targets)
            
            loss = clip_loss 


            optimizer.zero_grad()
            self._scaler.scale(loss).backward()
            self._scaler.step(optimizer)
            self._scaler.update()

            
            clip_losses += clip_loss.item()
        
            _, predicted = (logits_per_image.max(1))
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            losses += loss.item()
            
            if scheduler != None:
                scheduler.step()
        train_loss_acc = ['Loss', losses/len(train_loader),  'clip_loss', clip_losses/len(train_loader), 'train_acc', correct / (total+EPSILON)*100]

        return model, train_loss_acc

    def _epoch_test(self, model, test_loader,  text_features=None, task_begin=None, task_end=None, task_id=None):

        correct = 0.
        total = 0
        
        model.eval()
        with torch.no_grad():
            for _, inputs, targets in test_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                targets = targets - task_begin
                with autocast():
                    image_embeds = model(inputs)
                
                image_features = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                logit_scale = model.logit_scale.exp()
                logits_per_image = torch.matmul(image_features.float(), text_features.t()) * logit_scale

                _, predicted = (logits_per_image.max(1))
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        test_acc =correct / (total+EPSILON)*100

        return test_acc
    
    def _cal_image_prototype(self, model, train_loader, task_id=None):
        task_begin = sum(self._increment_steps[:task_id])
        cur_num_classes = self._increment_steps[task_id]
        self.image_prototype = torch.zeros(cur_num_classes, 512).float().cuda()
        model.eval()
        with torch.no_grad():
            for _, inputs, targets in train_loader:
                inputs, targets = inputs.cuda(), targets.cuda()
                targets = targets - task_begin
                with autocast():
                    image_embeds = model(inputs)
                image_features = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                self.image_prototype.scatter_add_(0, targets.unsqueeze(1).expand(-1, image_features.size(1)), image_features.float())
            self.image_prototype /= self.image_prototype.norm(p=2, dim=-1, keepdim=True)

    def eval_task(self):
        # Prepare checkpoints for each stage
        # seed_checkpoint_paths: Save the checkpoint paths obtained after continual learning under each seed
        
        checkpoint_paths = [i for i in os.listdir(self._logdir) if i.endswith('.pkl')]
        chks_path = self._logdir
        if self._config.test_dir:
            checkpoint_paths = [i for i in os.listdir(self._config.test_dir) if i.endswith('.pkl')]
            chks_path = self._config.test_dir
        checkpoint_paths.sort()
        seed_checkpoint_paths = {}
        for path in checkpoint_paths:
            splited_text = path.split('_')
            checkpoint_seed = int(splited_text[0].replace('seed', ''))

            if re.match('task[0-9]+$', splited_text[1]): # for multi_steps checkpoints
                checkpoint_task_id = int(splited_text[1].replace('task', ''))
            else: # for single_step checkpoints
                checkpoint_task_id = 0

            # gather checkpoints with the same random seed into a group
            if checkpoint_seed in seed_checkpoint_paths.keys():
                seed_checkpoint_paths[checkpoint_seed][checkpoint_task_id] = path
            else:
                seed_checkpoint_paths[checkpoint_seed] = {checkpoint_task_id:path}

        chk_paths = seed_checkpoint_paths[self._seed]


        pre_tasks_classes = torch.tensor([sum(self._increment_steps[:i]) for i in range(len(self._increment_steps))]).cuda()
        if self._is_openset_test  and self._cur_task < self._nb_tasks-1:
            self._test_loader = self._openset_test_loader

        for cur_task in range(len(chk_paths)):
            chk_name = chk_paths[cur_task]
            class_num = self._increment_steps[cur_task]
            tmp_checkpoint = torch.load(os.path.join(chks_path, chk_name))
            self._network.load_state_dict(tmp_checkpoint, strict=False)
            self.image_prototype = tmp_checkpoint['image_prototype']
            self.image_prototype = self.image_prototype.cuda()
            self._network = self._network.cuda()
            task_begin = sum(self._increment_steps[:cur_task])
            task_end = task_begin + self._increment_steps[cur_task]
            cur_task_classes = self._order[task_begin:task_end]
            cur_task_id_text_features = self._id_text_features[:,cur_task_classes,:].mean(dim=0)
            cur_task_text_features = cur_task_id_text_features
            cur_task_text_features /= cur_task_text_features.norm(p=2, dim=-1, keepdim=True)
            self._network.eval()
            idx = 0
            with torch.no_grad():
                for _, inputs, targets in self._test_loader:
                    idx = idx + 1
                    inputs, targets = inputs.cuda(), targets.cuda()
                    with autocast():
                        image_embeds = self._network(inputs)
                    image_features = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                    img_img_sim = torch.matmul(image_features.float(), self.image_prototype.t())
                    img_text_sim = torch.matmul(image_features.float(), cur_task_text_features.t())
                    logits_per_image = self._lambda * img_img_sim + img_text_sim
                    # logits_per_image = torch.matmul(image_features.float(), cur_task_text_features.t())
                    ood_scores_per_image, predicted = torch.max(logits_per_image, dim=1, keepdim=True)
                    if idx == 1:
                        cur_task_ood_scores =  ood_scores_per_image
                        cur_task_preds = predicted
                        cur_targets = targets
                        cur_task_logits = logits_per_image
                    else:
                        cur_task_ood_scores = torch.cat((cur_task_ood_scores, ood_scores_per_image), dim=0)
                        cur_task_preds = torch.cat((cur_task_preds, predicted), dim=0)
                        cur_targets = torch.cat((cur_targets, targets), dim=0)
                        cur_task_logits = torch.cat((cur_task_logits, logits_per_image), dim=0)
                    
            if cur_task == 0:
                all_tasks_ood_scores = cur_task_ood_scores
                all_tasks_preds = cur_task_preds
                all_task_logits = cur_task_logits
            else:
                all_tasks_ood_scores = torch.cat((all_tasks_ood_scores, cur_task_ood_scores), dim=1)
                all_tasks_preds = torch.cat((all_tasks_preds, cur_task_preds), dim=1)
                all_task_logits = torch.cat((all_task_logits, cur_task_logits), dim=1)
            
            # self._network = self._network.cpu()

        task_id_per_image = torch.argmax(all_tasks_ood_scores, dim=1)
        task_pred_per_image = all_tasks_preds[range(len(all_tasks_ood_scores)), task_id_per_image]
        all_tasks_preds = pre_tasks_classes[task_id_per_image] + task_pred_per_image
        cur_all_preds = all_tasks_preds[:self._cur_task_test_samples_num]
        cur_total = cur_targets[:self._cur_task_test_samples_num]
        if self._eval_metric == "acc":
            total = cur_total.size(0)
            correct = cur_all_preds.eq(cur_total).sum().item()
            t_id_correct = task_id_per_image[:self._cur_task_test_samples_num].eq(cur_total//self._increment).sum().item()
            t_id_acc = t_id_correct/total*100
            acc = correct/total*100
            self._logger.info("After training the {}th task, the accuracy of the test set: {}".format(len(chk_paths)-1, acc))
            self._cil_record.append(acc)
            self._tid_record.append(t_id_acc)
            self._logger.info("Task ID: {} curve of all task is [\t".format(self._eval_metric) + ("{:2.2f}\t"*len(self._tid_record)).format(*self._tid_record) + ']')
        elif self._eval_metric == "mcr":
            cm = confusion_matrix(cur_total.cpu(), cur_all_preds.cpu())
            right_of_class = np.diag(cm)
            num_of_class = cm.sum(axis=1)
            task_size = cm.shape[0]
            mcr = np.around((right_of_class*100 / (num_of_class+1e-8)).sum() / task_size, decimals=2)
            self._logger.info("After training the {}th task, the mean class recall of the test set: {}".format(len(chk_paths)-1, mcr))
            self._cil_record.append(mcr)
        else:
            assert self._eval_metric != "mcr" and self._eval_metric != "acc", "Please enter the correct eval metric (mcr or acc)!"  
        
        if self._is_openset_test and self._cur_task < self._nb_tasks-1:
            labels_list = [1]*self._cur_task_test_samples_num
            labels_list.extend([0]*(len(self._openset_test_dataset) - self._cur_task_test_samples_num))
            scores = all_task_logits
            if self._T==0 or self._T == None:
                scores_softmax = scores
            else:
                scores_softmax = torch.softmax(scores.float()/self._T, dim=1)
            max_scores = torch.max(scores_softmax, dim=1)[0]
            scores_list = max_scores.tolist()
            rocauc = roc_auc_score(labels_list, scores_list)
            fpr, tpr, thresholds = roc_curve(labels_list, scores_list)
            fpr95_idx = np.where(tpr>=0.95)[0]
            fpr95 = fpr[fpr95_idx[0]]
            self._AUC_record.append(rocauc*100)
            self._FPR95_record.append(fpr95*100)
            
            self._logger.info("AUC curve of all stages is [\t" + ("{:2.2f}\t"*len(self._AUC_record)).format(*self._AUC_record) + ']')
            self._logger.info("FPR95 curve of all stages is [\t" + ("{:2.2f}\t"*len(self._FPR95_record)).format(*self._FPR95_record) + ']')

        self._logger.info("CIL: {} curve of all task is [\t".format(self._eval_metric) + ("{:2.2f}\t"*len(self._cil_record)).format(*self._cil_record) + ']')
        
    

    def _clip_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)
    
    def _ce_loss(self, logits, labels):
        return F.cross_entropy(logits, labels)

        
    def after_task(self):
        self._known_classes = self._total_classes
        


    def _save_checkpoint(self, filename, model=None):
        save_path = os.path.join(self._logdir, filename)

        # save lora and final fc
        my_state_dict = model.state_dict()
        model_state_dict = {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k } # or 'final_fc'

        # save model config
        model_state_dict.update({'config':self._config.get_parameters_dict()})

        # save current task
        model_state_dict.update({'task_id':self._cur_task})

        # save prototype
        model_state_dict.update({'image_prototype':self.image_prototype})

        torch.save(model_state_dict, save_path)
        self._logger.info('checkpoint saved at: {}'.format(save_path))


    def store_samples(self):
        pass
        

    def eval_cil_task(self):
        # initialize the task_metric_curve
        if len(self.task_metric_curve) == 0:
            self.task_metric_curve = np.zeros((self._nb_tasks, self._nb_tasks))
        self._logger.info(50*"-")
        self._logger.info("log {} of every task".format(self._eval_metric))
        self._logger.info(50*"-")

        task_id = self._cur_task
        # Prepare checkpoints for each stage
        # seed_checkpoint_paths: Save the checkpoint paths obtained after continual learning under each seed
        
        checkpoint_paths = [i for i in os.listdir(self._logdir) if i.endswith('.pkl')]
        chks_path = self._logdir
        if self._config.test_dir:
            checkpoint_paths = [i for i in os.listdir(self._config.test_dir) if i.endswith('.pkl')]
            chks_path = self._config.test_dir
        checkpoint_paths.sort()
        seed_checkpoint_paths = {}
        for path in checkpoint_paths:
            splited_text = path.split('_')
            checkpoint_seed = int(splited_text[0].replace('seed', ''))

            if re.match('task[0-9]+$', splited_text[1]): # for multi_steps checkpoints
                checkpoint_task_id = int(splited_text[1].replace('task', ''))
            else: # for single_step checkpoints
                checkpoint_task_id = 0

            # gather checkpoints with the same random seed into a group
            if checkpoint_seed in seed_checkpoint_paths.keys():
                seed_checkpoint_paths[checkpoint_seed][checkpoint_task_id] = path
            else:
                seed_checkpoint_paths[checkpoint_seed] = {checkpoint_task_id:path}

        chk_paths = seed_checkpoint_paths[self._seed]


        pre_tasks_classes = torch.tensor([sum(self._increment_steps[:i]) for i in range(len(self._increment_steps))]).cuda()

        if self._is_openset_test  and self._cur_task < self._nb_tasks-1:
            self._test_loader = self._openset_test_loader
        ##
        for cur_task in range(task_id+1):
            chk_name = chk_paths[cur_task]
            class_num = self._increment_steps[cur_task]
            tmp_checkpoint = torch.load(os.path.join(chks_path, chk_name))
            self._network.load_state_dict(tmp_checkpoint, strict=False)

            self.image_prototype = tmp_checkpoint['image_prototype']
            self.image_prototype = self.image_prototype.cuda()            
            self._network = self._network.cuda()
            task_begin = sum(self._increment_steps[:cur_task])
            task_end = task_begin + self._increment_steps[cur_task]
            cur_task_classes = self._order[task_begin:task_end]
            cur_task_id_text_features = self._id_text_features[:,cur_task_classes,:].mean(dim=0)
            cur_task_text_features = cur_task_id_text_features
            cur_task_text_features /= cur_task_text_features.norm(p=2, dim=-1, keepdim=True)


            self._network.eval()
            self._network._training_mode = 'test'
            idx = 0
            with torch.no_grad():
                for _, inputs, targets in self._test_loader:
                    idx = idx + 1
                    inputs, targets = inputs.cuda(), targets.cuda()
                    with autocast():
                        image_embeds = self._network(inputs)
                    image_features = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                    img_img_sim = torch.matmul(image_features.float(), self.image_prototype.t())
                    img_text_sim = torch.matmul(image_features.float(), cur_task_text_features.t())
                    logits_per_image = self._lambda * img_img_sim + img_text_sim
                    ood_scores_per_image, predicted = torch.max(logits_per_image, dim=1, keepdim=True)
                    if idx == 1:
                        cur_task_ood_scores =  ood_scores_per_image
                        cur_task_preds = predicted
                        cur_targets = targets
                        cur_task_logits = logits_per_image
                    else:
                        cur_task_ood_scores = torch.cat((cur_task_ood_scores, ood_scores_per_image), dim=0)
                        cur_task_preds = torch.cat((cur_task_preds, predicted), dim=0)
                        cur_targets = torch.cat((cur_targets, targets), dim=0)
                        cur_task_logits = torch.cat((cur_task_logits, logits_per_image), dim=0)
            if cur_task == 0:
                all_tasks_ood_scores = cur_task_ood_scores
                all_tasks_preds = cur_task_preds
                all_task_logits = cur_task_logits
            else:
                all_tasks_ood_scores = torch.cat((all_tasks_ood_scores, cur_task_ood_scores), dim=1)
                all_tasks_preds = torch.cat((all_tasks_preds, cur_task_preds), dim=1)
                all_task_logits = torch.cat((all_task_logits, cur_task_logits), dim=1)

            self._network = self._network.cpu()
            # self._network = self._network.cpu()
        
        task_id_per_image = torch.argmax(all_tasks_ood_scores, dim=1)
        task_pred_per_image = all_tasks_preds[range(len(all_tasks_ood_scores)), task_id_per_image]
        all_tasks_preds = pre_tasks_classes[task_id_per_image] + task_pred_per_image
        cur_all_preds = all_tasks_preds[:self._cur_task_test_samples_num]
        cur_total = cur_targets[:self._cur_task_test_samples_num]
        if self._eval_metric == "acc":
            cnn_total, cnn_task = accuracy(tensor2numpy(cur_all_preds), tensor2numpy(cur_total), self._total_classes, self._increment_steps)
            self.cnn_metric_curve.append(cnn_total)
            self._logger.info("CNN : {} curve of all task is [\t".format(self._eval_metric) + 
            ("{:2.2f}\t"*len(self.cnn_metric_curve)).format(*self.cnn_metric_curve) + ']')
            for i in range(len(cnn_task)):
                self.task_metric_curve[i][self._cur_task] = cnn_task[i]
                self._logger.info("CNN : task {} {} curve is [\t".format(i, self._eval_metric)+
                        ("{:2.2f}\t"*len(cnn_task)).format(*self.task_metric_curve[i][:len(cnn_task)].tolist()) + ']')
            self._logger.info("CNN : Backward Transfer: {:.2f}".format(cal_bwf(self.task_metric_curve, self._cur_task)))

            total = cur_total.size(0)
            correct = cur_all_preds.eq(cur_total).sum().item()
            t_id_correct = task_id_per_image[:self._cur_task_test_samples_num].eq(cur_total//self._increment).sum().item()
            t_id_acc = t_id_correct/total*100
            acc = correct/total*100
            self._logger.info("After training the {}th task, the accuracy of the test set: {}".format(task_id, acc))
            self._cil_record.append(acc)
            self._tid_record.append(t_id_acc)
            self._logger.info("Task ID: {} curve of all task is [\t".format(self._eval_metric) + ("{:2.2f}\t"*len(self._tid_record)).format(*self._tid_record) + ']')
        elif self._eval_metric == "mcr":
            cnn_total, cnn_task = mean_class_recall(tensor2numpy(cur_all_preds), tensor2numpy(cur_total), self._total_classes, self._increment_steps)
            self.cnn_metric_curve.append(cnn_total)
            self._logger.info("CNN : {} curve of all task is [\t".format(self._eval_metric) + 
            ("{:2.2f}\t"*len(self.cnn_metric_curve)).format(*self.cnn_metric_curve) + ']')
            for i in range(len(cnn_task)):
                self.task_metric_curve[i][self._cur_task] = cnn_task[i]
                self._logger.info("CNN : task {} {} curve is [\t".format(i, self._eval_metric)+
                        ("{:2.2f}\t"*len(cnn_task)).format(*self.task_metric_curve[i][:len(cnn_task)].tolist()) + ']')
            self._logger.info("CNN : Backward Transfer: {:.2f}".format(cal_bwf(self.task_metric_curve, self._cur_task)))

            cm = confusion_matrix(cur_total.cpu(), cur_all_preds.cpu())
            right_of_class = np.diag(cm)
            num_of_class = cm.sum(axis=1)
            task_size = cm.shape[0]
            mcr = np.around((right_of_class*100 / (num_of_class+1e-8)).sum() / task_size, decimals=2)
            self._logger.info("After training the {}th task, the mean class recall of the test set: {}".format(task_id, mcr))
            self._cil_record.append(mcr)
        else:
            assert self._eval_metric != "mcr" and self._eval_metric != "acc", "Please enter the correct eval metric (mcr or acc)!"  
        
        if self._is_openset_test and self._cur_task < self._nb_tasks-1:
            labels_list = [1]*self._cur_task_test_samples_num
            labels_list.extend([0]*(len(self._openset_test_dataset) - self._cur_task_test_samples_num))
            scores = all_task_logits
            if self._T==0 or self._T == None:
                scores_softmax = scores
            else:
                scores_softmax = torch.softmax(scores.float()/self._T, dim=1)
            max_scores = torch.max(scores_softmax, dim=1)[0]
            scores_list = max_scores.tolist()
            rocauc = roc_auc_score(labels_list, scores_list)
            fpr, tpr, thresholds = roc_curve(labels_list, scores_list)
            fpr95_idx = np.where(tpr>=0.95)[0]
            fpr95 = fpr[fpr95_idx[0]]
            self._AUC_record.append(rocauc*100)
            self._FPR95_record.append(fpr95*100)
            
            self._logger.info("AUC curve of all stages is [\t" + ("{:2.2f}\t"*len(self._AUC_record)).format(*self._AUC_record) + ']')
            self._logger.info("FPR95 curve of all stages is [\t" + ("{:2.2f}\t"*len(self._FPR95_record)).format(*self._FPR95_record) + ']')
        
        self._logger.info("CIL: {} curve of all task is [\t".format(self._eval_metric) + ("{:2.2f}\t"*len(self._cil_record)).format(*self._cil_record) + ']')
        

    def cal_logits(self):
        task_id = self._cur_task
        # Prepare checkpoints for each stage
        # seed_checkpoint_paths: Save the checkpoint paths obtained after continual learning under each seed
        
        checkpoint_paths = [i for i in os.listdir(self._logdir) if i.endswith('.pkl')]
        chks_path = self._logdir
        if self._config.test_dir:
            checkpoint_paths = [i for i in os.listdir(self._config.test_dir) if i.endswith('.pkl')]
            chks_path = self._config.test_dir
        checkpoint_paths.sort()
        seed_checkpoint_paths = {}
        for path in checkpoint_paths:
            splited_text = path.split('_')
            checkpoint_seed = int(splited_text[0].replace('seed', ''))

            if re.match('task[0-9]+$', splited_text[1]): # for multi_steps checkpoints
                checkpoint_task_id = int(splited_text[1].replace('task', ''))
            else: # for single_step checkpoints
                checkpoint_task_id = 0

            # gather checkpoints with the same random seed into a group
            if checkpoint_seed in seed_checkpoint_paths.keys():
                seed_checkpoint_paths[checkpoint_seed][checkpoint_task_id] = path
            else:
                seed_checkpoint_paths[checkpoint_seed] = {checkpoint_task_id:path}

        chk_paths = seed_checkpoint_paths[self._seed]


        pre_tasks_classes = torch.tensor([sum(self._increment_steps[:i]) for i in range(len(self._increment_steps))]).cuda()
        ##
        for cur_task in range(task_id+1):
            chk_name = chk_paths[cur_task]
            class_num = self._increment_steps[cur_task]
            tmp_checkpoint = torch.load(os.path.join(chks_path, chk_name))
            self._network.load_state_dict(tmp_checkpoint, strict=False)
            self.image_prototype = tmp_checkpoint['image_prototype']
            self.image_prototype = self.image_prototype.cuda()
            self._network = self._network.cuda()
            task_begin = sum(self._increment_steps[:cur_task])
            task_end = task_begin + self._increment_steps[cur_task]
            cur_task_classes = self._order[task_begin:task_end]
            cur_task_id_text_features = self._id_text_features[:,cur_task_classes,:].mean(dim=0)
            cur_task_text_features = cur_task_id_text_features
            cur_task_text_features /= cur_task_text_features.norm(p=2, dim=-1, keepdim=True)


            self._network.eval()
            self._network._training_mode = 'test'
            idx = 0
            with torch.no_grad():
                for _, inputs, targets in self._test_loader:
                    idx = idx + 1
                    inputs, targets = inputs.cuda(), targets.cuda()
                    with autocast():
                        image_embeds = self._network(inputs)
                    image_features = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                    img_img_sim = torch.matmul(image_features.float(), self.image_prototype.t())
                    img_text_sim = torch.matmul(image_features.float(), cur_task_text_features.t())
                    logits_per_image = self._lambda * img_img_sim + img_text_sim
                    
                    if idx == 1:
                        cur_task_ood_logits =  logits_per_image
                    else:
                        cur_task_ood_logits = torch.cat((cur_task_ood_logits, logits_per_image), dim=0)
            if cur_task == 0:
                all_tasks_ood_scores = cur_task_ood_logits
            else:
                all_tasks_ood_scores = torch.cat((all_tasks_ood_scores, cur_task_ood_logits), dim=1)
            self._network = self._network.cpu()
        filename = ('seed{}_task{}_logits.pt'.format(self._seed, self._cur_task))
        save_path = os.path.join(self._logdir, filename)
        torch.save(all_tasks_ood_scores.cpu(), save_path)
        self._logger.info('logits saved at: {}'.format(save_path))
        
    def eval_cil_task_features(self):
        # initialize the task_metric_curve
        if len(self.task_metric_curve) == 0:
            self.task_metric_curve = np.zeros((self._nb_tasks, self._nb_tasks))
        self._logger.info(50*"-")
        self._logger.info("log {} of every task".format(self._eval_metric))
        self._logger.info(50*"-")

        task_id = self._cur_task
        # Prepare checkpoints for each stage
        # seed_checkpoint_paths: Save the checkpoint paths obtained after continual learning under each seed
        
        checkpoint_paths = [i for i in os.listdir(self._logdir) if i.endswith('.pkl')]
        chks_path = self._logdir
        if self._config.test_dir:
            checkpoint_paths = [i for i in os.listdir(self._config.test_dir) if i.endswith('.pkl')]
            chks_path = self._config.test_dir
        checkpoint_paths.sort()
        seed_checkpoint_paths = {}
        for path in checkpoint_paths:
            splited_text = path.split('_')
            checkpoint_seed = int(splited_text[0].replace('seed', ''))

            if re.match('task[0-9]+$', splited_text[1]): # for multi_steps checkpoints
                checkpoint_task_id = int(splited_text[1].replace('task', ''))
            else: # for single_step checkpoints
                checkpoint_task_id = 0

            # gather checkpoints with the same random seed into a group
            if checkpoint_seed in seed_checkpoint_paths.keys():
                seed_checkpoint_paths[checkpoint_seed][checkpoint_task_id] = path
            else:
                seed_checkpoint_paths[checkpoint_seed] = {checkpoint_task_id:path}

        chk_paths = seed_checkpoint_paths[self._seed]


        pre_tasks_classes = torch.tensor([sum(self._increment_steps[:i]) for i in range(len(self._increment_steps))]).cuda()

        if self._is_openset_test  and self._cur_task < self._nb_tasks-1:
            self._test_loader = self._openset_test_loader
        ##
        for cur_task in range(task_id+1):
            chk_name = chk_paths[cur_task]
            class_num = self._increment_steps[cur_task]
            tmp_checkpoint = torch.load(os.path.join(chks_path, chk_name))
            self._network.load_state_dict(tmp_checkpoint, strict=False)
            self.image_prototype = tmp_checkpoint['image_prototype']
            self.image_prototype = self.image_prototype.cuda()            
            self._network = self._network.cuda()
            task_begin = sum(self._increment_steps[:cur_task])
            task_end = task_begin + self._increment_steps[cur_task]
            cur_task_classes = self._order[task_begin:task_end]
            cur_task_id_text_features = self._id_text_features[:,cur_task_classes,:].mean(dim=0)
            cur_task_text_features = cur_task_id_text_features
            cur_task_text_features /= cur_task_text_features.norm(p=2, dim=-1, keepdim=True)


            self._network.eval()
            self._network._training_mode = 'test'
            idx = 0
            with torch.no_grad():
                for _, inputs, targets in self._test_loader:
                    idx = idx + 1
                    inputs, targets = inputs.cuda(), targets.cuda()
                    with autocast():
                        image_embeds = self._network(inputs)
                    image_features = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                    img_img_sim = torch.matmul(image_features.float(), self.image_prototype.t())
                    img_text_sim = torch.matmul(image_features.float(), cur_task_text_features.t())
                    logits_per_image = self._lambda * img_img_sim + img_text_sim
                    ood_scores_per_image, predicted = torch.max(logits_per_image, dim=1, keepdim=True)
                    if idx == 1:
                        cur_task_ood_scores =  ood_scores_per_image
                        cur_task_preds = predicted
                        cur_targets = targets
                        cur_task_logits = logits_per_image
                        all_features = image_features
                    else:
                        cur_task_ood_scores = torch.cat((cur_task_ood_scores, ood_scores_per_image), dim=0)
                        cur_task_preds = torch.cat((cur_task_preds, predicted), dim=0)
                        cur_targets = torch.cat((cur_targets, targets), dim=0)
                        cur_task_logits = torch.cat((cur_task_logits, logits_per_image), dim=0)
                        all_features = torch.cat((all_features, image_features), dim=0)
            

            self._network = self._network.cpu()
            # self._network = self._network.cpu()
        torch.save(all_features, 'all_features_no_ood.pt')
        torch.save(cur_targets, 'all_targets_no_ood.pt')
        
        