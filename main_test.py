from utils.config import Config
from utils.logger import MyLogger
import copy
import os
# os.environ['CUDA_VISIBLE_DEVICES']= '1'
import torch
from utils.data_manager import DataManager
import methods
import numpy as np
import random
import re

os.environ['WANDB_MODE']='offline'

def set_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    config = Config()
    # os.environ['CUDA_VISIBLE_DEVICES']=config.device

    # test model with checkpoint
    
    seed_list = copy.deepcopy(config.seed)
    try:
        for seed in seed_list:
            temp_config = copy.deepcopy(config)
            temp_config.seed = seed
            logger = MyLogger(temp_config)
            logger.info('seed list ready to apply: {}'.format(seed_list))
            set_random(seed)
            data_manager = DataManager(logger, temp_config.dataset, temp_config.img_size, temp_config.split_dataset,
                    temp_config.shuffle, temp_config.seed, temp_config.init_cls, temp_config.increment, temp_config.use_valid)
            temp_config.update({'total_class_num':data_manager.total_classes, 'nb_tasks':data_manager.nb_tasks,
                        'increment_steps':data_manager.increment_steps, 'img_size':data_manager.img_size})
            temp_config.print_config(logger)

            logger.init_visual_log(temp_config)

            trainer = methods.get_trainer(logger, temp_config)

            while trainer.cur_taskID < data_manager.nb_tasks - 1:
                trainer.prepare_model()
                trainer.prepare_task_data(data_manager)
                # trainer.incremental_train()
                # trainer.store_samples()
                
                # trainer.task_test()
                trainer.eval_cil_task()
                # trainer.eval_cil_task_features()
                # trainer.cal_logits()
                trainer.after_task()
                if temp_config.method_type == 'single_step':
                    break
                
            logger.info('='*10 + 'Test Finished !' + '='*10)
            logger.info(' ')

            trainer.release()
            logger.release_handlers()
            del trainer
            torch.cuda.empty_cache()

    except Exception as e:
        logger.error(e, exc_info=True, stack_info=True)
        logger.release_handlers()
    except KeyboardInterrupt as e:
        logger.error(e, exc_info=True, stack_info=True)
        logger.release_handlers()
