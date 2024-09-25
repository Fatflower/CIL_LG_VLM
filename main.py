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
import warnings
warnings.filterwarnings("ignore")

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
    if config.checkpoint_dir:
        checkpoint_paths = [i for i in os.listdir(config.checkpoint_dir) if i.endswith('.pkl')]
        checkpoint_paths.sort()

        # statistic checkpoints with different seed
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

        # resume or test
        try:
            # you can set the value of config.seed in the terminal to only test specific seed in the checkpoint folder
            # if you do not set the config.seed, the program will test all the saved checkpoints in the checkpoint folder
            if config.seed is None:
                seed_list = seed_checkpoint_paths.keys()
            else:
                seed_list = config.seed

            for seed in seed_list:
                # prepare temp config
                temp_config = copy.deepcopy(config)
                temp_config.seed = seed
                temp_config.checkpoint_names = list(seed_checkpoint_paths[seed].values())
                start_first_key = list(seed_checkpoint_paths[seed].keys())[0]
                task_checkpoint = torch.load(os.path.join(config.checkpoint_dir, seed_checkpoint_paths[seed][start_first_key]))
                temp_config.load_saved_config(task_checkpoint['config'])
                temp_config.save_models = False

                logger = MyLogger(temp_config)

                set_random(seed)
                data_manager = DataManager(logger, temp_config.dataset, temp_config.img_size, temp_config.split_dataset,
                        temp_config.shuffle, temp_config.seed, temp_config.init_cls, temp_config.increment, temp_config.use_valid)
                temp_config.print_config(logger)
                logger.init_visual_log(temp_config)
                trainer = methods.get_trainer(logger, temp_config)
                trainer.set_save_models(False)
            
                for task_id in range(data_manager.nb_tasks):
                    trainer.prepare_task_data(data_manager)
                    if task_id in seed_checkpoint_paths[seed].keys():
                        trainer.set_save_models(False)
                        task_checkpoint = torch.load(os.path.join(temp_config.checkpoint_dir, seed_checkpoint_paths[seed][task_id]))
                        logger.info('Applying checkpoint: {}'.format(os.path.join(temp_config.checkpoint_dir,
                                seed_checkpoint_paths[seed][task_id])))
                        trainer.prepare_model(task_checkpoint)
                    else:
                        trainer.set_save_models(True)
                        trainer.prepare_model()
                        trainer.incremental_train()
                    
                    trainer.store_samples()
                    trainer.eval_task()
                    trainer.after_task()
                    if temp_config.method_type == 'single_step':
                        break

                logger.info('='*10 + 'Checkpoint Testing/Resuming Finished !' + '='*10)
                logger.info(' ')

                trainer.release()
                torch.cuda.empty_cache()
                logger.release_handlers()
                
        except Exception as e:
            logger.error(e, exc_info=True, stack_info=True)
            logger.release_handlers()
        except KeyboardInterrupt as e:
            logger.error(e, exc_info=True, stack_info=True)
            logger.release_handlers()

    else: # train model
        # torch.set_num_threads(config.num_workers) # limit cpu usage, important for DarkER, X-DER
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
                    trainer.incremental_train()
                    trainer.store_samples()
                    trainer.eval_task()
                    trainer.after_task()
                    if temp_config.method_type == 'single_step':
                        break
                    
                logger.info('='*10 + 'Training Finished !' + '='*10)
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
