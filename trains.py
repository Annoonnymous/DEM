import utils
from train_tools import mlm
import numpy as np


def train(model, data_loader, optimizer, tokenizer, epoch, device, scheduler, config, mask_generator=None):
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itc', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['mlm']:
        metric_logger.add_meter('loss_mlm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['diffusion']:
        metric_logger.add_meter('loss_diffusion', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['soft_margin']:
        metric_logger.add_meter('loss_soft', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['gmm']:
        metric_logger.add_meter('loss_gmm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    if config['slot']:
        metric_logger.add_meter('loss_slot', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    if config['eda']:
        for i, (image, text, text_eda, idx, gpt) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            image = image.to(device, non_blocking=True)
            idx = idx.to(device, non_blocking=True)

            text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'],
                                   return_tensors="pt").to(device)
            text_input_eda = tokenizer(text_eda, padding='max_length', truncation=True, max_length=config['max_tokens'],
                                       return_tensors="pt").to(device)
            text_ids_masked, masked_pos, masked_ids = mlm(text, text_input, tokenizer, device, mask_generator,
                                                        config)
            
            gpt_input = tokenizer(gpt, padding='max_length', truncation=True, max_length=config['max_tokens'],
                                   return_tensors="pt").to(device)
            add_loss= model(image, text_input.input_ids, text_input.attention_mask,
                                                text_ids_masked=text_ids_masked,
                                                masked_pos=masked_pos, masked_ids=masked_ids, idx=idx,
                                                text_ids_eda=text_input_eda.input_ids,
                                                text_atts_eda=text_input_eda.attention_mask, cur_epoch=epoch, gpt_input=gpt_input)
            loss = 0
            for k, v in add_loss.items():
                loss = loss + v

            optimizer.zero_grad()
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(name)

            optimizer.step()
            scheduler.step()

            metric_logger.update(loss_itc=add_loss['loss_itc'].item())
            metric_logger.update(loss_itm=add_loss['loss_itm'].item())
            if config['mlm']:
                metric_logger.update(loss_mlm=add_loss['loss_mlm'].item())
            if config['diffusion']:
                metric_logger.update(loss_diffusion=add_loss['loss_diffusion'].item())
            if config['soft_margin']:
                metric_logger.update(loss_soft=add_loss['loss_soft'].item())
            if config['gmm']:
                metric_logger.update(loss_gmm=add_loss['loss_gmm'].item())
            if config['slot']:
                metric_logger.update(loss_slot=add_loss['loss_slot'].item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.5f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
