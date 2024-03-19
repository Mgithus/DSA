def freeze_block(model, block_name):
    for name, param in model.named_parameters():
        if block_name in name:
            param.requires_grad = False
