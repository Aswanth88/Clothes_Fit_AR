from torch.nn import init


def weights_init_normal(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif 'Linear' in classname:
        init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif 'BatchNorm2d' in classname:
        init.normal_(m.weight.data, mean=1.0, std=0.02)
        init.constant_(m.bias.data, val=0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif 'Linear' in classname:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif 'BatchNorm2d' in classname:
        init.normal_(m.weight.data, mean=1.0, std=0.02)
        init.constant_(m.bias.data, val=0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif 'Linear' in classname:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif 'BatchNorm2d' in classname:
        init.normal_(m.weight.data, mean=1.0, std=0.02)
        init.constant_(m.bias.data, val=0.0)


def init_weights(net, init_type='normal'):
    print(f'weight initialization method [{init_type}]')
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError(f'weights initialization method [{init_type}] is not implemented')

