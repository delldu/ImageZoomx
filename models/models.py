import copy
import pdb

models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(model_spec, args=None, load_sd=False):
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
        model_args = model_spec['args']
    model = models[model_spec['name']](**model_args)
    # pdb.set_trace()
    # model_spec = {'name': 'edsr-baseline', 'args': {'no_upsampling': True}}
    # args = None
    # load_sd = False

    # ==> (Pdb) model_args -- {'no_upsampling': True}

    # (Pdb) pp models
    # {'edsr': <function make_edsr at 0x7fd963eef6a8>,
    #  'edsr-baseline': <function make_edsr_baseline at 0x7fd963eef620>,
    #  'liif': <class 'models.liif.LIIF'>,
    #  'metasr': <class 'models.misc.MetaSR'>,
    #  'mlp': <class 'models.mlp.MLP'>,
    #  'rcan': <function make_rcan at 0x7fd963ef8950>,
    #  'rdn': <function make_rdn at 0x7fd963eefbf8>}


    if load_sd:
        model.load_state_dict(model_spec['sd'])

    # pp model
    # EDSR(
    #   (sub_mean): MeanShift(3, 3, kernel_size=(1, 1), stride=(1, 1))
    #   (add_mean): MeanShift(3, 3, kernel_size=(1, 1), stride=(1, 1))
    #   (head): Sequential(
    #     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   )
    #   (body): Sequential(
    #     (0): ResBlock(
    #       (body): Sequential(
    #         (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (1): ReLU(inplace=True)
    #         (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #       )
    #     )
    #     (1): ResBlock(
    #       (body): Sequential(
    #         (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (1): ReLU(inplace=True)
    #         (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #       )
    #     )
    #     ...
    #     (15): ResBlock(
    #       (body): Sequential(
    #         (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #         (1): ReLU(inplace=True)
    #         (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #       )
    #     )
    #     (16): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #   )
    # )

    return model
