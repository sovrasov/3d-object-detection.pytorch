data = dict(
    root="./data",
    resize=(224,128),
    batch_size=128,
    max_epochs=1,
    num_workers=4,
    normalization=dict(mean=[0.5931, 0.4690, 0.4229],
                       std=[0.2471, 0.2214, 0.2157])
)

data_parallel = dict(use_parallel=False,
                     parallel_params=dict(device_ids=[0,1], output_device=0))

output_dir = './log_out'

optim = dict(name='sgd', lr=0.01, momentum=0.9, wd=5e-4, betas=(0.9, 0.999), rho=0.9, alpha=0.99)

loss=dict(names=['smoothl1', 'cross_entropy'])

debug_mode = True
