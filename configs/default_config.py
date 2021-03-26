data = dict(
    root="./data_cereal_box",
    resize=(224,128),
    batch_size=128,
    max_epochs=70,
    num_workers=4,
    normalization=dict(mean=[0.5931, 0.4690, 0.4229],
                       std=[0.2471, 0.2214, 0.2157])
)

model = dict(pretrained=True, num_classes=1,load_weights='')

data_parallel = dict(use_parallel=True,
                     parallel_params=dict(device_ids=[0,1], output_device=0))

optim = dict(name='sgd', lr=0.01, momentum=0.9, wd=5e-4, betas=(0.9, 0.999), rho=0.9, alpha=0.99)

scheduler = dict(name='cosine', gamma=0.1, exp_gamma=0.965, steps=[50])

loss=dict(names=['smoothl1', 'cross_entropy'], lam=1., coeffs=([1.],[.2]))

output_dir = './output/exp_2'

debug_mode = False

regime = dict(type='training', vis_only=False)
