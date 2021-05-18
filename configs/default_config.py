data = dict(
    root="./data",
    resize=(224,224),
    train_batch_size=164,
    val_batch_size=128,
    max_epochs=130,
    num_workers=8,
    category_list='all',
    normalization=dict(mean=[0.5931, 0.4690, 0.4229],
                       std=[0.2471, 0.2214, 0.2157])
)

model = dict(name='mobilenetv3_large_21k', pretrained=True, num_classes=9)

data_parallel = dict(use_parallel=True,
                     parallel_params=dict(device_ids=[0,1], output_device=0))

optim = dict(name='adam', lr=0.001, momentum=0.9, wd=1e-4, betas=(0.9, 0.999), rho=0.9, alpha=0.99, nesterov=True)

scheduler = dict(name='multistepLR', gamma=0.6, exp_gamma=0.975, steps=[60, 90, 120])

loss = dict(names=['l1', 'add_loss', 'cross_entropy'], coeffs=([1., .1],[.2]), smoothl1_beta=0.2,
            alwa=dict(use=False, lam_cls=1., lam_reg=1., C=100, compute_std=True), w=5.18, eps=1.)

output_dir = './output/log'

utils = dict(debug_mode=False, random_seeds=5, save_freq=10, print_freq=20, debug_steps=100, eval_freq=5)

regime = dict(type='training', vis_only=False)

train_data_pipeline = [('convert_color', dict()),
                       ('resize', dict(height=data['resize'][0], width=data['resize'][1])),
                       ('horizontal_flip', dict(p=0.4)),
                       ('random_brightness_contrast', dict(p=0.3)),
                       ('random_rotate', dict(angle_limit=10., p=0.4)),
                       ('normalize', data['normalization']),
                       ('to_tensor', dict(img_shape=data['resize']))]

test_data_pipeline = [('convert_color', dict()),
                      ('resize', dict(height=data['resize'][0], width=data['resize'][1])),
                      ('normalize', data['normalization']),
                      ('to_tensor', dict(img_shape=data['resize']))]
