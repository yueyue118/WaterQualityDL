# root_dir = '/slurm/home/yrd/yulab/zhengyue/Water_quality_prediction/src/'
# base_dir = '/slurm/home/yrd/yulab/zhengyue/Water_quality_prediction/src/'

root_dir = 'F:/河道水质预测/src'
base_dir = 'F:/河道水质预测/src'

log_dir= root_dir + 'logs'
tblog_dir= root_dir + 'tblogs'

data = dict(
    model_name='Transformer',
    num_stations=7,
    dataset_dir='data',
    model_dir='model',
    model_pkl_filename='./model/model.pkl',
    batch_size=32,
    test_batch_size=1,
    data_files = ['3142000027.csv', '3142000028.csv', '3142000023.csv', '3142000038.csv'],
    W_data_name=['3142000027.csv'],
    M_data_name=['3142000027_M.csv'],
    mask_ratio=0.5,
    train_ratio=0.8,
)

model = dict(
    n_in=8, # 输入的seq数量
    n_out=1, # 输出的seq数量
    num_heads=8,
    e_layer=4,
    hidden_size= 64,
    input_dim=4, # 输入的每个站点的水质参数数量
    feature_dim=7, # 输入的每个站点的气象参数数量
    output_dim=4, # 输出的每个站点的水质参数数量
)

train = dict(
    n_gpu=0, #表示有一块GPU
    epochs=200,
    save_dir= "saved",
    base_lr=1e-1,
    epsilon=1.0e-1,
    max_grad_norm= 1e-1,
    lr_milestones= [40, 80, 120, 160, 180, 200, 220, 240, 260, 280],
    lr_decay_ratio=0.5
)