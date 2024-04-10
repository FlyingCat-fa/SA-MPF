from DxFormer_utils import *
from data_utils.common_utils import write_json

def count(train_data, test_data):
    datas = train_data + test_data
    exp_nums = 0
    imp_nums = 0
    max_exp_nums = 0
    max_imp_nums = 0
    for example in datas:
        num = len(example['exp_sxs'])
        if num > max_exp_nums:
            max_exp_nums = num
        exp_nums += num

        num = len(example['imp_sxs'])
        if num > max_imp_nums:
            max_imp_nums = num
        imp_nums += num
    return len(datas), len(train_data), len(test_data), round(exp_nums/len(datas), 2), max_exp_nums, round(imp_nums/len(datas), 2), max_imp_nums
    

def replace_(data):
    """
    mz4中'上呼吸道感染'在症状和疾病中均出现了，将疾病中的重标记为'上呼吸道感染_疾病'
    """
    for example in data:
        if example['label'] == '上呼吸道感染':
            example['label'] = '上呼吸道感染_疾病'
        elif example['label'] ==  "新生儿黄疸":
            example['label'] = '新生儿黄疸_疾病'
        elif example['label'] ==  "Diaper rash":
            example['label'] = 'Diaper rash_疾病'
    return  data

def get_sx_dx(data):
    sxs = []
    dxs = []
    for example in data:
        for sx in (list(example['exp_sxs'].keys()) + list(example['imp_sxs'].keys())):
            if sx not in sxs:
                sxs.append(sx)
        if example['label'] not in dxs:
            dxs.append(example['label'])
    return  sxs, dxs

# dxy 数据集有 423 个训练集 和 104 个测试集
dxy_path = 'data/dxy/raw/dxy_dialog_data_dialog_v2.pickle'

# mz4 数据集有多个版本
# 第一个版本为 ACL 2018 年文章的版本，共 568 个训练集 和 142 个测试集合。
# 第二个版本为 HRL 2020 年文章的版本，共 1214 个训练集，174 个验证集 和 345 个测试集。
mz4_path = 'data/mz4/raw/acl2018-mds.p'
mz4_1k_path = 'data/mz4-1k/raw/'

# mz10 数据集
mz10_path = 'data/mz10/raw/'

syn_path = 'data/synthetic/goal_set.p'
MDD_path = 'data/MDD'

# symcat200
symcat200_path = 'data/symcat200/'
train_data = pd.read_pickle(os.path.join(symcat200_path, 'symcat_200_train_df.pkl'))
dev_data = pd.read_pickle(os.path.join(symcat200_path, 'symcat_200_val_df.pkl'))
test_data = pd.read_pickle(os.path.join(symcat200_path, 'symcat_200_test_df.pkl'))
train_samples = convert_symcat(test_data, 'train') + convert_symcat(test_data, 'dev')
test_samples = convert_symcat(test_data, 'test')
json_dump(train_samples, test_samples, 'symcat200')
sxs, dxs = get_sx_dx(train_samples + test_samples)
nums, train_nums, test_nums, avg_exp_nums, max_exp_nums, avg_imp_nums, max_imp_nums = count(train_samples, test_samples)
write_json({'nums': nums, 'train_nums': train_nums, 'test_nums':test_nums, 'avg_exp_nums':avg_exp_nums, 
            'max_exp_nums': max_exp_nums, 'avg_imp_nums': avg_imp_nums, 'max_imp_nums': max_imp_nums, 'sx_nums': len(sxs), 'dx_nums': len(dxs)}, path='data/symcat200/count.json')
write_json({'症状': sxs, '疾病': dxs}, path='data/symcat200/ontology.json')

# symcat300
symcat300_path = 'data/symcat300/'
train_data = pd.read_pickle(os.path.join(symcat300_path, 'symcat_300_train_df.pkl'))
dev_data = pd.read_pickle(os.path.join(symcat300_path, 'symcat_300_val_df.pkl'))
test_data = pd.read_pickle(os.path.join(symcat300_path, 'symcat_300_test_df.pkl'))
train_samples = convert_symcat(test_data, 'train') + convert_symcat(test_data, 'dev')
test_samples = convert_symcat(test_data, 'test')
json_dump(train_samples, test_samples, 'symcat300')
sxs, dxs = get_sx_dx(train_samples + test_samples)
nums, train_nums, test_nums, avg_exp_nums, max_exp_nums, avg_imp_nums, max_imp_nums = count(train_samples, test_samples)
write_json({'nums': nums, 'train_nums': train_nums, 'test_nums':test_nums, 'avg_exp_nums':avg_exp_nums, 
            'max_exp_nums': max_exp_nums, 'avg_imp_nums': avg_imp_nums, 'max_imp_nums': max_imp_nums, 'sx_nums': len(sxs), 'dx_nums': len(dxs)}, path='data/symcat300/count.json')
write_json({'症状': sxs, '疾病': dxs}, path='data/symcat300/ontology.json')

# symcat400
symcat400_path = 'data/symcat400/'
train_data = pd.read_pickle(os.path.join(symcat400_path, 'symcat_400_train_df.pkl'))
dev_data = pd.read_pickle(os.path.join(symcat400_path, 'symcat_400_val_df.pkl'))
test_data = pd.read_pickle(os.path.join(symcat400_path, 'symcat_400_test_df.pkl'))
train_samples = convert_symcat(test_data, 'train') + convert_symcat(test_data, 'dev')
test_samples = convert_symcat(test_data, 'test')
json_dump(train_samples, test_samples, 'symcat400')
sxs, dxs = get_sx_dx(train_samples + test_samples)
nums, train_nums, test_nums, avg_exp_nums, max_exp_nums, avg_imp_nums, max_imp_nums = count(train_samples, test_samples)
write_json({'nums': nums, 'train_nums': train_nums, 'test_nums':test_nums, 'avg_exp_nums':avg_exp_nums, 
            'max_exp_nums': max_exp_nums, 'avg_imp_nums': avg_imp_nums, 'max_imp_nums': max_imp_nums, 'sx_nums': len(sxs), 'dx_nums': len(dxs)}, path='data/symcat400/count.json')
write_json({'症状': sxs, '疾病': dxs}, path='data/symcat400/ontology.json')

# GMD-12
# 将dev添加至train中
data_path, disease_path, symptom_path = 'data/gmd-12/raw/gmd.pk', 'data/gmd-12/raw/gmd_disease.txt', 'data/gmd-12/raw/gmd_symptom.txt'
data = pickle.load(open(data_path, 'rb'))
train_data, test_data = data['train'] + data['dev'], data['test']
train_samples, test_samples = convert_MDD(train_data, prefix='train'), convert_MDD(test_data, prefix='test')
json_dump(train_samples, test_samples, 'gmd-12')
sxs, dxs = get_sx_dx(train_samples)
nums, train_nums, test_nums, avg_exp_nums, max_exp_nums, avg_imp_nums, max_imp_nums = count(train_samples, test_samples)
write_json({'nums': nums, 'train_nums': train_nums, 'test_nums':test_nums, 'avg_exp_nums':avg_exp_nums, 
            'max_exp_nums': max_exp_nums, 'avg_imp_nums': avg_imp_nums, 'max_imp_nums': max_imp_nums, 'sx_nums': len(sxs), 'dx_nums': len(dxs)}, path='data/gmd-12/count.json')
write_json({'症状': sxs, '疾病': dxs}, path='data/gmd-12/ontology.json')

# MDD
train_data = pickle.load(open(os.path.join(MDD_path, 'train.pk'), 'rb'))
test_data = pickle.load(open(os.path.join(MDD_path, 'dev.pk'), 'rb'))
# test_data = pickle.load(open(os.path.join(MDD_path, 'test.pk'), 'rb'))
train_samples, test_samples = convert_MDD(train_data, prefix='train'), convert_MDD(test_data, prefix='test')
json_dump(train_samples, test_samples, 'MDD')
sxs, dxs = get_sx_dx(train_samples)
# aa = set(sxs).intersection(set(dxs))
# print(list(aa))
nums, train_nums, test_nums, avg_exp_nums, max_exp_nums, avg_imp_nums, max_imp_nums = count(train_samples, test_samples)
write_json({'nums': nums, 'train_nums': train_nums, 'test_nums':test_nums, 'avg_exp_nums':avg_exp_nums, 
            'max_exp_nums': max_exp_nums, 'avg_imp_nums': avg_imp_nums, 'max_imp_nums': max_imp_nums, 'sx_nums': len(sxs), 'dx_nums': len(dxs)}, path='data/MDD/count.json')
write_json({'症状': sxs, '疾病': dxs}, path='data/MDD/ontology.json')

# dxy
data = pickle.load(open(dxy_path, 'rb'))

train_samples, test_samples = convert_dxy(data['train'], prefix='train'), convert_dxy(data['test'], prefix='test')

json_dump(train_samples, test_samples, 'dxy')

sxs, dxs = get_sx_dx(train_samples)
write_json({'症状': sxs, '疾病': dxs}, path='data/dxy/ontology.json')
nums, train_nums, test_nums, avg_exp_nums, max_exp_nums, avg_imp_nums, max_imp_nums = count(train_samples, test_samples)
write_json({'nums': nums, 'train_nums': train_nums, 'test_nums':test_nums, 'avg_exp_nums':avg_exp_nums, 
            'max_exp_nums': max_exp_nums, 'avg_imp_nums': avg_imp_nums, 'max_imp_nums': max_imp_nums, 'sx_nums': len(sxs), 'dx_nums': len(dxs)}, path='data/dxy/count.json')

# mz4
data = pickle.load(open(os.path.join(mz4_path), 'rb'))

train_samples = convert_mz4(data['train'], prefix='train')
test_samples = convert_mz4(data['test'], prefix='test')
train_samples = replace_(train_samples)
test_samples = replace_(test_samples)
json_dump(train_samples, test_samples, 'mz4')

sxs, dxs = get_sx_dx(train_samples)
write_json({'症状': sxs, '疾病': dxs}, path='data/mz4/ontology.json')
nums, train_nums, test_nums, avg_exp_nums, max_exp_nums, avg_imp_nums, max_imp_nums = count(train_samples, test_samples)
write_json({'nums': nums, 'train_nums': train_nums, 'test_nums':test_nums, 'avg_exp_nums':avg_exp_nums, 
            'max_exp_nums': max_exp_nums, 'avg_imp_nums': avg_imp_nums, 'max_imp_nums': max_imp_nums, 'sx_nums': len(sxs), 'dx_nums': len(dxs)}, path='data/mz4/count.json')

# mz4-1k
train_data = pickle.load(open(os.path.join(mz4_1k_path, 'goal_set.p'), 'rb'))
# test_data = pickle.load(open(os.path.join(mz4_1k_path, 'goal_test_set.p'), 'rb'))

train_samples = convert_mz4(train_data['train'] + train_data['validate'], prefix='train')
# test_samples = convert_mz4(test_data['test'], prefix='test')
test_samples = convert_mz4(train_data['test'], prefix='test')

json_dump(train_samples, test_samples, 'mz4-1k')

sxs, dxs = get_sx_dx(train_samples)
write_json({'症状': sxs, '疾病': dxs}, path='data/mz4-1k/ontology.json')
nums, train_nums, test_nums, avg_exp_nums, max_exp_nums, avg_imp_nums, max_imp_nums = count(train_samples, test_samples)
write_json({'nums': nums, 'train_nums': train_nums, 'test_nums':test_nums, 'avg_exp_nums':avg_exp_nums, 
            'max_exp_nums': max_exp_nums, 'avg_imp_nums': avg_imp_nums, 'max_imp_nums': max_imp_nums, 'sx_nums': len(sxs), 'dx_nums': len(dxs)}, path='data/mz4-1k/count.json')

# mz10
train_data = load_json(os.path.join(mz10_path, 'train.json'))
dev_data = load_json(os.path.join(mz10_path, 'dev.json'))
test_data = load_json(os.path.join(mz10_path, 'test.json'))

train_samples = convert_mz10(train_data) + convert_mz10(dev_data)
test_samples = convert_mz10(test_data)

train_samples = replace_(train_samples)
test_samples = replace_(test_samples)
json_dump(train_samples, test_samples, 'mz10')

sxs, dxs = get_sx_dx(train_samples)
write_json({'症状': sxs, '疾病': dxs}, path='data/mz10/ontology.json')
nums, train_nums, test_nums, avg_exp_nums, max_exp_nums, avg_imp_nums, max_imp_nums = count(train_samples, test_samples)
write_json({'nums': nums, 'train_nums': train_nums, 'test_nums':test_nums, 'avg_exp_nums':avg_exp_nums, 
            'max_exp_nums': max_exp_nums, 'avg_imp_nums': avg_imp_nums, 'max_imp_nums': max_imp_nums, 'sx_nums': len(sxs), 'dx_nums': len(dxs)}, path='data/mz10/count.json')

# synthetic
data = pickle.load(open(os.path.join(syn_path), 'rb'))

train_samples = convert_mz4(data['train'], prefix='train')
test_samples = convert_mz4(data['test'], prefix='test')
train_samples = replace_(train_samples)
test_samples = replace_(test_samples)
json_dump(train_samples, test_samples, 'synthetic')

sxs, dxs = get_sx_dx(train_samples)
# aa = set(sxs).intersection(set(dxs))
# print(list(aa))
write_json({'症状': sxs, '疾病': dxs}, path='data/synthetic/ontology.json')
nums, train_nums, test_nums, avg_exp_nums, max_exp_nums, avg_imp_nums, max_imp_nums = count(train_samples, test_samples)
write_json({'nums': nums, 'train_nums': train_nums, 'test_nums':test_nums, 'avg_exp_nums':avg_exp_nums, 
            'max_exp_nums': max_exp_nums, 'avg_imp_nums': avg_imp_nums, 'max_imp_nums': max_imp_nums, 'sx_nums': len(sxs), 'dx_nums': len(dxs)}, path='data/synthetic/count.json')
