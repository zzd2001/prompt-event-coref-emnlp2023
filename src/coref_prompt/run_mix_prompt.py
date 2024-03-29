# 基础包
import os
import logging
import json
from tqdm.auto import tqdm
from collections import defaultdict
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import AdamW, get_scheduler
from sklearn.metrics import classification_report
import sys
sys.path.append('../../')  # 在系统路径中增加指定目录

from src.tools import seed_everything, NpEncoder
from src.coref_prompt.arg import parse_args
from src.coref_prompt.data import KBPCoref, KBPCorefTiny, get_dataLoader
from src.coref_prompt.data import get_pred_related_info
from src.coref_prompt.modeling import BertForMixPrompt, RobertaForMixPrompt
from src.coref_prompt.modeling import BertForSimpMixPrompt, RobertaForSimpMixPrompt
from src.coref_prompt.prompt import EVENT_SUBTYPES, id2subtype
from src.coref_prompt.prompt import create_prompt, create_verbalizer, get_special_tokens
# 配置logger，指定log的输出格式：时间+信息级别+logger name+信息
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger("Model") # 创建一个model的logger实例对象，之后可以使用logger.info('message')

MIX_PROMPT_MODELS = {
    'bert': BertForMixPrompt, 
    'roberta': RobertaForMixPrompt
}
SIMP_MIX_PROMPT_MODELS = {
    'bert': BertForSimpMixPrompt, 
    'roberta': RobertaForSimpMixPrompt
}

def to_device(args, batch_data):
    new_batch_data = {}
    # 针对不同的batch内的数据类型，转换为对应的格式，放到字典的指定值
    for k, v in batch_data.items():
        if k in ['batch_inputs', 'batch_mask_inputs']:
            new_batch_data[k] = {
                k_: v_.to(args.device) for k_, v_ in v.items()
            }
        elif k in ['batch_event_idx', 'label_word_id', 'match_label_word_id', 'subtype_label_word_id']:
            new_batch_data[k] = v
        else:
            new_batch_data[k] = torch.tensor(v).to(args.device)
    return new_batch_data

def train_loop(args, dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss: {0:>7f}')
    finish_step_num = epoch * len(dataloader)

    model.train()
    for step, batch_data in enumerate(dataloader, start=1):
        batch_data = to_device(args, batch_data)
        outputs = model(**batch_data)
        loss = outputs[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f'loss: {total_loss/(finish_step_num + step):>7f}')
        progress_bar.update(1)
    return total_loss

def test_loop(args, dataloader, model):
    true_labels, predictions = [], []
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(dataloader):
            true_labels += batch_data['labels']
            batch_data = to_device(args, batch_data)
            outputs = model(**batch_data)
            logits = outputs[1]
            predictions += logits.argmax(dim=-1).cpu().numpy().tolist()
    return classification_report(true_labels, predictions, output_dict=True) # 返回分类指标的一个字典

def train(args, train_dataset, dev_dataset, model, tokenizer, prompt_type, verbalizer):
    """ Train the model """
    # Set seed
    seed_everything(args.seed)
    train_dataloader = get_dataLoader(args, train_dataset, tokenizer, prompt_type, verbalizer, shuffle=True)
    dev_dataloader = get_dataLoader(args, dev_dataset, tokenizer, prompt_type, verbalizer, with_mask=False, shuffle=False)
    t_total = len(train_dataloader) * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    # 针对no_decay的参数，无需进行参数优化，以保证PLM的特性，针对其他的参数，需要进行优化
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
    ]
    args.warmup_steps = int(t_total * args.warmup_proportion)
    '''
    betas=(args.adam_beta1, args.adam_beta2)：这两个参数是Adam优化算法中的衰减率参数，通常用于计算梯度及其平方的指数移动平均值。beta1通常设置得比较高（例如0.9），而beta2通常接近1（例如0.999）。这些值决定了过去梯度的权重，影响着优化器如何根据过去的梯度信息调整参数更新方向和幅度。

    eps=args.adam_epsilon：这是一个非常小的数（例如1e-8），用来提高数值稳定性，避免在计算时出现除以零的错误。
    '''
    optimizer = AdamW(
        optimizer_grouped_parameters, 
        lr=args.learning_rate, 
        betas=(args.adam_beta1, args.adam_beta2), 
        eps=args.adam_epsilon
    )
    lr_scheduler = get_scheduler(
        'linear',
        optimizer, 
        num_warmup_steps=args.warmup_steps,
        num_training_steps=t_total
    )
    # Train!
    logger.info("***** Running training *****")
    logger.info(f"Num examples - {len(train_dataset)}")
    logger.info(f"Num Epochs - {args.num_train_epochs}")
    logger.info(f"Total optimization steps - {t_total}")
    # 将本次参数设置写入文件保存
    with open(os.path.join(args.output_dir, 'args.txt'), 'wt') as f:
        f.write(str(args))

    total_loss = 0.
    best_f1 = 0.
    for epoch in range(args.num_train_epochs):
        print(f"Epoch {epoch+1}/{args.num_train_epochs}\n" + "-" * 30)
        total_loss = train_loop(args, train_dataloader, model, optimizer, lr_scheduler, epoch, total_loss)
        metrics = test_loop(args, dev_dataloader, model)
        dev_p, dev_r, dev_f1 = metrics['1']['precision'], metrics['1']['recall'], metrics['1']['f1-score']
        logger.info(f'Dev: P - {(100*dev_p):0.4f} R - {(100*dev_r):0.4f} F1 - {(100*dev_f1):0.4f}')
        # 如果验证集中效果有提升，那么保存模型的权重
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            logger.info(f'saving new weights to {args.output_dir}...\n')
            save_weight = f'epoch_{epoch+1}_dev_f1_{(100*dev_f1):0.4f}_weights.bin'
            torch.save(model.state_dict(), os.path.join(args.output_dir, save_weight))
        # 将每一轮训练中的dev数据集表现写入文件
        with open(os.path.join(args.output_dir, 'dev_metrics.txt'), 'at') as f:
            f.write(f'epoch_{epoch+1}\n' + json.dumps(metrics, cls=NpEncoder) + '\n\n')
    logger.info("Done!")

def test(args, test_dataset, model, tokenizer, save_weights:list, prompt_type, verbalizer):
    test_dataloader = get_dataLoader(args, test_dataset, tokenizer, prompt_type=prompt_type, verbalizer=verbalizer, with_mask=False, shuffle=False)
    logger.info('***** Running testing *****')
    # 在test步骤中，针对dev每轮训练有提升的model权重都进行测试
    for save_weight in save_weights:
        logger.info(f'loading {save_weight}...')
        model.load_state_dict(torch.load(os.path.join(args.output_dir, save_weight)))
        metrics = test_loop(args, test_dataloader, model)
        with open(os.path.join(args.output_dir, 'test_metrics.txt'), 'at') as f:
            f.write(f'{save_weight}\n{json.dumps(metrics, cls=NpEncoder)}\n\n')

def predict(
    args, model, tokenizer, 
    e1_global_offset:int, e1_trigger:str, e1_related_info:dict, 
    e2_global_offset:int, e2_trigger:str, e2_related_info:dict, 
    sentences:list, sentences_lengths:list, 
    prompt_type, select_arg_strategy, verbalizer
    ):

    def find_event_sent(event_start, trigger, sent_list):
        '''find out which sentence the event come from
        '''
        for idx, sent in enumerate(sent_list):
            s_start, s_end = sent['start'], sent['start'] + len(sent['text']) - 1
            # 如果找得到event对应的句子，返回对应句子的id和该event在句子开始的位置
            if s_start <= event_start <= s_end:
                e_s_start = event_start - s_start  # 事件在句子中出现的位置
                assert sent['text'][e_s_start:e_s_start+len(trigger)] == trigger  # assert检查，声明trigger必须对应sent的位置
                return idx, e_s_start
        # 如果没有找到event对应的句子，则打印输出事件开始的位置和触发词，以及每个句子开始和结束的位置
        print(event_start, trigger, '\n')
        for sent in sent_list:
            print(sent['start'], sent['start'] + len(sent['text']) - 1)
        return None
    # 两个事件对应句子的id和在句子中开始的位置
    e1_sent_idx, e1_sent_start = find_event_sent(e1_global_offset, e1_trigger, sentences)
    e2_sent_idx, e2_sent_start = find_event_sent(e2_global_offset, e2_trigger, sentences)

    prompt_data = create_prompt(
        e1_sent_idx, e1_sent_start, e1_trigger, e1_related_info, 
        e2_sent_idx, e2_sent_start, e2_trigger, e2_related_info, 
        sentences, sentences_lengths, 
        prompt_type, select_arg_strategy, 
        args.model_type, tokenizer, args.max_seq_length
    )
    prompt_text = prompt_data['prompt']
    # convert char offsets to token idxs
    # 把位置转换为对应token的idxs
    encoding = tokenizer(prompt_text)
    mask_idx = encoding.char_to_token(prompt_data['mask_offset'])
    # 在第三种模版中，存在两种匹配的检测作为ECR判定的辅助任务：事件类型匹配和事件论元匹配
    # 这个分支判断比较负责，如果设为-1，表示不需要该位置的匹配
    type_match_mask_idx, arg_match_mask_idx = (
        -1, -1
    ) if prompt_type == 'ma_remove-match' else (
        -1, encoding.char_to_token(prompt_data['arg_match_mask_offset'])
    ) if prompt_type == 'ma_remove-subtype-match' else (
        encoding.char_to_token(prompt_data['type_match_mask_offset']), -1
    ) if prompt_type == 'ma_remove-arg-match' else (
        encoding.char_to_token(prompt_data['type_match_mask_offset']), 
        encoding.char_to_token(prompt_data['arg_match_mask_offset']), 
    )
    # 寻找事件对应的token的开始下标和结束下标
    e1s_idx, e1e_idx, e2s_idx, e2e_idx = (
        encoding.char_to_token(prompt_data['e1s_offset']), 
        encoding.char_to_token(prompt_data['e1e_offset']), 
        encoding.char_to_token(prompt_data['e2s_offset']), 
        encoding.char_to_token(prompt_data['e2e_offset'])
    )
    # 在第二种模版中，两个事件类型的推测。与上面类似，-1表示不需要该种类型的推测
    e1_type_mask_idx, e2_type_mask_idx = (
        -1, -1
     ) if prompt_type == 'ma_remove-anchor' else (
        encoding.char_to_token(prompt_data['e1_type_mask_offset']), 
        encoding.char_to_token(prompt_data['e2_type_mask_offset'])
    )
    assert None not in [
        mask_idx, type_match_mask_idx, arg_match_mask_idx, 
        e1s_idx, e1e_idx, e2s_idx, e2e_idx, e1_type_mask_idx, e2_type_mask_idx
    ] # 非空检查
    # 
    event_idx = [e1s_idx, e1e_idx, e2s_idx, e2e_idx]
    inputs = tokenizer(
        prompt_text, 
        max_length=args.max_seq_length, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    inputs = {
        'batch_inputs': inputs, 
        'batch_mask_idx': [mask_idx], 
        'batch_type_match_mask_idx': [type_match_mask_idx], 
        'batch_arg_match_mask_idx': [arg_match_mask_idx], 
        'batch_event_idx': [event_idx], 
        'label_word_id': [verbalizer['non-coref']['id'], verbalizer['coref']['id']], 
        'match_label_word_id': [verbalizer['match']['id'], verbalizer['mismatch']['id']], 
    } if prompt_type == 'ma_remove-anchor' else {
        'batch_inputs': inputs, 
        'batch_mask_idx': [mask_idx], 
        'batch_event_idx': [event_idx], 
        'batch_t1_mask_idx': [e1_type_mask_idx], 
        'batch_t2_mask_idx': [e2_type_mask_idx], 
        'label_word_id': [verbalizer['non-coref']['id'], verbalizer['coref']['id']], 
        'subtype_label_word_id': [
            verbalizer[id2subtype[s_id]]['id'] 
            for s_id in range(len(EVENT_SUBTYPES) + 1)
        ]  
    # 把第三个模版中的所有匹配删除
    } if prompt_type == 'ma_remove-match' else {
        'batch_inputs': inputs, 
        'batch_mask_idx': [mask_idx], 
        'batch_arg_match_mask_idx': [arg_match_mask_idx], 
        'batch_event_idx': [event_idx], 
        'batch_t1_mask_idx': [e1_type_mask_idx], 
        'batch_t2_mask_idx': [e2_type_mask_idx], 
        'label_word_id': [verbalizer['non-coref']['id'], verbalizer['coref']['id']], 
        'match_label_word_id': [verbalizer['match']['id'], verbalizer['mismatch']['id']], 
        'subtype_label_word_id': [
            verbalizer[id2subtype[s_id]]['id'] 
            for s_id in range(len(EVENT_SUBTYPES) + 1)
        ]
    } if prompt_type == 'ma_remove-subtype-match' else {
        'batch_inputs': inputs, 
        'batch_mask_idx': [mask_idx], 
        'batch_type_match_mask_idx': [type_match_mask_idx], 
        'batch_event_idx': [event_idx], 
        'batch_t1_mask_idx': [e1_type_mask_idx], 
        'batch_t2_mask_idx': [e2_type_mask_idx], 
        'label_word_id': [verbalizer['non-coref']['id'], verbalizer['coref']['id']], 
        'match_label_word_id': [verbalizer['match']['id'], verbalizer['mismatch']['id']], 
        'subtype_label_word_id': [
            verbalizer[id2subtype[s_id]]['id'] 
            for s_id in range(len(EVENT_SUBTYPES) + 1)
        ]
    } if prompt_type == 'ma_remove-arg-match' else {
        'batch_inputs': inputs, 
        'batch_mask_idx': [mask_idx], 
        'batch_type_match_mask_idx': [type_match_mask_idx], 
        'batch_arg_match_mask_idx': [arg_match_mask_idx], 
        'batch_event_idx': [event_idx], 
        'batch_t1_mask_idx': [e1_type_mask_idx], 
        'batch_t2_mask_idx': [e2_type_mask_idx], 
        'label_word_id': [verbalizer['non-coref']['id'], verbalizer['coref']['id']], 
        'match_label_word_id': [verbalizer['match']['id'], verbalizer['mismatch']['id']], 
        'subtype_label_word_id': [
            verbalizer[id2subtype[s_id]]['id'] 
            for s_id in range(len(EVENT_SUBTYPES) + 1)
        ]
    }
    inputs = to_device(args, inputs)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs[1]
        prob = torch.nn.functional.softmax(logits, dim=-1)[0]
    # 返回一个样本的预测label以及自信分数
    pred = logits.argmax(dim=-1)[0].item()
    prob = prob[pred].item()
    return pred, prob

if __name__ == '__main__':
    args = parse_args()
    if args.do_train and os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError(f'Output directory ({args.output_dir}) already exists and is not empty.')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()
    logger.warning(f'Using {args.device} device, n_gpu: {args.n_gpu}')
    # Set seed
    seed_everything(args.seed)
    # Load pretrained model and tokenizer
    logger.info(f'loading pretrained model and tokenizer of {args.model_type} ...')
    # cache_dir节省模型的下载时间，第一次从hf官方下载，之后再使用该model可以直接从cache_dir的路径中加载
    config = AutoConfig.from_pretrained(args.model_checkpoint, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, max_length=args.max_seq_length, cache_dir=args.cache_dir)
    # SIMP_MIX_PROMPT_MODELS代表不使用anchor模版，MIX_PROMPT_MODELS代表使用anchor模版
    if args.prompt_type == 'ma_remove-anchor': 
        model = SIMP_MIX_PROMPT_MODELS[args.model_type].from_pretrained(
            args.model_checkpoint,
            config=config, 
            args=args, 
            cache_dir=args.cache_dir
        ).to(args.device)
    else: 
        model = MIX_PROMPT_MODELS[args.model_type].from_pretrained(
            args.model_checkpoint,
            config=config, 
            args=args, 
            cache_dir=args.cache_dir
        ).to(args.device)
    # 四种类型的special token
    # base:基础的特殊token，比如[SEP],[CLS]
    # connect:将不同的template连接的特殊token
    # match：是否匹配的token，可学习的虚拟token
    # event_subtype：事件类型的token（一般为33种），可学习的虚拟token
    base_sp_tokens = get_special_tokens(args.model_type, 'base')
    connect_tokens = get_special_tokens(args.model_type, 'connect')
    match_tokens = get_special_tokens(args.model_type, 'match')
    event_subtype_tokens = get_special_tokens(args.model_type, 'event_subtype')
    print('c' in args.prompt_type)
    # 根据prompt的类型设定对应的特殊token的使用
    sp_tokens = (
        base_sp_tokens + match_tokens 
        if args.prompt_type == 'ma_remove-anchor' else
        base_sp_tokens + event_subtype_tokens
        if args.prompt_type == 'ma_remove-match' else
        base_sp_tokens + connect_tokens + match_tokens + event_subtype_tokens 
        if 'c' in args.prompt_type and not args.prompt_type.startswith('ma') else 
        base_sp_tokens + match_tokens + event_subtype_tokens
    )
    logger.info(f"adding special mark tokens {sp_tokens} to tokenizer...")
    tokenizer.add_special_tokens({'additional_special_tokens': sp_tokens})
    model.resize_token_embeddings(len(tokenizer))  # 这一步根据词表大小，调整head的维度

    # build verbalizer
    # 
    verbalizer = create_verbalizer(tokenizer, args.model_type, args.prompt_type)
    logger.info(f"verbalizer: {verbalizer}")
    # 四种类型prompt_type：ma_remove-anchor，ma_remove-match，ma_remove-subtype-match，ma_remove-arg-match
    # 如果没有设置，则选择完整的三种类型的prompt
    if 'c' in args.prompt_type and not args.prompt_type.startswith('ma'):
        logger.info(f"initialize embeddings for {verbalizer['coref']['token']} and {verbalizer['non-coref']['token']}...")
        subtype_sp_token_num = len(EVENT_SUBTYPES) + 1
        match_sp_token_num = 2
        refer_idx, norefer_idx = -(subtype_sp_token_num+match_sp_token_num+2), -(subtype_sp_token_num+match_sp_token_num+1)
        with torch.no_grad():
            refer_tokenized = tokenizer.tokenize(verbalizer['coref']['description'])
            refer_tokenized_ids = tokenizer.convert_tokens_to_ids(refer_tokenized)
            norefer_tokenized = tokenizer.tokenize(verbalizer['non-coref']['description'])
            norefer_tokenized_ids = tokenizer.convert_tokens_to_ids(norefer_tokenized)
            if args.model_type == 'bert':
                new_embedding = model.bert.embeddings.word_embeddings.weight[refer_tokenized_ids].mean(axis=0)
                model.bert.embeddings.word_embeddings.weight[refer_idx, :] = new_embedding.clone().detach().requires_grad_(True)
                new_embedding = model.bert.embeddings.word_embeddings.weight[norefer_tokenized_ids].mean(axis=0)
                model.bert.embeddings.word_embeddings.weight[norefer_idx, :] = new_embedding.clone().detach().requires_grad_(True)
            elif args.model_type == 'roberta':
                new_embedding = model.roberta.embeddings.word_embeddings.weight[refer_tokenized_ids].mean(axis=0)
                model.roberta.embeddings.word_embeddings.weight[refer_idx, :] = new_embedding.clone().detach().requires_grad_(True)
                new_embedding = model.roberta.embeddings.word_embeddings.weight[norefer_tokenized_ids].mean(axis=0)
                model.roberta.embeddings.word_embeddings.weight[norefer_idx, :] = new_embedding.clone().detach().requires_grad_(True)
            elif args.model_type == 'deberta':
                new_embedding = model.deberta.embeddings.word_embeddings.weight[refer_tokenized_ids].mean(axis=0)
                model.deberta.embeddings.word_embeddings.weight[refer_idx, :] = new_embedding.clone().detach().requires_grad_(True)
                new_embedding = model.deberta.embeddings.word_embeddings.weight[norefer_tokenized_ids].mean(axis=0)
                model.deberta.embeddings.word_embeddings.weight[norefer_idx, :] = new_embedding.clone().detach().requires_grad_(True)
            else: # longformer
                new_embedding = model.longformer.embeddings.word_embeddings.weight[refer_tokenized_ids].mean(axis=0)
                model.longformer.embeddings.word_embeddings.weight[refer_idx, :] = new_embedding.clone().detach().requires_grad_(True)
                new_embedding = model.longformer.embeddings.word_embeddings.weight[norefer_tokenized_ids].mean(axis=0)
                model.longformer.embeddings.word_embeddings.weight[norefer_idx, :] = new_embedding.clone().detach().requires_grad_(True)
    if args.prompt_type != 'ma_remove-match':
        logger.info(f"initialize embeddings for {verbalizer['match']['token']} and {verbalizer['mismatch']['token']}...")
        subtype_sp_token_num = 0 if args.prompt_type == 'ma_remove-anchor' else len(EVENT_SUBTYPES) + 1
        match_idx, mismatch_idx = -(subtype_sp_token_num+2), -(subtype_sp_token_num+1)
        with torch.no_grad():
            match_tokenized = tokenizer.tokenize(verbalizer['match']['description'])
            match_tokenized_ids = tokenizer.convert_tokens_to_ids(match_tokenized)
            mismatch_tokenized = tokenizer.tokenize(verbalizer['mismatch']['description'])
            mismatch_tokenized_ids = tokenizer.convert_tokens_to_ids(mismatch_tokenized)
            if args.model_type == 'bert':
                new_embedding = model.bert.embeddings.word_embeddings.weight[match_tokenized_ids].mean(axis=0)
                model.bert.embeddings.word_embeddings.weight[match_idx, :] = new_embedding.clone().detach().requires_grad_(True)
                new_embedding = model.bert.embeddings.word_embeddings.weight[mismatch_tokenized_ids].mean(axis=0)
                model.bert.embeddings.word_embeddings.weight[mismatch_idx, :] = new_embedding.clone().detach().requires_grad_(True)
            elif args.model_type == 'roberta':
                new_embedding = model.roberta.embeddings.word_embeddings.weight[match_tokenized_ids].mean(axis=0)
                model.roberta.embeddings.word_embeddings.weight[match_idx, :] = new_embedding.clone().detach().requires_grad_(True)
                new_embedding = model.roberta.embeddings.word_embeddings.weight[mismatch_tokenized_ids].mean(axis=0)
                model.roberta.embeddings.word_embeddings.weight[mismatch_idx, :] = new_embedding.clone().detach().requires_grad_(True)
            elif args.model_type == 'deberta':
                new_embedding = model.deberta.embeddings.word_embeddings.weight[match_tokenized_ids].mean(axis=0)
                model.deberta.embeddings.word_embeddings.weight[match_idx, :] = new_embedding.clone().detach().requires_grad_(True)
                new_embedding = model.deberta.embeddings.word_embeddings.weight[mismatch_tokenized_ids].mean(axis=0)
                model.deberta.embeddings.word_embeddings.weight[mismatch_idx, :] = new_embedding.clone().detach().requires_grad_(True)
            else: # longformer
                new_embedding = model.longformer.embeddings.word_embeddings.weight[match_tokenized_ids].mean(axis=0)
                model.longformer.embeddings.word_embeddings.weight[match_idx, :] = new_embedding.clone().detach().requires_grad_(True)
                new_embedding = model.longformer.embeddings.word_embeddings.weight[mismatch_tokenized_ids].mean(axis=0)
                model.longformer.embeddings.word_embeddings.weight[mismatch_idx, :] = new_embedding.clone().detach().requires_grad_(True)
    if args.prompt_type != 'ma_remove-anchor': 
        logger.info(f"initialize embeddings for event subtype special tokens...")
        subtype_descriptions = [
            verbalizer[id2subtype[s_id]]['description'] for s_id in range(len(EVENT_SUBTYPES) + 1)
        ]
        with torch.no_grad():
            for i, description in enumerate(reversed(subtype_descriptions), start=1):
                tokenized = tokenizer.tokenize(description)
                tokenized_ids = tokenizer.convert_tokens_to_ids(tokenized)
                if args.model_type == 'bert':
                    new_embedding = model.bert.embeddings.word_embeddings.weight[tokenized_ids].mean(axis=0)
                    model.bert.embeddings.word_embeddings.weight[-i, :] = new_embedding.clone().detach().requires_grad_(True)
                elif args.model_type == 'roberta':
                    new_embedding = model.roberta.embeddings.word_embeddings.weight[tokenized_ids].mean(axis=0)
                    model.roberta.embeddings.word_embeddings.weight[-i, :] = new_embedding.clone().detach().requires_grad_(True)
                elif args.model_type == 'deberta':
                    new_embedding = model.deberta.embeddings.word_embeddings.weight[tokenized_ids].mean(axis=0)
                    model.deberta.embeddings.word_embeddings.weight[-i, :] = new_embedding.clone().detach().requires_grad_(True)
                else: # longformer
                    new_embedding = model.longformer.embeddings.word_embeddings.weight[tokenized_ids].mean(axis=0)
                    model.longformer.embeddings.word_embeddings.weight[-i, :] = new_embedding.clone().detach().requires_grad_(True)
    # Training
    if args.do_train:
        if args.sample_strategy == 'no':
            train_dataset = KBPCoref(
                args.train_file, 
                args.train_simi_file, 
                prompt_type=args.prompt_type, 
                select_arg_strategy=args.select_arg_strategy, 
                model_type=args.model_type, 
                tokenizer=tokenizer, 
                max_length=args.max_seq_length
            )
        else:
            train_dataset = KBPCorefTiny(
                args.train_file, 
                args.train_file_with_cos, 
                args.train_simi_file, 
                prompt_type=args.prompt_type, 
                select_arg_strategy=args.select_arg_strategy, 
                model_type=args.model_type, 
                tokenizer=tokenizer, 
                max_length=args.max_seq_length, 
                sample_strategy=args.sample_strategy, 
                neg_top_k=args.neg_top_k, 
                neg_threshold=args.neg_threshold, 
                rand_seed=args.seed
            )
        labels = [train_dataset[s_idx]['label'] for s_idx in range(len(train_dataset))]
        logger.info(f"[Train] Coref: {labels.count(1)} non-Coref: {labels.count(0)}")
        dev_dataset = KBPCoref(
            args.dev_file, 
            args.dev_simi_file, 
            prompt_type=args.prompt_type, 
            select_arg_strategy=args.select_arg_strategy, 
            model_type=args.model_type, 
            tokenizer=tokenizer, 
            max_length=args.max_seq_length
        )
        labels = [dev_dataset[s_idx]['label'] for s_idx in range(len(dev_dataset))]
        logger.info(f"[Dev] Coref: {labels.count(1)} non-Coref: {labels.count(0)}")
        train(args, train_dataset, dev_dataset, model, tokenizer, prompt_type=args.prompt_type, verbalizer=verbalizer)
    save_weights = [file for file in os.listdir(args.output_dir) if file.endswith('.bin')]
    # Testing
    if args.do_test:
        test_dataset = KBPCoref(
            args.test_file, 
            args.test_simi_file, 
            prompt_type=args.prompt_type, 
            select_arg_strategy=args.select_arg_strategy, 
            model_type=args.model_type, 
            tokenizer=tokenizer, 
            max_length=args.max_seq_length
        )
        labels = [test_dataset[s_idx]['label'] for s_idx in range(len(test_dataset))]
        logger.info(f"[Test] Coref: {labels.count(1)} non-Coref: {labels.count(0)}")
        logger.info(f'loading trained weights from {args.output_dir} ...')
        test(args, test_dataset, model, tokenizer, save_weights, prompt_type=args.prompt_type, verbalizer=verbalizer)
    # Predicting
    if args.do_predict:
        sentence_dict = defaultdict(list) # {filename: [Sentence]}
        sentence_len_dict = defaultdict(list) # {filename: [sentence length]}
        related_dict = get_pred_related_info(args.pred_test_simi_file)
        with open(args.test_file, 'rt', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                sentences = sample['sentences']
                sentences_lengths = [len(tokenizer.tokenize(sent['text'])) for sent in sentences]
                sentence_dict[sample['doc_id']] = sentences
                sentence_len_dict[sample['doc_id']] = sentences_lengths
        
        pred_event_file = '../../data/epoch_3_test_pred_events.json'

        for best_save_weight in save_weights:
            logger.info(f'loading weights from {best_save_weight}...')
            model.load_state_dict(torch.load(os.path.join(args.output_dir, best_save_weight)))
            logger.info(f'predicting coref labels of {best_save_weight}...')

            results = []
            model.eval()
            with open(pred_event_file, 'rt' , encoding='utf-8') as f_in:
                for line in tqdm(f_in.readlines()):
                    sample = json.loads(line.strip())
                    events_from_file = sample['pred_label']
                    sentences = sentence_dict[sample['doc_id']]
                    sentence_lengths = sentence_len_dict[sample['doc_id']]
                    doc_related_info = related_dict[sample['doc_id']]
                    preds, probs = [], []
                    for i in range(len(events_from_file) - 1):
                        for j in range(i + 1, len(events_from_file)):
                            e_i, e_j = events_from_file[i], events_from_file[j]
                            pred, prob = predict(
                                args, model, tokenizer,
                                e_i['start'], e_i['trigger'], doc_related_info[e_i['start']], 
                                e_j['start'], e_j['trigger'], doc_related_info[e_j['start']], 
                                sentences, sentence_lengths, 
                                args.prompt_type, args.select_arg_strategy, verbalizer
                            )
                            preds.append(pred)
                            probs.append(prob)
                    results.append({
                        "doc_id": sample['doc_id'], 
                        "document": sample['document'], 
                        "sentences": sentences, 
                        "events": [
                            {
                                'start': e['start'], 
                                'end': e['start'] + len(e['trigger']) - 1, 
                                'trigger': e['trigger']
                            } for e in events_from_file
                        ], 
                        "pred_label": preds, 
                        "pred_prob": probs
                    })
            save_name = f'_{args.model_type}_{args.prompt_type}_test_pred_corefs.json'
            with open(os.path.join(args.output_dir, best_save_weight + save_name), 'wt', encoding='utf-8') as f:
                for example_result in results:
                    f.write(json.dumps(example_result) + '\n')
