from data_utils.common_utils import read_json

class diaTokenizer():
    vocab = {}
    id_to_token = {}
    attrs = {"1":0, "0": 1} # true: 0 False: 1
    def __init__(self, vocab_file, unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]", sx_mask_token="[SX_MASK]"):
        # attr_mask_token="[ATTR_MASK]",true_token="1",false_token="0", not_sure_token="2"
        # special_tokens = [pad_token, attr_mask_token, sx_mask_token, cls_token, sep_token, unk_token, true_token, false_token, not_sure_token]
        special_tokens = [pad_token, sx_mask_token, cls_token, sep_token, unk_token]
        ontology = read_json(vocab_file)
        all_tokens = special_tokens + ontology['症状'] + ontology['疾病']
        self.sx_num = len(ontology['症状'])
        self.dx_num = len(ontology['疾病'])

        for index, token in enumerate(all_tokens):
            self.vocab[token] = index
            self.id_to_token[index] = token
            index += 1
        self.unk_token_id = self.vocab[unk_token]
        self.sep_token_id = self.vocab[sep_token]
        self.pad_token_id = self.vocab[pad_token]
        self.cls_token_id = self.vocab[cls_token]
        # self.attr_mask_token_id = self.vocab[attr_mask_token]
        self.sx_mask_token_id = self.vocab[sx_mask_token]
        # self.true_token_id = self.vocab[true_token]
        # self.false_token_id = self.vocab[false_token]
        # self.not_sure_token_id = self.vocab[not_sure_token]
        self.special_tokens_id =  list(range(len(special_tokens)))
        self.disease_tokens_id =  list(range(len(self.vocab)-self.dx_num,len(self.vocab)))

    def convert_token_to_id(self, token):
        if token not in self.vocab:
            return self.unk_token_id
        return self.vocab[token]
    
    def convert_attr_to_id(self, token):
        if token not in self.attrs:
            return self.unk_token_id
        return self.attrs[token]
    
    def __len__(self):
        return len(self.vocab)

    def convert_id_to_token(self,id):
        if id in self.id_to_token:
            return self.id_to_token[id]
        return self.unk_token_id

    def convert_ids_to_tokens(self,ids):
        tokens = []
        for id in ids:
            tokens.append(self.convert_id_to_token(id))
        return tokens

    def get_disease_ids(self):
        return self.disease_tokens_id