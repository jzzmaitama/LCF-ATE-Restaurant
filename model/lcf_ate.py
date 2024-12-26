from transformers.models.bert.modeling_bert import BertForTokenClassification, BertPooler, BertSelfAttention
from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn
import numpy as np
import copy



class SelfAttention(nn.Module):
    def __init__(self, config, opt):
        super(SelfAttention, self).__init__()
        self.opt = opt
        self.config = config
        self.SA = BertSelfAttention(config)
        self.tanh = torch.nn.Tanh()

    def forward(self, inputs):
        zero_vec = np.zeros((inputs.size(0), 1, 1, self.opt.max_seq_length))
        zero_tensor = torch.tensor(zero_vec).float().to(self.opt.device)
        SA_out = self.SA(inputs, zero_tensor)
        return self.tanh(SA_out[0])


class LCF_ATE(BertForTokenClassification):
    def __init__(self, bert_base_model, args):
        super(LCF_ATE, self).__init__(config=bert_base_model.config)
        config = bert_base_model.config
        self.bert_for_global_context = bert_base_model
        self.args = args
        self.num_emotion_labels = 6

        self.emotion_classifier = nn.Linear(768, 3)  # 6 for the number of emotions
        # do not init lcf layer if BERT-SPC or BERT-BASE specified
        # if self.args.local_context_focus in {'cdw', 'cdm', 'fusion'}:
        if not self.args.use_unique_bert:
            self.bert_for_local_context = copy.deepcopy(self.bert_for_global_context)
        else:
            self.bert_for_local_context = self.bert_for_global_context
        self.pooler = BertPooler(config)
        self.bert_global_focus = self.bert_for_global_context
        self.dropout = nn.Dropout(self.args.dropout)
        self.SA1 = SelfAttention(config, args)
        self.SA2 = SelfAttention(config, args)
        self.linear_double = nn.Linear(768 * 2, 768)
        self.linear_triple = nn.Linear(768 * 3, 768)
        self.linear_concat = nn.Linear(768 * 5, 768)



    def get_batch_token_labels_bert_base_indices(self, labels):
        if labels is None:
            return
        # convert tags of BERT-SPC input to BERT-BASE format
        labels = labels.detach().cpu().numpy()
        for text_i in range(len(labels)):
            sep_index = np.argmax((labels[text_i] == 5))
            labels[text_i][sep_index + 1:] = 0
        return torch.tensor(labels).to(self.args.device)

    def get_batch_emotions(self, b_emotions):
        b_emotions = b_emotions.detach().cpu().numpy()
        shape = b_emotions.shape
        emotions = np.zeros((shape[0]))
        i = 0
        for emotion in b_emotions:
            emotion_idx = np.flatnonzero(emotion + 1)
            try:
                emotions[i] = emotion[emotion_idx[0]]
            except:
                pass
            i += 1
        emotions = torch.from_numpy(emotions).long().to(self.args.device)
        return emotions

    def emotions_feature_dynamic_weighted(self, text_local_indices, emotions):
        text_ids = text_local_indices.detach().cpu().numpy()
        emotion_ids = emotions.detach().cpu().numpy()
        weighted_text_raw_indices = np.ones((text_local_indices.size(0), text_local_indices.size(1), 768),
                                            dtype=np.float32)

        for text_i in range(len(text_ids)):
            emotion_labels = emotion_ids[text_i]
            text_len = np.flatnonzero(text_ids[text_i])[-1] + 1

            # Calculate weights based on emotion labels
            for i in range(text_len):
                if emotion_labels[i] != -1:
                    emotion_labels_tensor = torch.tensor(emotion_labels[i],dtype=torch.float32)
                    transformed_label = torch.sigmoid(emotion_labels_tensor)
                    weighted_text_raw_indices[text_i][i] *= transformed_label.item()

        weighted_text_raw_indices = torch.from_numpy(weighted_text_raw_indices)
        return weighted_text_raw_indices.to(self.args.device)

    def emotions_feature_dynamic_mask(self, text_local_indices, emotions):
        text_ids = text_local_indices.detach().cpu().numpy()
        emotion_ids = emotions.detach().cpu().numpy()
        masked_text_raw_indices = np.ones((text_local_indices.size(0), text_local_indices.size(1), 768),
                                          dtype=np.float32)
        for text_i in range(len(text_ids)):
            emotion_labels = emotion_ids[text_i]
            text_len = np.flatnonzero(text_ids[text_i])[-1] + 1
            # Mask out tokens without emotion labels
            for i in range(text_len):
                if emotion_labels[i] == -1:
                    masked_text_raw_indices[text_i][i] = np.zeros((768,), dtype=np.float32)
        masked_text_raw_indices = torch.from_numpy(masked_text_raw_indices)
        return masked_text_raw_indices.to(self.args.device)
    def get_ids_for_local_context_extractor(self, text_indices):
        text_ids = text_indices.detach().cpu().numpy()
        for text_i in range(len(text_ids)):
            sep_index = np.argmax((text_ids[text_i] == 102))
            text_ids[text_i][sep_index + 1:] = 0
        return torch.tensor(text_ids).to(self.args.device)

    def forward(self, input_ids_spc, token_type_ids=None, attention_mask=None, labels=None,
                valid_ids=None, attention_mask_label=None,emotions=None):
        if not self.args.use_bert_spc:
            input_ids_spc = self.get_ids_for_local_context_extractor(input_ids_spc)
            labels = self.get_batch_token_labels_bert_base_indices(labels)
        global_context_out = self.bert_for_global_context(input_ids_spc, token_type_ids, attention_mask)['last_hidden_state']
        emotion_labels = self.get_batch_emotions(emotions)
        batch_size, max_len, feat_dim = global_context_out.shape
        global_valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32).to(self.args.device)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if valid_ids[i][j].item() == 1:
                    jj += 1
                    global_valid_output[i][jj] = global_context_out[i][j]
        global_context_out = self.dropout(global_valid_output)
        ate_logits = self.classifier(global_context_out)

        if self.args.local_context_focus is not None:

            if self.args.use_bert_spc:
                local_context_ids = self.get_ids_for_local_context_extractor(input_ids_spc)
            else:
                local_context_ids = input_ids_spc

            local_context_out = self.bert_for_local_context(input_ids_spc)['last_hidden_state']
            batch_size, max_len, feat_dim = local_context_out.shape
            local_valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32).to(self.args.device)
            for i in range(batch_size):
                jj = -1
                for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        local_valid_output[i][jj] = local_context_out[i][j]
            local_context_out = self.dropout(local_valid_output)

            if 'cdm' in self.args.local_context_focus:
                cdm_e_vec = self.emotions_feature_dynamic_mask(local_context_ids, emotions)
                cdm_e_context_out = torch.mul(local_context_out, cdm_e_vec)
                cdm_e_context_out = self.SA1(cdm_e_context_out)
                cat_e_out = torch.cat((global_context_out, cdm_e_context_out), dim=-1)
                cat_e_out = self.linear_double(cat_e_out)
            elif 'cdw' in self.args.local_context_focus:
                cdw_e_vec = self.emotions_feature_dynamic_weighted(local_context_ids, emotions)
                cdw_e_context_out = torch.mul(local_context_out, cdw_e_vec)
                cdw_e_context_out = self.SA1(cdw_e_context_out)
                cat_e_out = torch.cat((global_context_out, cdw_e_context_out), dim=-1)
                cat_e_out = self.linear_double(cat_e_out)
            elif 'fusion' in self.args.local_context_focus:
                cdm_e_vec = self.emotions_feature_dynamic_mask(local_context_ids, emotions)
                cdm_e_context_out = torch.mul(local_context_out, cdm_e_vec)
                cdw_e_vec = self.emotions_feature_dynamic_weighted(local_context_ids, emotions)
                cdw_e_context_out = torch.mul(local_context_out, cdw_e_vec)
                cat_e_out = torch.cat((global_context_out, cdw_e_context_out, cdm_e_context_out), dim=-1)
                cat_e_out = self.linear_triple(cat_e_out)

            sa_e_out = self.SA2(cat_e_out)
            pooled_e_out = self.pooler(sa_e_out)
        else:
            pooled_e_out = self.pooler(global_context_out)
        pooled_e_out = self.dropout(pooled_e_out)
        emotion_logits = self.emotion_classifier(pooled_e_out)

        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=0)
            loss_sen = CrossEntropyLoss()
            loss_ate = loss_fct(ate_logits.view(-1, self.num_labels), labels.view(-1))
            loss_emo = loss_sen(emotion_logits, emotion_labels)
            return loss_ate, loss_emo
        else:
            return ate_logits, emotion_logits
