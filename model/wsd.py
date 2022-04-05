import os, sys, json, torch, pickle
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from tqdm.autonotebook import tqdm as tqdm
import rowordnet
from transformers import AutoModel, AutoTokenizer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class WSD(pl.LightningModule):
    def __init__(self, model="dumitrescustefan/bert-base-romanian-cased-v1", lr=3e-04, model_max_length=512, synset_embedding_size=50):
        super(WSD, self).__init__()

        self.lr = lr
        self.model_max_length = model_max_length
        self.synset_embedding_size = synset_embedding_size

        print("Initializing RoWordNet ...")
        self.rown = rowordnet.RoWordNet()
        self.synset2id = {id: k+1 for k, id in enumerate(self.rown.synsets())} # 0 is for padding

        print(f"Initializing transformer model {model} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.transformer_model = AutoModel.from_pretrained(model)

        self.tokenizer.add_special_tokens({'additional_special_tokens': ["[START_ENTITY]","[STOP_ENTITY]"]})
        self.transformer_model.resize_token_embeddings(len(self.tokenizer))

        hidden_size = self.get_hidden_size()
        print(f"\tDetected hidden size is {hidden_size}")

        self.mixer = nn.Linear(4*hidden_size, synset_embedding_size)
        self.synset_embedding = nn.Embedding(num_embeddings=len(self.synset2id), embedding_dim=synset_embedding_size, padding_idx=0)

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        # add pad token
        self.validate_pad_token()

        # logging variables
        self.train_acc = []
        self.train_loss = []
        self.valid_acc = []
        self.valid_loss = []


    def validate_pad_token(self):
        if self.tokenizer.pad_token is not None:
            return
        if self.tokenizer.sep_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the SEP token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.sep_token
            return
        if self.tokenizer.eos_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the EOS token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            return
        if self.tokenizer.bos_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the BOS token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.bos_token
            return
        if self.tokenizer.cls_token is not None:
            print(f"\tNo PAD token detected, automatically assigning the CLS token as PAD.")
            self.tokenizer.pad_token = self.tokenizer.cls_token
            return
        raise Exception("Could not detect SEP/EOS/BOS/CLS tokens, and thus could not assign a PAD token which is required.")

    def get_hidden_size(self):
        inputs = self.tokenizer("text", return_tensors="pt")
        outputs = self.transformer_model(**inputs)
        return outputs.last_hidden_state.size(-1)

    def load(self, folder):
        pass

    def predict(self, text_prefix, text, text_suffix, synset_ids) -> str:
        pass

    def forward(self, prefix, word, suffix, sentence_with_special_tokens, word_positions, choices):
        """
        text is a batch of tokenized texts (list of tokenization result objects)
        word_positions is a batch of the position of the <WORD> token (list of ints)
        choices is a batch of the choices for each text (a list of lists)
        """
        # run texts through model
        prefix_model_output = self.transformer_model(input_ids=prefix["input_ids"].to(self.device), attention_mask=prefix["attention_mask"].to(self.device), return_dict=True).last_hidden_state # [batch_size, seq_len, hidden_size]
        word_model_output = self.transformer_model(input_ids=word["input_ids"].to(self.device), attention_mask=word["attention_mask"].to(self.device), return_dict=True).last_hidden_state # [batch_size, seq_len, hidden_size]
        suffix_model_output = self.transformer_model(input_ids=suffix["input_ids"].to(self.device), attention_mask=suffix["attention_mask"].to(self.device), return_dict=True).last_hidden_state # [batch_size, seq_len, hidden_size]
        sentence_model_output = self.transformer_model(input_ids=sentence_with_special_tokens["input_ids"].to(self.device), attention_mask=sentence_with_special_tokens["attention_mask"].to(self.device), return_dict=True).last_hidden_state # [batch_size, seq_len, hidden_size]

        # for each sentence compute cosine with all candidate synsets
        batch_size = sentence_model_output.size(0)
        sims = torch.zeros((batch_size, choices.size(1)), dtype=torch.float).to(self.device) # store cos values here, [batch_size, number_of_choices]
        for i in range(batch_size): # for each sentence
            # compute sentence representation
            prefix_embedding = torch.mean(prefix_model_output[i], dim=0) # [hidden_size]
            word_embedding = torch.mean(word_model_output[i], dim=0)  # [hidden_size]
            suffix_embedding = torch.mean(suffix_model_output[i], dim=0)  # [hidden_size]
            entity_embedding = sentence_model_output[i,word_positions[i],:] # [hidden_size]
            mixer_input = torch.cat([prefix_embedding, word_embedding, suffix_embedding, entity_embedding])
            projected_sentence_embedding = torch.tahn(self.mixer(mixer_input)) # [syn_emb_size]
            #----projected_sentence_embedding = self.mixer(sentence_embeddings[i,word_positions[i],:]) # reduce to synset embedding size, [batch_size, syn_emb_size]

            synset_embeddings = self.synset_embedding(choices[i]) # [choices, syn_emb_size]
            # repeat the sentence embedding the number of choices and compute cosinus with each embedding
            # cos ([batch_size, syn_emb], [batch_size, syn_emb]) => [batch_size]
            sim = self.cos(projected_sentence_embedding.repeat(choices[i].size(0),1), synset_embeddings)
            sims[i,:] = sim

        return sims

    def compute_loss(self, sims, choices, targets):
        """
        sims: padded tensor of [batch_size, number_of_choices]
        choices: padded tensor of [batch_size, number of choices]
        targets: tensor of [batch_size] with the correct id
        Loss is computed based on the cos similarity [-1,1] of the sentence with the synset embeddings
            the target synset should be 1, the others should be -1)
            compute 0.5*MSE(2-target) + 0.5*MSE(mean(non targets))
        """
        batch_size = sims.size(0)
        loss = torch.tensor(0.).to(self.device)
        for i in range(batch_size):
            instance_target_loss = torch.tensor(0.).to(self.device)
            instance_nontarget_loss = torch.tensor(0.).to(self.device)
            count_nontarget = 0
            for j in range(choices.size(1)): # for each choice
                if choices[i,j] == 0: # end of valid choices
                    break
                if choices[i,j] == targets[i]: # compute target loss
                    instance_target_loss = nn.functional.mse_loss(sims[i,j], torch.tensor(1.).to(self.device))
                else:
                    instance_nontarget_loss = instance_nontarget_loss + nn.functional.mse_loss(sims[i,j], torch.tensor(-1.).to(self.device))
                    count_nontarget += 1
            instance_nontarget_loss /= count_nontarget
            loss += 0.5*instance_target_loss + 0.5*instance_nontarget_loss

        return loss

    def training_step(self, batch, batch_idx):
        prefix = batch['prefix']
        word = batch['word']
        suffix = batch['suffix']
        sentence_with_special_tokens = batch['sentence_with_special_tokens']
        start_entity_position_special_tokens = batch['start_entity_position_special_tokens']
        choices = batch['choices']
        target = batch['target']

        # get predicted sims
        predicted_sims = self(prefix, word, suffix, sentence_with_special_tokens, start_entity_position_special_tokens, choices)

        # mask predicted sims
        mask = choices == 0  # true where choices are padded, false otherwise, mask has same shape as choices
        predicted_sims.masked_fill_(mask, 0)  # in-place filling

        # compute loss
        loss = self.compute_loss(predicted_sims, choices, target)

        # get predictions
        indices = torch.argmax(predicted_sims, dim=1)  # [batch_size]
        predicted_synsets = torch.gather(choices, 1, indices.unsqueeze(0)).squeeze(0)  # [batch_size]

        # compute accuracy
        batch_accuracy = (predicted_synsets == target).detach().cpu().numpy()

        self.train_acc.extend(batch_accuracy)
        self.train_loss.append(loss.detach().cpu().numpy())

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        accuracy = sum(self.train_acc) / len(self.train_acc)
        self.log("train/accuracy", accuracy, prog_bar=True)
        self.train_acc = []
        self.train_loss = []

    def validation_step(self, batch, batch_idx):
        prefix = batch['prefix']
        word = batch['word']
        suffix = batch['suffix']
        sentence_with_special_tokens = batch['sentence_with_special_tokens']
        start_entity_position_special_tokens = batch['start_entity_position_special_tokens']
        choices = batch['choices']
        target = batch['target']

        # get predicted sims
        predicted_sims = self(prefix, word, suffix, sentence_with_special_tokens, start_entity_position_special_tokens, choices)

        # mask predicted sims
        mask = choices == 0 # true where choices are padded, false otherwise, mask has same shape as choices
        predicted_sims.masked_fill_(mask, 0) # in-place filling

        # compute loss
        loss = self.compute_loss(predicted_sims, choices, target)

        # get predictions
        indices = torch.argmax(predicted_sims, dim=1) # [batch_size]
        predicted_synsets = torch.gather(choices, 1, indices.unsqueeze(0)).squeeze(0) # [batch_size]

        # compute accuracy
        batch_accuracy = (predicted_synsets == target).detach().cpu().numpy()

        self.valid_acc.extend(batch_accuracy)
        self.valid_loss.append(loss.detach().cpu().numpy())

        return {"loss": loss}

    def validation_epoch_end(self, outputs):
        accuracy = sum(self.valid_acc)/len(self.valid_acc)
        self.log("valid/accuracy", accuracy, prog_bar=True)
        self.valid_acc = []
        self.valid_loss = []

    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr, eps=1e-08)

class MyDataset(Dataset):
    def __init__(self, tokenizer:AutoModel, synset2id:{}, file_path: str):
        """
        Read a json file and process it.
        A json file has a list of elements(dicts) like:
        {
            "prefix_text": str,
            "text": str,
            "suffix_text": str,
            "choices": [] of strs (synset ids)
            "target": str (correct synset id)
        }
        """
        self.instances = []
        print("Reading json file: {}".format(file_path))

        # checks
        assert os.path.isfile(file_path)
        special_word_encoding = tokenizer("[START_ENTITY]", add_special_tokens=False)
        assert len(special_word_encoding['input_ids']) == 1, "There's a problem with the special token marker, it's not encoded properly it seems. Maybe your model does not support adding special tokens?"
        start_entity_id = special_word_encoding['input_ids'][0]

        with open(file_path, "r", encoding="utf8") as f:
            data = json.load(f)

        for entry in tqdm(data[:10000], desc=f"Tokenizing {file_path}"):
            if entry["target"] == "-1":
                continue

            """
            Compose marked sentence as: prefix + [ENTITY] + text + [\ENTITY] + suffix
            P.S. Not efficient, but it works, and have to account for weird stuff later on.
            """

            tokenized_prefix = tokenizer(entry["prefix_text"], add_special_tokens=False)
            tokenized_word = tokenizer(entry["text"], add_special_tokens=False)
            tokenized_suffix = tokenizer(entry["suffix_text"], add_special_tokens=False)
            sentence = f'{entry["prefix_text"]}[START_ENTITY] {entry["text"]} [STOP_ENTITY]{entry["suffix_text"]}'
            tokenized_sentence = tokenizer(sentence, add_special_tokens=False)
            tokenized_sentence_with_special_tokens = tokenizer(sentence, add_special_tokens=True)
            assert len(tokenized_sentence['input_ids']) == len(tokenized_prefix['input_ids']) + len(tokenized_word ['input_ids']) + len(tokenized_suffix['input_ids']) + 2
            start_entity_position = tokenized_sentence['input_ids'].index(start_entity_id)
            start_entity_position_special_tokens = tokenized_sentence_with_special_tokens['input_ids'].index(start_entity_id)

            instance = {
                "prefix": tokenized_prefix,
                "word": tokenized_word,
                "suffix": tokenized_suffix,
                "sentence": tokenized_sentence,
                "sentence_with_special_tokens": tokenized_sentence_with_special_tokens,
                "start_entity_position": start_entity_position,
                "start_entity_position_special_tokens": start_entity_position_special_tokens,
                "choices": [synset2id[choice] for choice in entry["choices"] if choice != "-1"],
                "target": synset2id[entry["target"]]
            }
            self.instances.append(instance)

        print(f"\tRead {len(self.instances)} entries.")

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, i):
        return self.instances[i]

class MyCollator():
    def __init__(self, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_value = tokenizer.pad_token_id

    def __call__(self, input_batch):
        """
        Input: batch is a list of batch_size number of instances; each instance is a dict, as given by MyDataset.__getitem__()
        Output:
            return a dict with the following batched tensors:
            "prefix": padded text prefix without the special marker, no special tokens, is a tokenizer object
            "word": padded target word without the special marker, no special tokens, is a tokenizer object
            "suffix": padded text suffix without the special marker, no special tokens, is a tokenizer object
            "sentence": padded sentence marked with the special markers, no special tokens, tokenizer object
            "sentence_with_special_tokens": padded sentence marked with the special markers, WITH special tokens, tokenizer object
            "start_entity_position": tensor with the int positions of the special start entity marker
            "start_entity_position_special_tokens": tensor with the int positions of the special start entity marker, for the special tokens sentence
            "choices": padded tensor with the list of the choices for each sentence
            "target": tensor with the int of the correct synset

        a padded tokenizer object is a dict that looks like:
            'input_ids': tensor([[101, 8667, 146, 112, 182, 170, 1423, 5650, 102],
                             [101, 1262, 1330, 5650, 102, 0, 0, 0, 0],
                             [101, 1262, 1103, 1304, 1304, 1314, 1141, 102, 0]]),
            'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0]]),
            'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1], ....
        """

        def pad_this (data, pad_value):
            """
            Given a list of lists, pad with pad_value if necessary
            """
            max_len = max([len(x) for x in data]) # get max len
            for i in range(len(data)):
                if len(data[i]) < max_len:
                    data[i].extend( [pad_value]*(max_len-len(data[i])) )
            return data

        def pad_and_convert_tokenizer_object(input_batch, key, pad_value):
            input_ids, token_type_ids, attention_mask = [], [], []
            for instance in input_batch:
                input_ids.append(instance[key]['input_ids'])
                #token_type_ids.append(instance[key]['token_type_ids'])
                attention_mask.append(instance[key]['attention_mask'])

            return {
                "input_ids": torch.tensor(pad_this(input_ids, pad_value=pad_value)),
                #"token_type_ids": torch.tensor(pad_this(token_type_ids, pad_value=0)),
                "attention_mask": torch.tensor(pad_this(attention_mask, pad_value=0))
            }

        # pad the tokenizer objects
        prefix = pad_and_convert_tokenizer_object(input_batch, "prefix", pad_value=self.pad_value)
        word = pad_and_convert_tokenizer_object(input_batch, "word", pad_value=self.pad_value)
        suffix = pad_and_convert_tokenizer_object(input_batch, "suffix", pad_value=self.pad_value)
        sentence = pad_and_convert_tokenizer_object(input_batch, "sentence", pad_value=self.pad_value)
        sentence_with_special_tokens = pad_and_convert_tokenizer_object(input_batch, "sentence_with_special_tokens", pad_value=self.pad_value)

        # collate rest of values
        start_entity_position, start_entity_position_special_tokens, choices, target = [], [], [], []
        for instance in input_batch:
            start_entity_position.append(instance['start_entity_position'])
            start_entity_position_special_tokens.append(instance['start_entity_position_special_tokens'])
            choices.append(instance['choices'])
            target.append(instance['target'])
        choices = pad_this(choices, 0) # pad choices

        # return nice dict
        return {
            "prefix": prefix,
            "word": word,
            "suffix": suffix,
            "sentence": sentence,
            "sentence_with_special_tokens": sentence_with_special_tokens,
            "start_entity_position": torch.tensor(start_entity_position),
            "start_entity_position_special_tokens": torch.tensor(start_entity_position_special_tokens),
            "choices": torch.tensor(choices),
            "target": torch.tensor(target)
        }


def train(model, gpus, batch_size, num_workers=0, accumulate_grad_batches=1):
    # load datasets
    train_dataset = MyDataset(file_path="train.json", tokenizer=model.tokenizer, synset2id=model.synset2id)
    val_dataset = MyDataset(file_path="dev.json", tokenizer=model.tokenizer, synset2id=model.synset2id)
    test_dataset = MyDataset(file_path="test.json", tokenizer=model.tokenizer, synset2id=model.synset2id)

    my_collator = MyCollator(tokenizer=model.tokenizer, max_seq_len=model.model_max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                  collate_fn=my_collator, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                collate_fn=my_collator, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                 collate_fn=my_collator, pin_memory=True)

    early_stop = EarlyStopping(
        monitor='valid/accuracy',
        patience=15,
        verbose=True,
        mode='max'
    )

    trainer = pl.Trainer(
        gpus=gpus,
        callbacks=[early_stop],
        #limit_train_batches=5,
        #limit_val_batches=2,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=1.0,
        enable_checkpointing=False
    )
    trainer.fit(model, train_dataloader, val_dataloader)


    return model


if __name__ == "__main__":
   # train the model
   wsd_model = WSD(model="racai/distilbert-base-romanian-cased")
   train(wsd_model, gpus=1, batch_size=8, accumulate_grad_batches=1, num_workers=0)



