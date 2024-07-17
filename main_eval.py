from torchvision import transforms, datasets
import torch
import torch.optim as optim
import torch.nn.functional as F
import clip
from PIL import Image
from sklearn.metrics import accuracy_score, normalized_mutual_info_score as nmi, rand_score as ri, adjusted_rand_score as ar, f1_score as f1
from sklearn.cluster import KMeans
import numpy as np
import os
import random
from parse import get_parse
import time


data_transfrom = transforms.Compose([
    transforms.ToTensor()
])


num_tokens = 77

def encode_text_with_learnt_tokens(self, text, asterix_token, learnable_codes):
    placeholder_rows, placeholder_cols = torch.where(text == asterix_token)
    x = self.token_embedding(text).type(self.dtype)

    for i in range(len(placeholder_rows)):
        x_i_longer = torch.cat((x[placeholder_rows[i]][:placeholder_cols[i]], learnable_codes[i].unsqueeze(0), x[placeholder_rows[i]][placeholder_cols[i]+1:]), 0).to(learnable_codes.dtype)
        x[i] = x_i_longer[:num_tokens]

    x = x.permute(1, 0, 2)
    x = self.transformer(x)
    x = x.permute(1, 0, 2)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
    return x

def reference_word_embedding(self, text):
    embedding = []
    idx = torch.argmin(text, dim=1) - 1
    x = self.token_embedding(text).type(self.dtype)
    for i in range(len(text)):
        tmp = torch.mean(x[i][1:idx[i]], dim=0)
        embedding.append(tmp)
    return torch.stack(embedding)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    args = get_parse()
    alpha = args.alpha


    # Image
    # # Fruit
    dataset_path_dict = {
        "fruit": ["dataset/fruit/species/", "dataset/fruit/color/"]
    }
    dataset_prompt_dict = {
        "fruit": ["Fruit with a species of", "Fruit with a color of"]
    }
    dataset_gpt_dict = {
        "fruit": ["apples, oranges, bananas, strawberries, grapes, raspberries, blueberries, cherries, pears, plums, peaches, nectarines, pineapple, kiwi, watermelon, cantaloupe, apricots", "red, yellow, green, orange, purple, blue"]
    }
    path_list = dataset_path_dict[args.dataset]
    prompt_list = dataset_prompt_dict[args.dataset]
    gpt_list = dataset_gpt_dict[args.dataset]

    # Text
    batch_size = 50
    text_batch = args.batch_size
    for _i, (_data_path, _prompt, _gpt) in enumerate(zip(path_list, prompt_list, gpt_list)):

        model, preprocess = clip.load('ViT-B/32', device, jit=False)

        funcType = type(model.encode_text)
        model.encode_text_with_learnt_tokens = funcType(encode_text_with_learnt_tokens, model)
        model.encode_reference_word = funcType(reference_word_embedding, model)
        model = model.float()

        for param in model.parameters():
            param.requires_grad = False
        for param in model.visual.parameters():
            param.requires_grad = True

        torch.cuda.empty_cache()
        print("Data path:", _data_path, _prompt)
        img1 = datasets.ImageFolder(root=_data_path, transform=data_transfrom)
        print(img1.class_to_idx, len(img1.class_to_idx))
        idx_to_class = {}
        for key, value in img1.class_to_idx.items():
            idx_to_class[value] = key
        gpt_candidate = _gpt.split(", ")
        print('candidate', len(gpt_candidate), gpt_candidate)

        label_list = []
        image_prob_list = []
        pred_label_list = []
        image_embedding_list = []
        idx = 0
        this_datetime = time.strftime("%m-%d-%Hh%Mm%Ss")

        input_image_list = []
        for img_path, label in img1.imgs:
            if idx % 1000 == 0:
                print(idx)
            idx += 1
            label_list.append(str(label))
            input_image_list.append(preprocess(Image.open(img_path)))


        prompt = prompt_list[_i] + " *"
        asterix_token = clip.tokenize(["*"]).to(device)[0][1]
        prompt_token = clip.tokenize([prompt] * len(input_image_list)).to(device)
        word_token = clip.tokenize(["*"] * len(input_image_list)).to(device)
        concept_prompt_token = clip.tokenize([_data_path.split('/')[2]] * len(input_image_list)).to(device)

        num_images = len(input_image_list)
        num_batch = num_images / batch_size if num_images % batch_size == 0 else num_images // batch_size + 1
        batch_list = [i for i in range(0, num_images, batch_size)]

        text_batch_list = [i for i in range(0, num_images, text_batch)]
        if batch_list[-1] < num_images:
            batch_list.append(num_images)
        if text_batch_list[-1] < num_images:
            text_batch_list.append(num_images)
        batch_block = []
        for i in range(len(text_batch_list) - 1):
            batch_block.append([text_batch_list[i], text_batch_list[i+1]])

        text_features = []
        concept_token = []
        gpt_pred_label_list = []

        image_inputs = torch.tensor(np.stack(input_image_list))

        # obtain reference word
        gpt_inputs = clip.tokenize([f"{_v}" for _v in gpt_candidate]).to(device)
        with torch.no_grad():
            # reference word embedding
            gpt_embeddings = model.encode_reference_word(gpt_inputs)

        # * embedding
        with torch.no_grad():
            for i in range(len(text_batch_list) - 1):
                _prompt_token = word_token[text_batch_list[i]: text_batch_list[i+1]]
                text_features.extend(model.encode_text(_prompt_token))
        text_features = torch.stack(text_features)

        # initialize p_ij with *
        trainable_estimated_tokens = torch.nn.Embedding.from_pretrained(text_features, freeze=False)  # create learnble tokens

        optimizer = optim.Adagrad([
            {'params': trainable_estimated_tokens.parameters()},
            {'params': model.visual.proj, 'lr': 1e-5}], lr=args.lr, weight_decay=args.weight_decay)
        ones = torch.ones(len(input_image_list)).to(torch.float16).to(device)
        criterion = torch.nn.CosineEmbeddingLoss()


        loss_log = [[], []]
        clip_embeddings = []

        for i in range(args.epoch):
            embedding_list = []
            loss = []
            random.shuffle(batch_block)
            for j, _block in enumerate(batch_block):
                # generate image embeddings
                _image_inputs = image_inputs[_block[0]: _block[1]].to(device)
                _image_embeddings = model.encode_image(_image_inputs)

                # compute weight, compute proxy word embedding
                _trainable_estimated_tokens = trainable_estimated_tokens.weight[_block[0]: _block[1]].to(device)
                _trainable_estimated_tokens = torch.mm(_trainable_estimated_tokens, gpt_embeddings.transpose(0, 1))
                _trainable_estimated_tokens = torch.mm(F.softmax(_trainable_estimated_tokens, dim=1), gpt_embeddings)
                # calculate prompt embedding with proxy word embedding
                _prompt_token = prompt_token[_block[0]: _block[1]]
                _clip_embeddings = model.encode_text_with_learnt_tokens(_prompt_token, asterix_token, _trainable_estimated_tokens)
                _clip_embeddings = F.normalize(_clip_embeddings, dim=-1)
                # Loss function
                _loss = criterion(_image_embeddings, _clip_embeddings, ones[_block[0]: _block[1]])
                loss.append(_loss.cpu().detach().item())

                optimizer.zero_grad()
                _loss.backward()
                optimizer.step()


            if i % 10 == 0:
                print(f'Epoch {i}\tLoss: {np.mean(loss):.4}')
                loss_log[0].append(i)
                loss_log[1].append(np.mean(loss))


            if  (i + 1) % 100 == 0:
                text_embedding_list = []
                word_embedding_list = []
                word_eod_embedding_list = []
                image_embedding_list = []
                for j in range(len(text_batch_list) - 1):
                    # compute weight and aggregate the reference word embeddings
                    _trainable_estimated_tokens = trainable_estimated_tokens.weight[text_batch_list[j]: text_batch_list[j + 1]].to(device)
                    _trainable_estimated_tokens = torch.mm(_trainable_estimated_tokens, gpt_embeddings.transpose(0, 1))
                    _trainable_estimated_tokens = torch.mm(F.softmax(_trainable_estimated_tokens, dim=1), gpt_embeddings)
                    # proxy word embedding
                    word_embedding_list.extend(_trainable_estimated_tokens.cpu().detach().numpy())
                    _prompt_token = prompt_token[text_batch_list[j]: text_batch_list[j + 1]]
                    # prompt embedding
                    _clip_embeddings = model.encode_text_with_learnt_tokens(_prompt_token, asterix_token,
                                                                            _trainable_estimated_tokens)
                    _clip_embeddings = F.normalize(_clip_embeddings, dim=-1)
                    text_embedding_list.extend(_clip_embeddings.cpu().detach().numpy())
                    _word_token = word_token[text_batch_list[j]: text_batch_list[j + 1]]
                    # proxy word eod embedding
                    _word_eod_embeddings = model.encode_text_with_learnt_tokens(_word_token, asterix_token,
                                                                            _trainable_estimated_tokens)
                    word_eod_embedding_list.extend(_word_eod_embeddings.cpu().detach().numpy())
                    # image embedding
                    with torch.no_grad():
                        _image_inputs = image_inputs[text_batch_list[j]: text_batch_list[j + 1]].to(device)
                        _image_embeddings = model.encode_image(_image_inputs)
                        image_embedding_list.extend(_image_embeddings.cpu().detach().numpy())

                prompt_embeddings = np.stack(text_embedding_list)
                word_embeddings = np.stack(word_embedding_list)
                word_eod_embeddings = np.stack(word_eod_embedding_list)
                image_embeddings = np.stack(image_embedding_list)
                combined_embeddings = np.hstack((word_embeddings, image_embeddings))



                nmi_list, ar_list, ri_list, f1_list = [], [], [], []
                for _i in range(10):
                    kmeans = KMeans(n_clusters=len(img1.class_to_idx), random_state=_i, n_init="auto").fit(prompt_embeddings)
                    pred_res = kmeans.labels_
                    nmi_list.append(nmi(label_list, pred_res))
                    ar_list.append(ar(label_list, pred_res))
                    ri_list.append(ri(label_list, pred_res))
                print(f"Prompt: NMI:{np.mean(nmi_list):.4f}\tARI:{np.mean(ar_list):.4f}\tRI:{np.mean(ri_list):.4f}")

                nmi_list2, ar_list2, ri_list2, f1_list2 = [], [], [], []
                for _i in range(10):
                    kmeans = KMeans(n_clusters=len(img1.class_to_idx), random_state=_i, n_init="auto").fit(word_embeddings)
                    pred_res = kmeans.labels_
                    nmi_list2.append(nmi(label_list, pred_res))
                    ar_list2.append(ar(label_list, pred_res))
                    ri_list2.append(ri(label_list, pred_res))
                print(f"Word: NMI:{np.mean(nmi_list2):.4f}\tARI:{np.mean(ar_list2):.4f}\tRI:{np.mean(ri_list2):.4f}")

                nmi_list2, ar_list2, ri_list2, f1_list2 = [], [], [], []
                nmi_list2, ar_list2, ri_list2 = [], [], []
                for _i in range(10):
                    kmeans = KMeans(n_clusters=len(img1.class_to_idx), random_state=_i, n_init="auto").fit(word_eod_embeddings)
                    pred_res = kmeans.labels_
                    nmi_list2.append(nmi(label_list, pred_res))
                    ar_list2.append(ar(label_list, pred_res))
                    ri_list2.append(ri(label_list, pred_res))
                print(f"Word EOD: NMI:{np.mean(nmi_list2):.4f}\tARI:{np.mean(ar_list2):.4f}\tRI:{np.mean(ri_list2):.4f}")

                nmi_list2, ar_list2, ri_list2, f1_list2 = [], [], [], []
                nmi_list2, ar_list2, ri_list2 = [], [], []
                for _i in range(10):
                    kmeans = KMeans(n_clusters=len(img1.class_to_idx), random_state=_i, n_init="auto").fit(image_embeddings)
                    pred_res = kmeans.labels_
                    nmi_list2.append(nmi(label_list, pred_res))
                    ar_list2.append(ar(label_list, pred_res))
                    ri_list2.append(ri(label_list, pred_res))
                print(f"Image: NMI:{np.mean(nmi_list2):.4f}\tARI:{np.mean(ar_list2):.4f}\tRI:{np.mean(ri_list2):.4f}")

                nmi_list2, ar_list2, ri_list2, f1_list2 = [], [], [], []
                for _i in range(10):
                    kmeans = KMeans(n_clusters=len(img1.class_to_idx), random_state=_i, n_init="auto").fit(combined_embeddings)
                    pred_res = kmeans.labels_
                    nmi_list2.append(nmi(label_list, pred_res))
                    ar_list2.append(ar(label_list, pred_res))
                    ri_list2.append(ri(label_list, pred_res))
                print(f"Combined: NMI:{np.mean(nmi_list2):.4f}\tARI:{np.mean(ar_list2):.4f}\tRI:{np.mean(ri_list2):.4f}")
