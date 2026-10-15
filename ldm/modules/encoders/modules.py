import copy
import math

import torch
import torch.nn as nn
from functools import partial
import clip
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel
import kornia
import torch.nn.functional as F
from ldm.modules.x_transformer import Encoder, \
    TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


class TransformerEmbedder(AbstractEncoder):
    """Some transformer encoder layers"""

    def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
        super().__init__()
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer))

    def forward(self, tokens):
        tokens = tokens.to(self.device)  # meh
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, x):
        return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""

    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


class BERTEmbedder(AbstractEncoder):
    """Uses the BERT tokenizr model and add some transformer encoder layers"""

    def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
                 device="cuda", use_tokenizer=True, embedding_dropout=0.0):
        super().__init__()
        self.use_tknz_fn = use_tokenizer
        if self.use_tknz_fn:
            self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
        self.device = device
        self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
                                              attn_layers=Encoder(dim=n_embed, depth=n_layer),
                                              emb_dropout=embedding_dropout)

    def forward(self, text):
        if self.use_tknz_fn:
            tokens = self.tknz_fn(text)  # .to(self.device)
        else:
            tokens = text
        z = self.transformer(tokens, return_embeddings=True)
        return z

    def encode(self, text):
        # output of length 77
        return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest', 'linear', 'bilinear', 'trilinear', 'bicubic', 'area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)

        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)





class FrozenCLIPTextEmbedder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """

    def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
        super().__init__()
        self.model, _ = clip.load(version, jit=False, device="cpu")
        self.device = device
        self.max_length = max_length
        self.n_repeat = n_repeat
        self.normalize = normalize

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = clip.tokenize(text).to(self.device)
        z = self.model.encode_text(tokens)
        if self.normalize:
            z = z / torch.linalg.norm(z, dim=1, keepdim=True)
        return z

    def encode(self, text):
        z = self(text)
        if z.ndim == 2:
            z = z[:, None, :]
        z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
        return z


class FrozenClipImageEmbedder(nn.Module):
    """
        Uses the CLIP image encoder.
        """

    def __init__(
            self,
            model,
            jit=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            antialias=False,
    ):
        super().__init__()
        self.model, _ = clip.load(name=model, device=device, jit=jit)

        self.antialias = antialias

        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224),
                                   interpolation='bicubic', align_corners=True,
                                   antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1,1]
        return self.model.encode_image(self.preprocess(x))



### original
class IDEmbedder(nn.Module):
    def __init__(self, device='cuda', embedding_dim=128):
        super().__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.half_embedding_dim = self.embedding_dim // 2
        self.half_embedding_size = 77 * self.half_embedding_dim
        self.embedding_size = 77 * embedding_dim
        self.fc1 = nn.Linear(512, self.half_embedding_size)
        self.fc2 = nn.Linear(self.half_embedding_size, self.embedding_size)
        self.leakyrelu = nn.LeakyReLU()
        # self.LearnableToken = nn.Parameter(torch.randn(1, 77, 640))

    def forward(self, x):
        id_embedding = self.fc1(x)
        id_embedding = self.leakyrelu(id_embedding)
        id_embedding = self.fc2(id_embedding)
        id_embedding = self.leakyrelu(id_embedding)
        id_embedding = id_embedding.view(id_embedding.size(0), 77, self.embedding_dim)
        # id_embedding = torch.cat((id_embedding, self.LearnableToken), dim=2)
        return id_embedding


class IDExtractor(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.netArc_checkpoint = '/mnt/data/zhonghuanlong/models/faceReclaim/checkpoints/arcface_checkpoint.tar'
        self.netArc_checkpoint = torch.load(self.netArc_checkpoint)
        self.netArc = self.netArc_checkpoint
        self.device = device
        self.netArc.eval()
        self.netArc.requires_grad_(False)

    def forward(self, x):
        model = self.netArc
        model.eval()
        images_id_downsample = F.interpolate(x, size=(112, 112), mode='bicubic')
        images_latend_id =model(images_id_downsample)
        id = F.normalize(images_latend_id, p=2, dim=1)

        return id



class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self, version="/mnt/data/zhonghuanlong/models/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state

        return z

    def encode(self, text):
        return self(text)


# class EnhancedTextEmbedder(nn.Module):
#     def __init__(self):
#         super(EnhancedTextEmbedder, self).__init__()
#         self.id_projection = nn.Linear(512, 768)
#         self.id_fc = nn.Linear(768, 768)
#
#         self.result_face_projection = nn.Linear(4096, 1536)
#         self.result_face_fc = nn.Linear(1536, 1536)
#
#     def forward(self, id, text_prompt, result_feature):
#         identity_index = 1
#         result_index = 5
#         face_index = 6
#
#         identity_text_embedding = text_prompt[:, identity_index, :].unsqueeze(1)
#         result_text_embedding = text_prompt[:, result_index, :].unsqueeze(1)
#         face_text_embedding = text_prompt[:, face_index, :].unsqueeze(1)
#         result_face_text_embedding = torch.cat([result_text_embedding, face_text_embedding], dim=2)
#
#
#         # source identity
#         id = self.id_projection(id)
#         enhanced_identity_text_embedding = id + identity_text_embedding
#         enhanced_identity_text_embedding = self.id_fc(enhanced_identity_text_embedding)
#
#         # result face
#         result_feature = result_feature.mean(dim=1)
#         result_embedding = result_feature.flatten(start_dim=1)
#         result_embedding = self.result_face_projection(result_embedding)
#         enhanced_result_face_text_embedding = result_embedding + result_face_text_embedding
#         enhanced_result_face_text_embedding = self.result_face_fc(enhanced_result_face_text_embedding)
#         enhanced_result_text_embedding, enhanced_face_text_embedding = torch.split(enhanced_result_face_text_embedding,
#                                                                                    768, dim=2)
#
#         # updated_text_prompt = copy.deepcopy(text_prompt)
#         updated_text_prompt = text_prompt.clone()
#         updated_text_prompt[:, identity_index, :] = enhanced_identity_text_embedding.squeeze(1)
#         updated_text_prompt[:, result_index, :] = enhanced_result_text_embedding.squeeze(1)
#         updated_text_prompt[:, face_index, :] = enhanced_face_text_embedding.squeeze(1)
#         return updated_text_prompt


class EnhancedTextEmbedder(nn.Module):
    def __init__(self):
        super(EnhancedTextEmbedder, self).__init__()
        self.id_projection = nn.Linear(512, 768)
        self.id_fc = nn.Linear(768, 768)

        self.result_face_projection = nn.Linear(4096, 1536)
        self.result_face_fc = nn.Linear(1536, 1536)

    def forward(self, id, text_prompt, result_feature):

        identity_index = 1
        result_index = 5
        face_index = 6

        # Get embeddings for different indices in the text_prompt
        identity_text_embedding = text_prompt[:, identity_index, :].unsqueeze(1).contiguous()
        result_text_embedding = text_prompt[:, result_index, :].unsqueeze(1).contiguous()
        face_text_embedding = text_prompt[:, face_index, :].unsqueeze(1).contiguous()

        # Concatenate result and face embeddings along the last dimension
        result_face_text_embedding = torch.cat([result_text_embedding, face_text_embedding], dim=2)

        # Process identity
        id = self.id_projection(id)  # Shape: (batch_size, 768)

        batch_size_id = id.size(0)
        batch_size_text_embedding = identity_text_embedding.size(0)

        if batch_size_id != batch_size_text_embedding:
            # If the batch sizes do not match, pad `id` with zeros to match `identity_text_embedding` size
            padding_size = batch_size_text_embedding - batch_size_id
            id = F.pad(id, (0, 0, 0, padding_size), value=0)  # Pad the first dimension

        # To ensure the addition works, we expand identity_text_embedding to match id's shape
        identity_text_embedding = identity_text_embedding.squeeze(1)  # Shape: (batch_size, embedding_dim)
        try:
            enhanced_identity_text_embedding = id + identity_text_embedding  # Shape: (batch_size, 768)
            enhanced_identity_text_embedding = self.id_fc(enhanced_identity_text_embedding)

            # Process result and face embeddings
            result_feature = result_feature.mean(dim=1)  # Shape: (batch_size, feature_dim)
            result_embedding = result_feature.flatten(start_dim=1)  # Shape: (batch_size, 4096)
            result_embedding = self.result_face_projection(result_embedding)  # Shape: (batch_size, 1536)

            # Add result_embedding to the concatenated result and face text embeddings
            result_face_text_embedding = result_face_text_embedding.squeeze(1)  # Shape: (batch_size, 2 * embedding_dim)
            enhanced_result_face_text_embedding = result_embedding + result_face_text_embedding
            enhanced_result_face_text_embedding = self.result_face_fc(enhanced_result_face_text_embedding)

            # Split the enhanced embedding back into result and face embeddings
            enhanced_result_text_embedding, enhanced_face_text_embedding = torch.split(enhanced_result_face_text_embedding,
                                                                                       768, dim=1)

            # Update the text_prompt
            updated_text_prompt = text_prompt.clone()
            updated_text_prompt[:, identity_index, :] = enhanced_identity_text_embedding
            updated_text_prompt[:, result_index, :] = enhanced_result_text_embedding
            updated_text_prompt[:, face_index, :] = enhanced_face_text_embedding

            return updated_text_prompt
        except RuntimeError:
            print(id.size(), identity_text_embedding.size())


if __name__ == "__main__":

    embedder = FrozenCLIPEmbedder().to("cuda")  # Ensure the model is on the correct device
    # text = "remove identity from the result face"
    # encoded_text = embedder.encode(text)
    # print(encoded_text.size())
    model = EnhancedTextEmbedder().to("cuda")
    id = torch.randn(4, 512).to("cuda")
    text_prompt = torch.randn(6, 77, 768).to("cuda")
    result_embedding = torch.randn(6, 4, 64, 64).to("cuda")
    updated_text_prompt = model(id, text_prompt, result_embedding)
    # print("Identity diff norm:",
    #       torch.norm(updated_text_prompt[:, identity_index, :] - encoded_text[:, identity_index, :]))

    print(updated_text_prompt.size())



    # x = torch.randn(1, 3, 224, 224).to('cuda')
    # exactor = IDExtractor()
    # y = exactor.extract_id(x)
    # print(y.size())
    # Initialize the FrozenCLIPEmbedder
    # module = IDEmbedder().cuda()
    # x = torch.randn(1, 1, 512).cuda()
    # y = module(x)
    # print(y.size())

