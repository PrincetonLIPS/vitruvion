"""Network modules for img2cad."""

from typing import Optional
from typing_extensions import TypedDict


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer

from . import dataset, mobilenet


def generate_square_subsequent_mask(source: torch.Tensor) -> torch.Tensor:
    return source.new_full((source.shape[0], source.shape[0]), fill_value=float('-inf')).triu_(diagonal=1)


class TransformerModel(nn.Module):
    """Transformer component of image-conditional primitive generation model.
    
    This module assumes inputs have already been embedded with value,
    coordinate, and position indicators.
    """
    def __init__(self,
                 embed_dim,
                 num_heads,
                 fc_size,
                 num_layers,
                 dropout=0):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        encoder_layers = TransformerEncoderLayer(embed_dim, num_heads, fc_size,
            dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers,
            num_layers)

    def forward(self, src, use_mask=True):
        if use_mask:
            src_mask = generate_square_subsequent_mask(src)
        else:
            src_mask = None
        output = self.transformer_encoder(src, src_mask)
        return output
    

class TokenInput(TypedDict):
    val: torch.Tensor
    coord: torch.Tensor
    pos: torch.Tensor


class PrimitiveModel(nn.Module):
    """Autoregressive generative model of quantized primitives.

    Input primitive parameters are embedded based on their value as well as
    learned parameter/coordinate and position indicators.
    """
    def __init__(self,
                 num_bins,
                 max_entities,
                 embed_dim, 
                 fc_size, 
                 num_heads,
                 num_layers, 
                 dropout=0,
                 use_mask=True,
                 linear_decode=True):
        super(PrimitiveModel, self).__init__()

        embed_layers, out = create_prim_embed_layers(
            num_bins, max_entities, embed_dim)

        self.val_embed, self.coord_embed, self.pos_embed = embed_layers

        # Transformer
        self.transformer = TransformerModel(embed_dim,
                                            num_heads,
                                            fc_size,
                                            num_layers)
        self.use_mask = use_mask
        # Linear output layer for softmax
        self.linear_decode = linear_decode
        if linear_decode:
            self.out = out

    def _embed_tokens(self, src):
        # Embed
        val_embeddings = self.val_embed(src['val'])
        coord_embeddings = self.coord_embed(src['coord'])
        pos_embeddings = self.pos_embed(src['pos'])
        embeddings = val_embeddings + coord_embeddings + pos_embeddings
        return embeddings

    def _feed_transformer(self, embeddings):
        # Transpose to match transformer dimensions
        embeddings = torch.transpose(embeddings, 0, 1)
        # Pass to transformer
        output = self.transformer(embeddings, use_mask=self.use_mask)
        # Decode via linear layer for softmax
        if self.linear_decode:
            output = self.out(output)
        return torch.transpose(output, 0, 1)

    def forward(self, src: TokenInput, init_embedding: Optional[torch.Tensor]=None) -> torch.Tensor:
        """Applies the given primitive model.

        Parameters
        ----------
        src : TokenInput
            Tokenized input sequence.
        init_embedding : torch.Tensor, optional
            If not `None`, a torch Tensor representing arbitrary values
            to add to the sequence embedding at time zero.
        """

        # Embed
        embeddings = self._embed_tokens(src)

        if init_embedding is not None:
            embeddings[:,0,:] += init_embedding

        # Pass to transformer
        output = self._feed_transformer(embeddings)
        return output

    def __call__(self, src: TokenInput, init_embedding: Optional[torch.Tensor]=None) -> torch.Tensor:
        return super().__call__(src, init_embedding)


class SimpleConvNet(nn.Module):
    def __init__(self, out_dim):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.num_flat = 16 * 13 * 13
        self.fc1 = nn.Linear(self.num_flat, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ImageToPrimitiveModel(torch.nn.Module):
    """Image-conditional version of PrimitiveModel."""
    def __init__(self, num_bins: int, max_entities: int, embed_dim: int, fc_size: int, num_heads: int, num_layers: int, input_size=64, width_multiplier: float=0.5):
        super().__init__()

        self.primitive_model = PrimitiveModel(num_bins, max_entities, embed_dim, fc_size, num_heads, num_layers)

        self.convnet = mobilenet.MobileNetV3(
            output_dim=embed_dim, input_size=input_size, dropout=0.0,
            width_multiplier=width_multiplier, input_channels=1)

    def forward(self, src):
        img_embedding = self.convnet(src['img'])
        return self.primitive_model(src, img_embedding)


def create_prim_embed_layers(num_bins, max_entities, embed_dim):
    # Value embeddings
    num_val_embeddings = len(dataset.Token) + num_bins
    if dataset.INCLUDE_CONSTRUCTION:
        num_val_embeddings += 2
    val_embed = nn.Embedding(num_val_embeddings, embed_dim,
        padding_idx=dataset.Token.Pad)
    # Coordinate embeddings
    num_coord_embeddings = 2 + sum(
        [len(coord_map) for coord_map in dataset.COORD_TOKEN_MAP.values()])
    coord_embed = nn.Embedding(num_coord_embeddings, embed_dim,
        padding_idx=dataset.Token.Pad)
    # Position embeddings
    num_pos_embeddings = 3 + max_entities
    pos_embed = nn.Embedding(num_pos_embeddings, embed_dim,
        padding_idx=dataset.Token.Pad)
    # Also create output layer
    out = nn.Linear(embed_dim, num_val_embeddings)
    return (val_embed, coord_embed, pos_embed), out


class SeqImageToPrimitiveModel(torch.nn.Module):
    """Image-conditional primitive model using sequence of patch embeddings."""
    def __init__(self, num_bins: int, max_entities: int, embed_dim: int,
        fc_size: int, num_heads: int, num_layers: int, input_size: int=64,
        patch_size=8, dropout=0, num_recognition_layers: Optional[int]=None):
        """Create a new image to primitive model based on a visual transformer recognition layer.

        Parameters
        ----------
        num_bins : int
            Number of bins used for quantizing position.
        max_entities : int
            Maximum number of entities in the sequence.
        embed_dim : int
            Dimension of embedding
        fc_size : int
            Size of fully connected readout layer.
        num_heads : int
            Number of attention heads to use.
        num_layers : int
            Number of transformer layers to use.
        input_size : int
            Size of the input images in pixels.
        patch_size : int
            Size of the patches to use when transforming the image to sequence.
            Must divide the ``input_size``.
        num_recognition_layers : int, optional
            If not `None`, number of layers to use for recognition transformer.
            Otherwise, uses the same number of layers as the primitives transformer.
        """
        super().__init__()

        # Image patch embeddings
        if input_size % patch_size != 0:
            raise ValueError("Patch size must divide input size.")
        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_size**2, embed_dim)
        num_patch_embeddings = (input_size // patch_size)**2
        self.patch_pos_embed = nn.Embedding(num_patch_embeddings, embed_dim)
        self.patch_extract = nn.Unfold(kernel_size=patch_size,
                                       stride=patch_size)

        # Primitive token embeddings
        embed_layers, out = create_prim_embed_layers(
            num_bins, max_entities, embed_dim)
        self.val_embed, self.coord_embed, self.pos_embed = embed_layers

        # Patch encoder
        encoder_layers = TransformerEncoderLayer(embed_dim, num_heads, fc_size,
            dropout)
        self.trans_encoder = TransformerEncoder(
            encoder_layers,
            num_recognition_layers if num_recognition_layers is not None else num_layers)

        # Primitive decoder
        decoder_layers = TransformerDecoderLayer(embed_dim, num_heads, fc_size,
            dropout)
        self.trans_decoder = TransformerDecoder(decoder_layers, num_layers)

        # Linear output layer for softmax
        self.out = out

        self.input_size = input_size

    def _embed_tokens(self, src):
        """Embed both tokens and image patches.
        """
        if src['img'].shape[-1] != self.input_size:
            raise ValueError(f'Input image does not have the expected size. Expected size {self.input_size} but got {src["img"].shape[-1]}')

        # Embed tokens
        val_embeddings = self.val_embed(src['val'])
        coord_embeddings = self.coord_embed(src['coord'])
        pos_embeddings = self.pos_embed(src['pos'])
        embeddings = val_embeddings + coord_embeddings + pos_embeddings

        # Embed image patches
        patches = self.patch_extract(src['img'])
        patches = torch.transpose(patches, 1, 2)  # patches are by row now
        patch_embeddings = self.patch_embed(patches)

        # Add patch positional embeddings
        patch_pos_toks = torch.arange(patch_embeddings.shape[1],
            device=patch_embeddings.device)
        patch_pos_toks = patch_pos_toks[None, ...]
        patch_pos_toks = patch_pos_toks.expand(patch_embeddings.shape[0], -1)
        patch_pos_embeddings = self.patch_pos_embed(patch_pos_toks)
        patch_embeddings = patch_embeddings + patch_pos_embeddings

        return patch_embeddings, embeddings

    def _feed_transformer(self, patch_embeddings, prim_embeddings):
        # Transpose to match transformer dimensions
        patch_embeddings = torch.transpose(patch_embeddings, 0, 1)
        prim_embeddings = torch.transpose(prim_embeddings, 0, 1)
        # Run patch embeddings through encoder
        memory = self.trans_encoder(patch_embeddings)
        # Run memory and prim_embeddings through decoder
        tgt_mask = generate_square_subsequent_mask(prim_embeddings)
        output = self.trans_decoder(prim_embeddings, memory, tgt_mask)
        # Decode via linear layer for softmax
        output = self.out(output)
        return torch.transpose(output, 0, 1)

    def forward(self, src) -> torch.Tensor:
        # Embed
        patch_embeddings, prim_embeddings = self._embed_tokens(src)
        # Pass to transformer
        output = self._feed_transformer(patch_embeddings, prim_embeddings)
        return output


class CondPrimModel(PrimitiveModel):  # TODO: dry-ify with Image model
    """Conditional version of PrimitiveModel."""
    def __init__(self,
                 num_bins,
                 max_entities,
                 embed_dim,
                 fc_size,
                 num_heads,
                 num_layers,
                 dropout=0,
                 use_mask=True,
                 linear_decode=True):
        super(CondPrimModel, self).__init__(num_bins,
                                            max_entities,
                                            embed_dim,
                                            fc_size,
                                            num_heads,
                                            num_layers,
                                            dropout=dropout,
                                            use_mask=use_mask,
                                            linear_decode=linear_decode)

    def forward(self, src, context):
        # Embed
        embeddings = self._embed_tokens(src)
        # Add context embedding to start token position
        embeddings[:,0,:context.shape[1]] += context
        # Pass to transformer
        output = self._feed_transformer(embeddings)
        return output


class PrimitiveAE(nn.Module):
    """Autoencoder of quantized primitives.

    Primitives are first embedded via a non-masked PrimitiveModel.
    A portion of the last primitive's final embedding serves as the entire
    sequence's encoding. This encoding then conditions a masked PrimitiveModel
    which attempts to reconstruct the original input.
    """
    def __init__(self,
                 encode_dim,
                 num_bins,
                 max_entities,
                 embed_dim,
                 fc_size,
                 num_heads,
                 num_layers,
                 dropout=0):
        super(PrimitiveAE, self).__init__()
        # Encoder primitive model
        self.encoder = PrimitiveModel(num_bins, max_entities, embed_dim,
            fc_size, num_heads, num_layers, dropout, use_mask=False,
            linear_decode=False)
        # Decoder primitive model
        self.decoder = CondPrimModel(num_bins, max_entities, embed_dim,
            fc_size, num_heads, num_layers, dropout, use_mask=True,
            linear_decode=True)
        self.encode_dim = encode_dim

    def encode(self, src):
        encoding = self.encoder(src)[:, -1, :self.encode_dim]
        return encoding

    def forward(self, src):
        # Learned sequence-level encoding
        encoding = self.encode(src)
        # Decode
        output = self.decoder(src, encoding)
        return output
