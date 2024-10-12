from typing import Type

import torch
from torch import nn
from torch.nn import functional as F


class RNNBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout_rate: float,
        rnn_type: nn.GRU | nn.LSTM | nn.RNN,
    ) -> None:
        super().__init__()
        self.rnn = rnn_type(
            input_size,
            hidden_size,
            dropout=dropout_rate,
            batch_first=False,
            bidirectional=True,
        )
        self.hidden_size = hidden_size
        self.ln = nn.LayerNorm([hidden_size])

    def forward(self, x, h=None):
        x_bidirectional_embed, h_next = self.rnn(x, h)
        x_embed = (
            x_bidirectional_embed[:, :, : self.hidden_size]
            + x_bidirectional_embed[:, :, self.hidden_size :]
        )
        x_normed = self.ln(x_embed)
        return x_normed, h_next


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_conv_num: int,
        input_kernel: list[int],
        hidden_kernel: list[int],
        output_kernel: list[int],
        input_padding: list[int],
        hidden_padding: list[int],
        output_padding: list[int],
        input_stride: list[int],
        hidden_stride: list[int],
        output_stride: list[int],
        hidden_channels: list[int],
        output_channels: list[int],
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels=1,
                out_channels=hidden_channels,
                kernel_size=input_kernel,
                padding=input_padding,
                stride=input_stride,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.Hardtanh(0, 20),
        ]
        for _ in range(hidden_conv_num):
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        kernel_size=hidden_kernel,
                        padding=hidden_padding,
                        stride=hidden_stride,
                    ),
                    nn.BatchNorm2d(hidden_channels),
                    nn.Hardtanh(0, 20),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(
                    in_channels=hidden_channels,
                    out_channels=output_channels,
                    kernel_size=output_kernel,
                    padding=output_padding,
                    stride=output_stride,
                ),
                nn.BatchNorm2d(output_channels),
                nn.Hardtanh(0, 20),
            ]
        )
        self.layers = nn.Sequential(*layers)
        self.hidden_conv_num = hidden_conv_num
        self.input_kernel = input_kernel
        self.hidden_kernel = hidden_kernel
        self.output_kernel = output_kernel
        self.input_padding = input_padding
        self.hidden_padding = hidden_padding
        self.output_padding = output_padding
        self.input_stride = input_stride
        self.hidden_stride = hidden_stride
        self.output_stride = output_stride
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

    def forward(self, spectrogram):
        output = self.layers(spectrogram)
        return torch.flatten(output, start_dim=1, end_dim=2)

    @staticmethod
    def get_size_after_conv(length, kernel, padding, stride):
        return (length + 2 * padding[0] - kernel[0]) // stride[0] + 1

    def get_output_lengths(self, input_length):
        cur_length = input_length
        cur_length = self.get_size_after_conv(
            cur_length, self.input_kernel, self.input_padding, self.input_stride
        )
        for _ in range(self.hidden_conv_num):
            cur_length = self.get_size_after_conv(
                cur_length, self.hidden_kernel, self.hidden_padding, self.hidden_stride
            )
        cur_length = self.get_size_after_conv(
            cur_length, self.output_kernel, self.output_padding, self.output_stride
        )
        return cur_length * self.output_channels


class Decoder(nn.Module):
    def __init__(
        self,
        num_rnn_layers: int,
        input_size: int,
        hidden_size: int,
        dropout_rate: float,
        rnn_type: str,
    ) -> None:
        super().__init__()
        assert num_rnn_layers > 0
        if rnn_type == "gru":
            rnn_type_class = nn.GRU
        elif rnn_type == "lstm":
            rnn_type_class = nn.LSTM
        else:
            rnn_type_class = nn.RNN
        self.layers = nn.ModuleList(
            [
                RNNBlock(input_size, hidden_size, dropout_rate, rnn_type_class),
                *[
                    RNNBlock(hidden_size, hidden_size, dropout_rate, rnn_type_class)
                    for _ in range(num_rnn_layers)
                ],
            ]
        )

    def forward(self, x):
        h = None
        for layer in self.layers:
            x, h = layer(x, h)
        return x


class DeepSpeech2Model(nn.Module):
    def __init__(
        self,
        encoder_n_feats: int,
        encoder_hidden_conv_num: int,
        encoder_input_kernel: list[int],
        encoder_hidden_kernel: list[int],
        encoder_output_kernel: list[int],
        encoder_input_padding: list[int],
        encoder_hidden_padding: list[int],
        encoder_output_padding: list[int],
        encoder_input_stride: list[int],
        encoder_hidden_stride: list[int],
        encoder_output_stride: list[int],
        encoder_hidden_channels: list[int],
        encoder_output_channels: list[int],
        decoder_num_rnn_layers: int,
        decoder_hidden_size: int,
        decoder_dropout_rate: float,
        decoder_rnn_type: str,
        n_tokens: int,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            encoder_hidden_conv_num,
            encoder_input_kernel,
            encoder_hidden_kernel,
            encoder_output_kernel,
            encoder_input_padding,
            encoder_hidden_padding,
            encoder_output_padding,
            encoder_input_stride,
            encoder_hidden_stride,
            encoder_output_stride,
            encoder_hidden_channels,
            encoder_output_channels,
        )
        decoder_input_size = self.encoder.get_output_lengths(encoder_n_feats)
        self.decoder = Decoder(
            decoder_num_rnn_layers,
            decoder_input_size,
            decoder_hidden_size,
            decoder_dropout_rate,
            decoder_rnn_type,
        )
        self.fc = nn.Linear(decoder_hidden_size, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        x_encoder_embed = self.encoder(spectrogram.unsqueeze(1))
        x_encoder_embed = x_encoder_embed.permute((2, 0, 1)).contiguous()
        x_decoder_embed = self.decoder(x_encoder_embed)
        log_probs = F.log_softmax(self.fc(x_decoder_embed).transpose(0, 1), dim=-1)
        return {
            "log_probs": log_probs,
            "log_probs_length": self.transform_input_lengths(spectrogram_length),
        }

    def transform_input_lengths(self, input_lengths):
        t_dim = input_lengths.max()
        cur_length = t_dim
        cur_length = self.encoder.get_size_after_conv(
            cur_length,
            self.encoder.input_kernel[::-1],
            self.encoder.input_padding[::-1],
            self.encoder.input_stride[::-1],
        )
        for _ in range(self.encoder.hidden_conv_num):
            cur_length = self.encoder.get_size_after_conv(
                cur_length,
                self.encoder.hidden_kernel[::-1],
                self.encoder.hidden_padding[::-1],
                self.encoder.hidden_stride[::-1],
            )
        cur_length = self.encoder.get_size_after_conv(
            cur_length,
            self.encoder.output_kernel[::-1],
            self.encoder.output_padding[::-1],
            self.encoder.output_stride[::-1],
        )
        return torch.zeros_like(input_lengths).fill_(cur_length)
