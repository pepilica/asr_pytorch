from pathlib import Path

import pandas as pd
import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_predictions(
        self,
        text,
        log_probs,
        log_probs_length,
        audio_path,
        audio,
        examples_to_log=10,
        use_torchaudio_ctc=False,
        **batch
    ):
        # TODO add beam search
        # Note: by improving text encoder and metrics design
        # this logging can also be improved significantly

        argmax_inds = log_probs.cpu().argmax(-1).numpy()
        argmax_inds = [
            inds[: int(ind_len)]
            for inds, ind_len in zip(argmax_inds, log_probs_length.numpy())
        ]
        argmax_texts_raw = [self.text_encoder.decode(inds) for inds in argmax_inds]
        argmax_texts = [self.text_encoder.ctc_decode(inds) for inds in argmax_inds]

        beam_inds = self.text_encoder.beam_search_ctc_decode(
            log_probs.cpu(), log_probs_length.cpu()
        )
        beam_texts = [self.text_encoder.decode(inds) for inds in beam_inds]

        tuples = list(
            zip(argmax_texts, beam_texts, text, argmax_texts_raw, audio_path, audio)
        )

        rows = {}
        for argmax_pred, beam_pred, target, raw_pred, audio_path, audio in tuples[
            :examples_to_log
        ]:
            target = self.text_encoder.normalize_text(target)
            argmax_wer = calc_wer(target, argmax_pred) * 100
            argmax_cer = calc_cer(target, argmax_pred) * 100
            beam_wer = calc_wer(target, beam_pred) * 100
            beam_cer = calc_cer(target, beam_pred) * 100

            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            rows[Path(audio_path).name] = {
                "audio": self.writer.wandb.Audio(
                    audio.reshape((-1,)), sample_rate=16000
                ),
                "target": target,
                "raw prediction": raw_pred,
                "argmax predictions": argmax_pred,
                "argmax wer": argmax_wer,
                "argmax cer": argmax_cer,
                "beam predictions": beam_pred,
                "beam wer": beam_wer,
                "beam cer": beam_cer,
            }
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )
