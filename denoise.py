import torch

import numpy as np

from scipy.io import wavfile
from tqdm import trange

from model.demucsv2.model import Demucs, center_trim
from utils.audio import get_clean_n_noisy
from utils.operation import (
    get_args,
    prepare_repetition_directory,
    update_state,
    get_logger,
    save_repetition_results,
    save_experiment_results,
    prepare_experiment_directory,
    get_paths,
    get_crit,
)
from utils.metrics import MetricsTracker
from model.models import get_ncp
from model.models import get_sashimi
from utils.plot import plot_something


MAX_EXPERIMENT_REPETITION = 1000


def main(args, logger):
    logger.info(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.info("CUDA not available switching to CPU")
        args.device = "cpu"
    paths = get_paths(args.source)
    experiment_dir = prepare_experiment_directory(args)
    for repetition_i in range(MAX_EXPERIMENT_REPETITION):
        wavs = get_clean_n_noisy(paths[repetition_i % len(paths)], args, logger)
        if wavs is None:
            continue
        clean, noisy, clean_path = wavs
        args.clean_path = clean_path
        repetition_directory = prepare_repetition_directory(experiment_dir, args)
        if repetition_directory is None:  # already finished all required repetitions
            logger.info("Finished required repetitions")
            break
        net_input = (torch.rand(size=clean.size(), device=clean.device) - 0.5) * 2
        # write network input
        wavfile.write(
            str(repetition_directory / "input_noise.wav"),
            rate=args.samplerate,
            data=net_input.squeeze().detach().cpu().numpy(),
        )
        if args.architecture == "NCP":
            net = get_ncp(args)
        elif args.architecture == "SaShiMi":
            net = get_sashimi(args)
        else:
            net = Demucs(
                sources=1,
                audio_channels=1,
                samplerate=args.samplerate,
                depth=args.depth,
                skip=args.skip,
                lstm_layers=args.lstm_layers,
                glu=args.glu,
                attention_layers=args.attention_layers,
                attention_heads=args.attention_heads,
                resample=args.resample,
            )
        net = net.to(args.device)
        crit = get_crit(args)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
        wavfile.write(
            str(repetition_directory / "noisy.wav"),
            rate=args.samplerate,
            data=noisy.squeeze().detach().cpu().numpy(),
        )
        wavfile.write(
            str(repetition_directory / "clean.wav"),
            rate=args.samplerate,
            data=clean.squeeze().detach().cpu().numpy(),
        )
        losses = []
        aaouts = []
        aains = []
        with torch.no_grad():
            print(f"Input dimensions: {net_input.shape}")
            out = net(net_input).squeeze(0)
            print(f"Output dimensions: {out.shape}")
            clean = center_trim(clean, out)
            noisy = center_trim(noisy, out)
        wavfile.write(
            str(repetition_directory / "init_pass.wav"),
            rate=args.samplerate,
            data=out.squeeze().detach().cpu().numpy(),
        )

        epochs_range = trange(1, args.epochs + 1)
        metrics_tracker = MetricsTracker(clean, noisy)

        for epoch in epochs_range:
            optimizer.zero_grad()
            out = net(net_input).squeeze(0)
            total_loss = crit(out.squeeze(), noisy.squeeze())
            total_loss.backward()
            loss = total_loss.item()
            losses.append(loss)
            epochs_range.set_description("Epoch %05d Loss=%f" % (epoch, loss))
            optimizer.step()
            update_state(
                clean,
                noisy,
                repetition_directory,
                epoch,
                losses,
                metrics_tracker,
                out,
                args,
            )
            aaouts.append(torch.mean(out.absolute()).item())
            aains.append(torch.mean(net_input.absolute()).item())
            if epoch % args.show_every == 0:
                plot_something(repetition_directory, aaouts, "Output average absolute")
                plot_something(
                    repetition_directory, aains, "Net input average absolute"
                )
        save_repetition_results(
            metrics_tracker, np.array(losses), args.samplerate, repetition_directory
        )
        try:
            torch.save(net.state_dict(), str(repetition_directory / "state.th"))
        except Exception as e:
            logger.exception(e)
    save_experiment_results(experiment_dir, logger)


if __name__ == "__main__":
    logger = get_logger()
    args = get_args()
    main(args, logger)
