import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


with app.setup:
    import sys

    sys.path.append("../src")
    import hydra
    from hydra import compose, initialize
    from radar.modeling_diff.diff_pipline import RDDPMPipeline
    from radar.misc.utils import instantiate
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    DEFAULT_CONFIG_PATH = "../../root/projects/Radar_Yao/configs"


@app.cell
def _():
    model = RDDPMPipeline.from_pretrained(
        "../outputs/first_demo/best_checkpoint/best_eval_mse=0.2885"
    )
    with initialize(config_path=DEFAULT_CONFIG_PATH, version_base=None):
        cfg = compose(config_name="default_train")
        # creat datset
        train_transform = instantiate(cfg.data.transform)
        train_set = instantiate(
            cfg.data.dataset,
            transform=train_transform,
        )
    collate_fn = instantiate(cfg.data.collator)
    loader = DataLoader(
        train_set, batch_size=2, shuffle=True, num_workers=2, collate_fn=collate_fn
    )

    return


@app.cell
def _(model, train_set, loader):
    idx = np.random.randint(0, len(train_set))

    sample = next(iter(loader))

    label = sample["label"].to("cuda:0")
    r_conditional_input = sample["r_conditional_input"].to("cuda:0")
    print(r_conditional_input.shape)

    model.to("cuda:0")
    with torch.no_grad():
        pred = model(
            batch_size=r_conditional_input.shape[0],
            r_conditional_input=r_conditional_input,
            output_type="numpy",
            return_dict=False,
            num_inference_steps=500,
        )[0]

    return


@app.cell
def _(label, r_conditional_input, pred):
    print(label.shape)
    print(pred.shape)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(label.squeeze().cpu().numpy()[0], cmap="gray")
    axs[0].set_title("Label")
    axs[1].imshow(r_conditional_input.squeeze().cpu().numpy()[0], cmap="gray")
    axs[1].set_title("Conditional Input")
    axs[2].imshow(pred.reshape(-1, 48, 64)[0], cmap="gray")
    axs[2].set_title("Predicted")
    axs[2].set_title("Predicted")


if __name__ == "__main__":
    app.run()
