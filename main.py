from mar_mamba import MARMambaModel


def build_model(
    num_stocks,
    d_feat=158,
    market_dim=63,
    gpu=0,
    seed=0,
    save_prefix="mar_mamba",
):
    return MARMambaModel(
        d_feat=d_feat,
        num_stocks=num_stocks,
        market_dim=market_dim,
        d_model=128,
        wavelet_levels=3,
        relay_ratio=0.125,
        market_context_dim=32,
        dropout=0.3,
        num_layers=1,
        num_heads=4,
        d_state=16,
        ffn_ratio=2,
        market_feature_start=d_feat,
        market_feature_end=d_feat + market_dim,
        lambda_mse=0.1,
        n_epochs=40,
        lr=1e-3,
        weight_decay=1e-4,
        GPU=gpu,
        seed=seed,
        train_stop_loss_thred=None,
        save_path="model",
        save_prefix=save_prefix,
    )


if __name__ == "__main__":
    model = build_model(num_stocks=500)
    print(model.model)
