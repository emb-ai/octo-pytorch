from ml_collections import ConfigDict
from ml_collections.config_dict import placeholder, required_placeholder


def update_config(_prototype, **kwargs):
    result = dict(_prototype)
    for key, value in kwargs.items():
        if type(result.get(key)) == dict or type(result.get(key)) == ConfigDict:
            if not kwargs[key].get("_overwrite", False):
                value = dict(update_config(_prototype=result[key], **kwargs[key]))
            value.pop("_overwrite", None)
        result[key] = value
    result.pop("_overwrite", None)
    return ConfigDict(result)


def get_config(config_string):
    base_wandb_config = dict(
        project="orca",
        group=placeholder(str),
        entity=placeholder(str),
    )
    # "/mnt2/homer/datasets/mujoco_sim/franka_shoe_pick_and_place_2K_20230709-201001"
    base_sim_config = dict(
        batch_size=256,
        num_steps=int(2e6),
        log_interval=100,
        eval_interval=5000,
        save_interval=int(2e6),
        save_dir=placeholder(str),
        data_path=placeholder(str),
        resume_path=placeholder(str),
        seed=42,
        env_name="franka_shoe_pick_and_place",
        save_video=True,
        max_episode_steps=55,
        deterministic_eval=True,
        num_episodes_per_video=8,
        num_episodes_per_row=4,
        eval_episodes=20,
        num_val_batches=8,
        pretrained_weights=[],
    )

    base_real_config = dict(
        batch_size=4,
        num_steps=int(2e6),
        log_interval=100,
        eval_interval=5000,
        save_interval=5000,
        save_dir=placeholder(str),
        data_path=placeholder(str),
        resume_path=placeholder(str),
        seed=42,
        text_processor="muse_embedding",
        text_processor_kwargs=dict(),
        pretrained_weights=[],
        wandb=base_wandb_config,
    )

    # params that need to be specified multiple places
    normalization_type = "normal"

    base_data_config = dict(
        shuffle_buffer_size=25000,
        prefetch_num_batches=20,
        augment=True,
        augment_next_obs_goal_differently=False,
        augment_kwargs=dict(
            random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
            random_brightness=[0.2],
            random_contrast=[0.8, 1.2],
            random_saturation=[0.8, 1.2],
            random_hue=[0.1],
            augment_order=[
                "random_resized_crop",
                "random_brightness",
                "random_contrast",
                "random_saturation",
                "random_hue",
            ],
        ),
        goal_relabeling_strategy="uniform",
        goal_relabeling_kwargs=dict(reached_proportion=0.0),
        normalization_type=normalization_type,
    )
    base_optimizer_config = dict(
        learning_rate=3e-4,
        warmup_steps=2000,
        decay_steps=int(2e6),
    )

    base_model_config = dict(
        policy_kwargs=dict(
            num_layers=4,
            layer_size=1024,
            vocab_size=256,
            num_heads=8,
            feed_forward_size=512,
            dropout_rate=0.1,
            normalization_type=normalization_type,
        ),
    )

    possible_structures = {
        "sim_transformer_bc": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizers=["sim-obs-tokenizer"],
                    observation_tokenizer_kwargs={"sim-obs-tokenizer": {}},
                    task_tokenizers=["sim-goal-obs-tokenizer"],
                    task_tokenizer_kwargs={"sim-goal-obs-tokenizer": {}},
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=base_data_config,
                **base_sim_config,
            )
        ),
        "transformer_bc": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizers=["obs-tokenizer"],
                    observation_tokenizer_kwargs={"obs-tokenizer": {}},
                    task_tokenizers=["goal-obs-tokenizer"],
                    task_tokenizer_kwargs={"goal-obs-tokenizer": {}},
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=base_data_config,
                **base_real_config,
            )
        ),
        "transformer_bc_film_lang": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizers=["obs-film-language-tokenizer"],
                    observation_tokenizer_kwargs={
                        "obs-film-language-tokenizer": {"num_tokens": 64}
                    },
                    task_tokenizers=[],
                    task_tokenizer_kwargs={},
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=base_data_config,
                **base_real_config,
            )
        ),
        "transformer_bc_lang": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizers=["obs-tokenizer"],
                    observation_tokenizer_kwargs={"obs-tokenizer": {"num_tokens": 64}},
                    task_tokenizers=["language-tokenizer"],
                    task_tokenizer_kwargs={"language-tokenizer": {"num_tokens": 16}},
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=base_data_config,
                **base_real_config,
            )
        ),
        "transformer_bc_clip_text": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizers=["obs-tokenizer"],
                    observation_tokenizer_kwargs={"obs-tokenizer": {"num_tokens": 64}},
                    task_tokenizers=["clip-text-tokenizer"],
                    task_tokenizer_kwargs={"clip-text-tokenizer": {"num_tokens": 64}},
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=base_data_config,
                **update_config(
                    base_real_config,
                    text_processor="clip_processor",
                    pretrained_weights=["clip"],
                ),
            )
        ),
        "transformer_bc_clip_vit_and_text": ConfigDict(
            dict(
                agent="transformer_bc",
                obs_horizon=1,
                model=update_config(
                    base_model_config,
                    observation_tokenizers=["clip-obs-tokenizer"],
                    observation_tokenizer_kwargs={
                        "clip-obs-tokenizer": {"num_tokens": 50}
                    },
                    task_tokenizers=["clip-text-tokenizer"],
                    task_tokenizer_kwargs={"clip-text-tokenizer": {"num_tokens": 64}},
                ),
                optimizer=base_optimizer_config,
                dataset_kwargs=update_config(base_data_config, image_processor="clip"),
                **update_config(
                    base_real_config,
                    text_processor="clip_processor",
                    pretrained_weights=["clip"],
                ),
            )
        ),
    }

    return possible_structures[config_string]
