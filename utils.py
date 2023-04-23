import argparse
import yaml
import glob
from PIL import Image

def make_gif(frame_folder):
    """
    https://www.blog.pythonlibrary.org/2021/06/23/creating-an-animated-gif-with-python/"""
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.JPG")]
    frame_one = frames[0]
    frame_one.save("my_awesome.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--config", type=str, default="configs/config_dqn.yaml", help="hyperparameter file path, default: config.yaml")
    parser.add_argument("--n_episodes", type=int, required = False, help="override config file for wandb sweep")
    parser.add_argument("--max_episode_steps", type=int, required = False, help="override config file for wandb sweep")
    parser.add_argument("--seed", type=int, required = False, help="override config file for wandb sweep")
    parser.add_argument("--lr_policy", type=float, required = False, help="override config file for wandb sweep")
    # parser.add_argument("--lr_critic", type=float, required = False, help="override config file for wandb sweep")
    # parser.add_argument("--lr_alpha", type=float, required = False, help="override config file for wandb sweep")
    parser.add_argument("--buffer_size", type=int, required = False, help="override config file for wandb sweep")
    parser.add_argument("--start_size", type=int, required = False, help="override config file for wandb sweep")
    parser.add_argument("--batch_size", type=int, required = False, help="override config file for wandb sweep")
    parser.add_argument("--hidden_units", type=int, required = False, help="override config file for wandb sweep")
    parser.add_argument("--tau", type=float, required = False, help="override config file for wandb sweep")
        
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader = yaml.FullLoader)
    config["config_path"] = args.config
    if args.n_episodes is not None:
        config["n_episodes"] = args.n_episodes
        config["seed"] = args.seed
        config["max_episode_steps"] = args.max_episode_steps
        config["learning_rate"]["policy"] = args.lr_policy
        # config["learning_rate"]["alpha"] = args.lr_alpha
        # config["learning_rate"]["critic"] = args.lr_critic
        config["buffer_size"] = args.buffer_size
        config["start_size"] = args.start_size
        config["hidden_units"] = args.hidden_units
        config["batch_size"] = args.batch_size
        config["target_smoothing_coefficient"] = args.tau
    return config
