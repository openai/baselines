import argparse
import progressbar

from baselines.common.azure_utils import Container


def parse_args():
    parser = argparse.ArgumentParser("Download a pretrained model from Azure.")
    # Environment
    parser.add_argument("--model-dir", type=str, default=None,
                        help="save model in this directory this directory. ")
    parser.add_argument("--account-name", type=str, default="openaisciszymon",
                        help="account name for Azure Blob Storage")
    parser.add_argument("--account-key", type=str, default=None,
                        help="account key for Azure Blob Storage")
    parser.add_argument("--container", type=str, default="dqn-blogpost",
                        help="container name and blob name separated by colon serparated by colon")
    parser.add_argument("--blob", type=str, default=None, help="blob with the model")
    return parser.parse_args()


def main():
    args = parse_args()
    c = Container(account_name=args.account_name,
                  account_key=args.account_key,
                  container_name=args.container)

    if args.blob is None:
        print("Listing available models:")
        print()
        for blob in sorted(c.list(prefix="model-")):
            print(blob)
    else:
        print("Downloading {} to {}...".format(args.blob, args.model_dir))
        bar = None

        def callback(current, total):
            nonlocal bar
            if bar is None:
                bar = progressbar.ProgressBar(max_value=total)
            bar.update(current)

        assert c.exists(args.blob), "model {} does not exist".format(args.blob)

        assert args.model_dir is not None

        c.get(args.model_dir, args.blob, callback=callback)


if __name__ == '__main__':
    main()
