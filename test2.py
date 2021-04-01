import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--save_path",
    type = str,
    default = 'hello world!',
    help = 'path to save model'
)

args = parser.parse_args()

if __name__ == '__main__':
    print(args.save_path)
