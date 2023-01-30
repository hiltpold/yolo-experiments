import os
import yaml
import argparse
import pandas as pd
import shutil
import matplotlib
import seaborn as sns
import json
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from matplotlib.patches import Rectangle
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

sns.set(font_scale=1.3)
sns.set(font_scale=1.3)
sns.set_style("darkgrid", {"axes.facecolor": ".95"})
# set fonttype
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def get_paths(path: Path) -> tuple[list[str], list[Path]]:
    image_paths = []
    annotation_paths = []
    for file_name in os.listdir(path):
        if file_name.endswith(".jpg"):
            image_paths.append(path / file_name)

        if file_name.endswith(".csv"):
            annotation_paths.append(path / file_name)

    return image_paths, annotation_paths


def build_class_index(path: Path) -> dict[str, int]:
    class_names = sorted(os.listdir(path / "crop"))
    filtered_class_names = list(
        filter(lambda f: not f.startswith("."), class_names))

    class_idx = {
        class_name: i
        for i, class_name in enumerate(filtered_class_names)
    }

    json_path = path / "class2idx.json"
    json_path.write_text(json.dumps(class_idx))
    return class_idx


def load_json_as_dict(json_path: Path) -> dict:
    if os.path.exists(json_path):
        with open(json_path) as f:
            d = json.load(f)
        return d
    else:
        raise RuntimeError(f"Path {json_path} does not exisit.")


def invert_class_index(path: Path,
                       index_file_name: str = "class2idx.json",
                       save=True) -> dict[int, str]:
    json_path = path / index_file_name
    if os.path.exists(json_path):
        with open(json_path) as f:
            class_idx = json.load(f)
        # invert class2idx -> idx2class
        inverted_class_idx = {
            class_id: class_name
            for class_name, class_id in class_idx.items()
        }
        if save:
            json_path = path / "idx2class.json"
            json_path.write_text(json.dumps(inverted_class_idx))
        return inverted_class_idx
    else:
        raise RuntimeError(
            f"No file {json_path} found. Start with preprocessing step.")


def convert_bboxes_to_yolo_format(paths: list[str], lu: dict[str, int],
                                  out_dir: str):
    for annotation_path in tqdm(paths):
        # get image_id
        image_id = annotation_path.parts[-1].split('.')[0]
        annotation_df = pd.read_csv(annotation_path)
        # transform to yolo format
        annotation_df = to_yolo_format(annotation_df, lu)
        # save to .txt resulting df
        with open(Path(out_dir) / f'{image_id}.txt', 'w') as f:
            f.write(annotation_df.to_string(header=False, index=False))


def to_yolo_format(df: pd.DataFrame, idx_lu: dict):
    df['class'] = df['class'].apply(lambda x: idx_lu[x]).values
    df['xmin'] = (df['xmin'] / df['width']).values
    df['ymin'] = (df['ymin'] / df['height']).values
    df['xmax'] = (df['xmax'] / df['width']).values
    df['ymax'] = (df['ymax'] / df['height']).values
    df['xc'] = (df['xmin'] + df['xmax']) / 2
    df['yc'] = (df['ymin'] + df['ymax']) / 2
    df['w'] = (df['xmax'] - df['xmin'])
    df['h'] = (df['ymax'] - df['ymin'])
    df.drop(['filename', 'width', 'height', 'xmin', 'xmax', 'ymin', 'ymax'],
            axis=1,
            inplace=True)
    return df


def gnerate_sample_images(n_samples: int,
                          path,
                          image_dir,
                          label_dir,
                          sample_image_dir,
                          class_idx: dict[str, int] = None,
                          inverted_class_idx: dict[int, str] = None):

    if class_idx is None:
        class_idx = load_json_as_dict(path / "class2idx.json")
    if inverted_class_idx is None:
        inverted_class_idx = load_json_as_dict(path / "idx2class.json")

    cmap = plt.get_cmap('rainbow', len(class_idx))
    sample_ids = np.random.randint(0, len(os.listdir(image_dir)), n_samples)

    image_paths = [
        Path(image_dir) / image_path
        for image_path in sorted(os.listdir(image_dir))
    ]
    label_paths = [
        Path(label_dir) / label_path
        for label_path in sorted(os.listdir(label_dir))
    ]

    image = np.array(Image.open(image_paths[sample_ids[0]]))
    bboxes = np.loadtxt(label_paths[sample_ids[0]], ndmin=2)
    for i, sample_id in enumerate(sample_ids):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 4 * 7))
        # load image and bboxes
        image = np.array(Image.open(image_paths[sample_id]))
        bboxes = np.loadtxt(label_paths[sample_id], ndmin=2)
        # get image shape
        image_h, image_w = image.shape[:2]
        for bbox in bboxes:
            class_id, xc, yc, w, h = bbox
            # rescale to image size
            xc, yc, w, h = image_w * xc, image_h * yc, image_w * w, image_h * h
            xmin, ymin = xc - w / 2, yc - h / 2
            rect = Rectangle((xmin, ymin),
                             w,
                             h,
                             linewidth=4,
                             edgecolor=cmap(int(class_id)),
                             facecolor='none',
                             alpha=0.5)
            ax.add_patch(rect)
            ax.text(xmin,
                    ymin,
                    inverted_class_idx[str(int(class_id))],
                    ha='left',
                    va='bottom',
                    bbox={
                        'facecolor': cmap(int(class_id)),
                        'alpha': 0.5
                    })

        ax.imshow(image)
        ax.axis("off")
        fig.savefig(sample_image_dir /
                    f"sample_{inverted_class_idx[str(int(class_id))]}.png",
                    bbox_inches='tight',
                    pad_inches=0)


def plot_train_val_split_frequency(path: Path, label_dir: Path,
                                   output_dir: Path):
    class_counter = {'train': Counter(), 'val': Counter()}
    class_freqs = {}

    with open(path / 'train_split.txt', 'r') as f:
        for line in f:
            image_id = line.split('/')[-1].split('.')[0]
            df = np.loadtxt(label_dir / f'{image_id}.txt', ndmin=2)
            class_counter['train'].update(df[:, 0].astype(int))
    # get class freqs
    total = sum(class_counter['train'].values())
    class_freqs['train'] = {
        k: v / total
        for k, v in class_counter['train'].items()
    }

    with open(path / 'val_split.txt', 'r') as f:
        for line in f:
            image_id = line.split('/')[-1].split('.')[0]
            df = np.loadtxt(label_dir / f'{image_id}.txt', ndmin=2)
            class_counter['val'].update(df[:, 0].astype(int))
    # get class freqs
    total = sum(class_counter['val'].values())
    class_freqs['val'] = {
        k: v / total
        for k, v in class_counter['val'].items()
    }

    # plot
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(range(40), [class_freqs['train'][i] for i in range(40)],
            color='navy',
            label='train')
    ax.plot(range(40), [class_freqs['val'][i] for i in range(40)],
            color='tomato',
            label='val')
    ax.legend()
    ax.set_xlabel('Class ID')
    ax.set_ylabel('Class Frequency')
    fig.savefig(output_dir / "train_val_split_frquency.png")


def unzip(file_name: str, target_dir: str):
    shutil.unpack_archive(file_name, target_dir)


def move_images(paths: list[Path], target_dir: Path, file_ending=".jpg"):
    for img in paths:
        img_id = img.parts[-1].split('.')[0]
        shutil.move(img, target_dir / Path(img_id + file_ending))


def dir_exists(dir: Path):
    if not dir.exists():
        print(f"Directory {dir} does not exist.")
        raise SystemExit(1)


def rm_dir(dir: Path):
    if dir.exists():
        shutil.rmtree(dir)
    else:
        print("Nothing removed.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("data_dir")
    parser.add_argument("model_dir")
    parser.add_argument("--archive_name")
    parser.add_argument('--preprocess', action="store_true")
    parser.add_argument('--samples', action="store_true")
    parser.add_argument('--train_conf', action="store_true")

    parser.set_defaults(preprocessing=False)
    parser.set_defaults(samples=False)
    parser.set_defaults(train_conf=False)

    args = parser.parse_args()

    zip_file = args.archive_name
    preprocess = args.preprocess
    samples = args.samples
    train_conf = args.train_conf

    if (preprocess and zip_file == ""):
        raise RuntimeError(
            f"If preprocessing is wished, please provide and archive_name")

    # directories with raw data
    base_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)

    # directories for preprocessed images and labels
    image_dir = Path(model_dir) / "dataset/images"
    label_dir = Path(model_dir) / "dataset/labels"
    sample_dir = Path(base_dir) / "sample_images"
    output_dir = Path(base_dir) / "plots"

    # check directories exist
    dir_exists(base_dir)
    dir_exists(model_dir)

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(model_dir, exist_ok=True)

    class2idx = {}
    idx2class = {}
    if preprocess:
        print("Preprocess images for yolov5 training")
        # unzip raw data
        raw_data_dir = base_dir / "raw"
        unzip(base_dir / zip_file, raw_data_dir)
        dir_exists(raw_data_dir)
        # get image and annotations from raw data directory
        image_paths, annotation_paths = get_paths(path=raw_data_dir /
                                                  "dataset")
        # build index
        class2idx = build_class_index(path=raw_data_dir)
        idx2class = invert_class_index(path=base_dir)
        # dataset labels are not yet in the correct format for yolov5
        convert_bboxes_to_yolo_format(paths=annotation_paths,
                                      lu=class2idx,
                                      out_dir=label_dir)
        # move data to final directories
        move_images(image_paths, image_dir)
        # remove raw files
        rm_dir(raw_data_dir)
        rm_dir(raw_data_dir / 'crop')
        rm_dir(raw_data_dir / 'annotated_samples')

    elif samples:
        gnerate_sample_images(5,
                              path=base_dir,
                              image_dir=image_dir,
                              label_dir=label_dir,
                              sample_image_dir=sample_dir)
    elif train_conf:
        print("Prepare train configuration")
        image_paths = [
            f'images/{image_path}'
            for image_path in sorted(os.listdir(image_dir))
        ]
        train_size = 0.8
        train_image_paths, val_image_paths = train_test_split(
            image_paths, train_size=train_size, random_state=42, shuffle=True)
        # make train split
        with open(model_dir / 'dataset/train_split.txt', 'w') as f:
            f.writelines(f'./{image_path}\n'
                         for image_path in train_image_paths)

        # make val split
        with open(model_dir / 'dataset/val_split.txt', 'w') as f:
            f.writelines(f'./{image_path}\n' for image_path in val_image_paths)

        # plot_train_val_split_frequency(path=base_dir / "dataset",
        #                               label_dir=label_dir,
        #                               output_dir=output_dir)

    else:
        print("Write configuration for training")
        idx = {
            int(k): v
            for k, v in load_json_as_dict(Path(base_dir) /
                                          "idx2class.json").items()
        }
        #idx = {}
        yaml_conf = {}
        yaml_conf["path"] = "../dataset/"
        yaml_conf["train"] = "train_split.txt"
        yaml_conf["val"] = "val_split.txt"
        yaml_conf["names"] = idx

        with open(Path("./yolov7") / "data/MilitaryAircraft.yaml", 'w') as f:
            yaml.dump(yaml_conf, f, default_flow_style=False)


if __name__ == "__main__":
    main()