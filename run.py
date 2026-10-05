"""Extract HE/VPR features and evaluate full-database VPR retrieval."""

import argparse
import glob
import json
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from models.aggregators.gem_dino import GeMPoolDino
from models.aggregators.salad_crica import SALAD_CRICA
from models.backbones.mona_vit import get_mona_backbone

HERE = Path(__file__).resolve().parent
DIM = {"vpr": 8448, "he": 128}
KS = (1, 5, 10, 20)


class Model(nn.Module):
    def __init__(self, task):
        super().__init__()
        self.backbone = get_mona_backbone(False, dino_name="dinov2_vitb14")
        self.aggregator = (SALAD_CRICA(768, 64, 128, 256) if task == "vpr"
                           else GeMPoolDino(p=3))

    def forward(self, images):
        return self.aggregator(self.backbone(images))


def load_model(task, device):
    model = Model(task)
    checkpoint = torch.load(HERE / "weights" / (task + ".ckpt"),
                            map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval().to(device)


def discover(dataset, root, task):
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if task == "he":
        db = glob.glob(str(root / "height_database_large" / "**" / "*.tif"), recursive=True)
        qr = glob.glob(str(root / "query_images" / "**" / "*.png"), recursive=True)
    elif dataset == "gstudio":
        db = glob.glob(str(root / "map_database" / "**" / "*@*.tif"), recursive=True)
        qr = [p for p in glob.glob(str(root / "query_images" / "**" / "*.png"), recursive=True)
              if "@" in Path(p).name]
    else:
        db = glob.glob(str(root / "map_database_2" / "*@*.tif"))
        qr = []
        for folder in ("Traj1", "Traj2"):
            qr += glob.glob(str(root / "query_images" / folder / "**" / "*.png"), recursive=True)
    if not db or not qr:
        raise ValueError(f"No database/query images found under {root}")
    return root, list(map(Path, db)), list(map(Path, qr))


class Images(Dataset):
    def __init__(self, paths, is_db, dataset):
        self.paths, self.is_db, self.dataset = paths, is_db, dataset
        self.transform = T.Compose((
            T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as source:
            image = source.convert("RGB")
            if self.is_db and self.dataset == "jimo":
                image = image.rotate(-90)
            elif not self.is_db:
                width, height = image.size
                image = image.crop(((width-height)/2, 0, (width+height)/2, height))
            return self.transform(image)


def paths(directory, dataset, task):
    stem = dataset + "_" + task
    return (directory / (stem + "_db.npy"),
            directory / (stem + "_query.npy"),
            directory / (stem + "_manifest.json"))


def manifest(root, db, qr):
    return {key: [str(p.relative_to(root)).replace("\\", "/") for p in items]
            for key, items in (("db", db), ("query", qr))}


def extract(args):
    root, db, qr = discover(args.dataset, args.data_root, args.model)
    directory = Path(args.features_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    db_file, qr_file, manifest_file = paths(directory, args.dataset, args.model)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    model = load_model(args.model, device)
    for items, target, is_db in ((db, db_file, True), (qr, qr_file, False)):
        loader = DataLoader(Images(items, is_db, args.dataset),
                            batch_size=args.batch_size, num_workers=args.workers)
        vectors = np.lib.format.open_memmap(target, mode="w+", dtype="float32",
                                            shape=(len(items), DIM[args.model]))
        offset = 0
        with torch.inference_mode():
            for images in loader:
                batch = model(images.to(device)).float().cpu().numpy()
                vectors[offset:offset+len(batch)] = batch
                offset += len(batch)
        vectors.flush()
        print(f"{len(items)} images -> {target}")
    manifest_file.write_text(json.dumps(manifest(root, db, qr), indent=2), encoding="utf-8")


def coordinates(path, dataset, query):
    fields = path.stem.split("@")
    try:
        return ((float(fields[3]), float(fields[2])) if dataset == "gstudio" and query
                else (float(fields[2]), float(fields[3])))
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Bad coordinate fields in {path.name}") from exc


def evaluate(args):
    root, db, qr = discover(args.dataset, args.data_root, "vpr")
    db_file, qr_file, manifest_file = paths(Path(args.features_dir).resolve(), args.dataset, "vpr")
    if json.loads(manifest_file.read_text(encoding="utf-8")) != manifest(root, db, qr):
        raise ValueError("Feature manifest differs from dataset order; extract again.")
    db_vec, qr_vec = np.load(db_file, mmap_mode="r"), np.load(qr_file, mmap_mode="r")
    if db_vec.shape != (len(db), DIM["vpr"]) or qr_vec.shape != (len(qr), DIM["vpr"]):
        raise ValueError("Feature arrays have unexpected shape.")
    db_coords = np.asarray([coordinates(p, args.dataset, False) for p in db])
    qr_coords = np.asarray([coordinates(p, args.dataset, True) for p in qr])
    radius = 0.01 if args.dataset == "gstudio" else 0.003
    positives = NearestNeighbors(n_jobs=1).fit(db_coords).radius_neighbors(qr_coords, radius)[1]
    faiss.omp_set_num_threads(args.threads)
    index = faiss.IndexFlatL2(DIM["vpr"])
    index.add(np.ascontiguousarray(db_vec))
    predicted = index.search(np.ascontiguousarray(qr_vec), max(KS))[1]
    print(f"{args.dataset}: {len(db)} database, {len(qr)} queries")
    print(" ".join(f"R@{k}={100*np.mean([np.isin(row[:k], positives[i]).any() for i, row in enumerate(predicted)]):.2f}%"
                   for k in KS))
    print(f"No-positive queries: {sum(len(p) == 0 for p in positives)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("extract", "evaluate"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--dataset", choices=("gstudio", "jimo"), required=True)
        cmd.add_argument("--data-root", type=Path, required=True)
        cmd.add_argument("--features-dir", type=Path, required=True)
        if name == "extract":
            cmd.add_argument("--model", choices=("vpr", "he"), default="vpr")
            cmd.add_argument("--batch-size", type=int, default=8)
            cmd.add_argument("--workers", type=int, default=0)
            cmd.add_argument("--device", default="auto")
        else:
            cmd.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    (extract if args.command == "extract" else evaluate)(args)


if __name__ == "__main__":
    main()
