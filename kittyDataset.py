from PIL import Image
from config import IMAGES, ANNOTATIONS, LABEL_TO_ID

class KittiDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, IMAGES))))
        self.texts = list(sorted(os.listdir(os.path.join(root, ANNOTATIONS))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, IMAGES, self.imgs[idx])
        label_path = os.path.join(self.root, ANNOTATIONS, self.texts[idx])
        img = Image.open(img_path).convert("RGB")
        
        f = open( label_path, "r")
        lines = list(f)

        # read the annotations
        boxes = []
        labels = []
        for line in lines:
            objects = list(np.array(line.split())[index_selects])
            if LABEL_TO_ID[objects[0]] > 0:
                b = list(map(float, objects[1:]))
                boxes.append(b)
                labels.append(LABEL_TO_ID[objects[0]])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)