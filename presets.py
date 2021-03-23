import transforms as T


class SegmentationPresetTrain:
    def __init__(self, scale_min=0.5, scale_max=1.75,
                 rotate_min=-1, rotate_max=1,
                 train_h=512, train_w=1024,
                 ignore_label=255,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        train_transform = T.Compose([
            T.RandScale([scale_min, scale_max]),
            T.RandRotate([rotate_min, rotate_max], padding=mean, ignore_label=ignore_label),
            T.RandomGaussianBlur(),
            T.RandomHorizontalFlip(),
            T.Crop([train_h, train_w], crop_type='rand', padding=mean, ignore_label=ignore_label),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
        ])

        # min_size = int(0.5 * base_size)
        # max_size = int(2.0 * base_size)

        # trans = [T.RandomResize(min_size, max_size)]
        # if hflip_prob > 0:
        #     trans.append(T.RandomHorizontalFlip(hflip_prob))
        # trans.extend([
        #     T.RandomCrop(crop_size),
        #     T.ToTensor(),
        #     T.Normalize(mean=mean, std=std),
        # ])
        # self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetEval:
    def __init__(self, base_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.RandomResize(base_size, base_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)
