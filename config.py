import torchvision
import torchvision.transforms as transforms

model = dict(in_channels_first=1,
             out_channels_first=20,
             kernel_size_first=5,
             stride_first=2,
             in_channels_second=20,
             out_channels_second=20,
             kernel_size_second=5,
             stride_second=2,
             in_features_first=4*4*20,
             out_features_first=125,
             in_features_second=125,
             out_features_second=10)

preprocessing = dict(transforms=[transforms.RandomRotation(30),
                                 transforms.RandomVerticalFlip(p=1),
                                 transforms.RandomHorizontalFlip(p=1),
                                 transforms.GaussianBlur(kernel_size=5, sigma=0.2)],
                      transformed_size=0.25)

training_and_validation = dict(training_size=0.75,
                               batch_size=15000,
                               num_workers=2,
                               epochs=50)

model_selection = dict(dropout_p=[0.25, 0.5],
                       learning_rate=[1e-2, 1e-3, 1e-4],
                       weight_decay=[1e-4, 1e-5],
                       epochs=10)

