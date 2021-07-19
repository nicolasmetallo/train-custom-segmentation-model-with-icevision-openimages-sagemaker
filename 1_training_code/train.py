import sys
import os
import argparse
from PIL import Image
from icevision.all import *

def _train(
    input_dir:str=None, class_label:str=None, coco_fname:str='coco_annotation.json', 
    epochs:int=20, lr:float=5e-4, img_presize:int=512, img_size:int=384, bs:int=16):
    """ Main function that launches model training and saves output in S3 """
    # create list of images/segmentations
    data_dir = Path(f'{input_dir}/{class_label.lower()}')
    images_dir = data_dir / 'images'
    
    # Parse records with random splits
    parser = parsers.COCOMaskParser(annotations_filepath=f'{input_dir}/{coco_fname}', img_dir=images_dir)
    train_records, valid_records = parser.parse()
    class_map = ClassMap([class_label.lower()])

    # Define the transforms and create the Datasets
    presize = img_presize
    size = img_size
    shift_scale_rotate = tfms.A.ShiftScaleRotate(rotate_limit=10)
    crop_fn = partial(tfms.A.RandomSizedCrop, min_max_height=(size // 2, size), p=0.5)
    train_tfms = tfms.A.Adapter(
        [
            *tfms.A.aug_tfms(
                size=size,
                presize=presize,
                shift_scale_rotate=shift_scale_rotate,
                crop_fn=crop_fn,
            ),
            tfms.A.Normalize(),
        ]
    )
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(size=size), tfms.A.Normalize()])
    train_ds = Dataset(train_records, train_tfms)
    valid_ds = Dataset(valid_records, valid_tfms)

    # Create DataLoaders
    train_dl = mask_rcnn.train_dl(train_ds, batch_size=bs, shuffle=True, num_workers=4)
    valid_dl = mask_rcnn.valid_dl(valid_ds, batch_size=bs, shuffle=False, num_workers=4)

    # Define metrics for the model
    # TODO: Currently broken for Mask RCNN
    # metrics = [COCOMetric(COCOMetricType.mask)]

    # Create model
    model = mask_rcnn.model(num_classes=len(class_map))

    # Create Fastai Learner and train the model
    learn = mask_rcnn.fastai.learner(dls=[train_dl, valid_dl], model=model)
    if args.num_gpus > 1: learn.to_parallel()
    learn.fine_tune(epochs, lr, freeze_epochs=2)
    
    # Save model
    _save_model(model, class_label, args.model_dir)

def _save_model(model, class_label, model_dir):
    print("saving model weights and classes...")
    
    # Save weights for eager mode
    # Recommended way from http://pytorch.org/docs/master/notes/serialization.html
    with open(os.path.join(model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)
        
    # Save list of classes, ordered by index!
    if not isinstance(class_label,list): class_label = [class_label]
    with open(os.path.join(model_dir, 'classes.txt'), 'w') as f:
        for c in class_label:
            f.write(f"{c}\n")
            
    # Save scripted model
    # https://discuss.pytorch.org/t/torch-jit-trace-is-not-working-with-mask-rcnn/83244/8
    with torch.no_grad():
        scripted_model = torch.jit.script(model.eval())
    scripted_model.save(os.path.join(model_dir,'model.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # hyperparameters
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--img-presize', type=int, default=512)
    parser.add_argument('--img-size', type=int, default=384)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--class-label', type=str, required=True)
    
    # sagemaker training parameters
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--num-gpus', type=int, default=int(os.environ['SM_NUM_GPUS']))

    args = parser.parse_args()
    
    # train
    _train(input_dir=args.train, 
           class_label=args.class_label, 
           coco_fname='coco_annotation.json',
           epochs=args.epochs,
           lr=args.lr,
           img_presize=args.img_presize,
           img_size=args.img_size,
           bs=args.bs)