
def load_checkpoint(filepath):
    
    checkpoint=torch.load(filepath)
   
    model=models.vgg16(pretrained=True)
    
    model.classifier=checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx=checkpoint['mapping_class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer_state'])
            
    epochs=checkpoint['epochs']
    
    return model

model=load_checkpoint('checkpoint_path.pth')

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    img_tensor=test_transforms(image_pil)
    
    return img_tensor
    


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax



def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image_pil=Image.open(image_path)
    
    image=process_image(image_pil)
    
    image=image.unsqueeze(0)
    
    logps=model.forward(image.to(device))
    ps=torch.exp(logps)
    top_p, top_class=ps.topk(5, dim=1)
    
    probs=top_p[0].tolist()
    inv_map= {v:k for k, v in model.class_to_idx.items()}
    classes=[inv_map[key] for key in top_class[0].tolist()]
    


#Display an image along with the top 5 classes

image_path=args.path/to/image

probs, classes=predict(image_path, model, topk=5)
flower_classes=[cat_to_name[str(x)] for x in classes]

image_pil=Image.open(image_path)
img_tensor=process_image(image_pil)


plt.subplot(2,1,1)
imshow(img_tensor)
plt.title(cat_to_name[str(10)])
plt.show()

plt.subplot(2,1,2)
plt.barh(flower_classes, probs)

plt.yticks(flower_classes)
plt.ylabel('flower classes')
plt.title('flower probabilities')

plt.xlabel('probabilities')

plt.show()



