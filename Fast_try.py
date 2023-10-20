import os
from PIL import Image

import torchvision.transforms
import torch.optim
import torchvision.transforms as T
import torch.backends.cudnn as cudnn


from model import Model
import matplotlib.pyplot as plt
import numpy as np


path = os.path.dirname(__file__)
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


def main():
    ##########setting seed
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.set_float32_matmul_precision('high')

    ##########setting models
    model = Model()
    model = model.cuda()
    checkpoint = torch.load(r"./model_last.pth")
    model.load_state_dict(checkpoint['state_dict'])

    ##Please load your 3D image
    #ct = np.load("")
    #ct = torch.from_numpy(ct).type(torch.FloatTensor)

    #No ct image
    ct = torch.from_numpy(np.zeros([250, 250, 200], dtype=float)).type(torch.FloatTensor)
    ct = ct[None,None, ...]
    ct = ct.cuda(non_blocking=True).requires_grad_()

    trans = torchvision.transforms.Compose([T.Pad((600,600)),T.CenterCrop((2048,2048)),T.PILToTensor(),T.ConvertImageDtype(torch.float32)])

    X_1 = trans(Image.open("BIMCV-COVID19-cIter_1_2/images/covid19_posi/sub-S03113/ses-E06208/mod-rx/sub-S03113_ses-E06208_run-1_bp-chest_cr.png")) #Please load your 2D Image

    #No X_1 image
    #X_1 = torch.unsqueeze(torch.from_numpy(np.zeros([2048, 2048], dtype=float)), 0).type(torch.FloatTensor)
    X_1 = X_1[None,...].type(torch.FloatTensor).cuda(non_blocking=True).requires_grad_()

    X_2 = trans(Image.open("BIMCV-COVID19-cIter_1_2/images/covid19_posi/sub-S03113/ses-E07104/mod-rx/sub-S03113_ses-E07104_run-1_bp-chest_cr.png")) #Please load your 3D Image

    # No X_2 image
    #X_2 = torch.unsqueeze(torch.from_numpy(np.zeros([2048, 2048], dtype=float)), 0).type(torch.FloatTensor)
    X_2 = X_2[None, ...].type(torch.FloatTensor).cuda(non_blocking=True).requires_grad_()

    model.zero_grad()
    output = model(ct, 1, X_1, X_2, False)

    output_idx = output.cpu().detach().argmax()
    print(output)

    output_max = output[0, output_idx]##Or manually put the label that should be obtained. The saliency map should not suffer big changes.

    output_max.backward()

    ##Uncomment the saliency map that wants to be visualized

    '''saliency, _ = torch.max(X_1.grad.data.abs()[0],dim=0)

    res = X_1.cpu().detach().numpy()[0].transpose(1, 2, 0)
    saliency = saliency.cpu().detach().numpy()
    print(np.max(saliency))'''

    '''saliency, _ = torch.max(X_2.grad.data.abs()[0],dim=0)

    res = X_2.cpu().detach().numpy()[0].transpose(1, 2, 0)
    saliency = saliency.cpu().detach().numpy()'''

    '''saliency, _ = torch.max(ct.grad.data.abs()[0], dim=0)

    res = ct.cpu().detach().numpy()[0][0][120]
    saliency = saliency.cpu().detach().numpy()[120]'''  #Change the number depending on the slide that wants to be observed
    # Some other 3D visualization methods can be used or the saliency map can even be saved and use ITK-SNAP to view it.

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(res,cmap='gray')
    ax[0].axis('off')
    ax[1].imshow(saliency, cmap='hot')
    ax[1].axis('off')
    plt.tight_layout()
    fig.suptitle('The Image and Its Saliency Map')
    plt.show()


if __name__ == '__main__':
    main()




