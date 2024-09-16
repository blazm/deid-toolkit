import argparse
import os
import lpips

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser(description="Evaluate FID score")
    parser.add_argument('path', type=str, nargs=2,
                    help=('Paths of the datasets aligned and deidentified'))
    args = parser.parse_args()
    aligned_path = args.path[0]
    deidentified_path = args.path[1]

    dataset_name = aligned_path.split("/")[-1]
    technique_name = deidentified_path.split("/")[-2]

    output_scores_file = f"./mse_{dataset_name}_{technique_name}.txt"
    print("Save file: ", output_scores_file)
    use_gpu =  True
 
    from torch import nn
    loss_fn = nn.MSELoss()
    if(use_gpu):
        loss_fn.cuda()
    f = open(output_scores_file,'w')
    files = os.listdir(aligned_path)
    #print(files[:10])
    #import random
    #random.shuffle(files)
    #print("After: ", files[:10])

    for file in files:
        if(os.path.exists(os.path.join(deidentified_path,file))):
            # Load images
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(aligned_path,file))) # RGB image from [-1,1]
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(deidentified_path,file)))            
            #if img1.shape[2] == 128: # CIAGAN images need to be resized
            #	#from torchvision import transforms
            #	#trans = transforms.Compose([transforms.Resize(1024, 1024)])
            #	img1 = lpips.upsample(img1, (1024, 1024))
            #print(img0.min(), img0.max(), img1.min(), img1.max())
            if(use_gpu):
                img0 = img0.cuda()
                img1 = img1.cuda()
            #print("Shapes: ", img0.shape, " ", img1.shape)
            # Compute distance
            dist01 = loss_fn(img0, img1) # mse 
            #dist01 = loss_fn.forward(img0,img1)
            #print('%s: %.3f'%(file,dist01)) # if using spatial, we need .mean()
            #f.writelines('%s: %.6f\n'%(file,dist01)) # original saves image name and score
            f.writelines('%.6f\n'%(dist01)) # we need only scores, to compute averages easily
            f.close()
    else:
        print("TODO call the file")
        print("TODO get the mean")
        print("TODO print the result")

if __name__ == '__main__':
    main()