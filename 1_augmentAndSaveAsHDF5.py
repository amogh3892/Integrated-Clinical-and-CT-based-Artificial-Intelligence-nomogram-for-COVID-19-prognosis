import SimpleITK as sitk 
from pathlib import Path
import sys 
sys.path.append(fr"../Code_general")
from dataUtil import DataUtil 
from medImageProcessingUtil import MedImageProcessingUtil
import tables
import os 
import numpy as np
from glob import glob 
import matplotlib.pyplot as plt 
from augmentation3DUtil import Augmentation3DUtil
from augmentation3DUtil import Transforms
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.morphology import opening, closing
from skimage.morphology import disk
from skimage.measure import regionprops
from skimage.measure import label as ConnectedComponent
from skimage.transform import resize as skresize
import pandas as pd 



def _getAugmentedData(imgs,masks,nosamples):
    
    """ 
    This function defines different augmentations/transofrmation sepcified for a single image 
    imgs,masks : to be provided as list of SimpleITK images 
    nosamples : (int) number of augmented samples to be returned
    
    """
    au = Augmentation3DUtil(imgs,masks=masks)

    au.add(Transforms.SHEAR,probability = 0.3, magnitude = (0.05,0.05))
    au.add(Transforms.SHEAR,probability = 0.3, magnitude = (0.02,0.05))
    au.add(Transforms.SHEAR,probability = 0.3, magnitude = (0.01,0.05))
    au.add(Transforms.SHEAR,probability = 0.3, magnitude = (0.03,0.05))

    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = 1)
    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = -1)
    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = 2)
    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = -2)
    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = 4)
    au.add(Transforms.ROTATE2D,probability = 0.4, degrees = -4)

    au.add(Transforms.FLIPHORIZONTAL,probability = 0.5)

    imgs, augs = au.process(nosamples)

    return imgs,augs


def createHDF5(splitspathname,patchSize,depth):
    
    """
    splitspathname : name of the file (json) which has train test splits info 
    patchSize : x,y dimension of the image 
    depth : z dimension of the image 
    """
    
    outputfolder = fr"outputs/hdf5/{splitspathname}"
    Path(outputfolder).mkdir(parents=True, exist_ok=True)

    img_dtype = tables.Float32Atom()
    pm_dtype = tables.UInt8Atom()

    if depth > 1:
        data_shape = (0, depth, patchSize[0], patchSize[1])
        data_chuck_shape = (1,depth,patchSize[0],patchSize[1])

        mask_shape = (0,depth,patchSize[0],patchSize[1])
        mask_chuck_shape = (1,depth,patchSize[0],patchSize[1])

    else:
        data_shape = (0, patchSize[0], patchSize[1])
        data_chuck_shape = (1,patchSize[0],patchSize[1])

        mask_shape = (0,patchSize[0],patchSize[1])
        mask_chuck_shape = (1,patchSize[0],patchSize[1])

    filters = tables.Filters(complevel=5)

    splitspath = fr"outputs/splits/{splitspathname}.json"
    splitsdict = DataUtil.readJson(splitspath)

    phases = np.unique(list(splitsdict.values()))

    for phase in phases:
        hdf5_path = fr'{outputfolder}/{phase}.h5'

        if os.path.exists(hdf5_path):
            Path(hdf5_path).unlink()

        hdf5_file = tables.open_file(hdf5_path, mode='w')


        data = hdf5_file.create_earray(hdf5_file.root, "data", img_dtype,
                                            shape=data_shape,
                                            chunkshape = data_chuck_shape,
                                            filters = filters)

        mask =  hdf5_file.create_earray(hdf5_file.root, "mask", pm_dtype,
                                            shape=mask_shape,
                                            chunkshape = mask_chuck_shape,
                                            filters = filters)

        hdf5_file.close()

def getAugmentedData(folderpath,name, nosamples = None):
    
    """
    folderpath : path to folder containing images, mask
    name: Case ID or patient ID- reference name to the case 
    no_sample: No. of augmentations to be performed. 
    """

    folderpath = Path(folderpath)

    ext = folderpath.glob(fr"{name}*").__next__().stem.split(".")[-1]

    if ext == "gz":
        ext = ".".join(glob(fr"{folderpath}/**")[0].split("//")[-1].split(".")[-2:])

    img = sitk.ReadImage(str(folderpath.joinpath(fr"{name}.{ext}")))

    ls = sitk.ReadImage(str(folderpath.joinpath(fr"{name}-ns-label.{ext}")))
    ls = DataUtil.convert2binary(ls)

    lm = sitk.ReadImage(str(folderpath.joinpath(fr"{name}-lungmask-label.{ext}")))
    lm = DataUtil.convert2binary(lm)

    ret = []
    
    orgimg,augs = _getAugmentedData([img],[ls,lm],nosamples)
    ret.append((orgimg))

    if augs is not None:
        for i in range(len(augs)):
            ret.append(augs[i])

    return ret

def normalizeImage(img,_min,_max,clipValue):

    imgarr = sitk.GetArrayFromImage(img)

    if clipValue is not None:
        imgarr[imgarr > clipValue] = clipValue 

    imgarr[imgarr < _min] = _min
    imgarr[imgarr > _max] = _max

    imgarr = (imgarr - _min)/(_max - _min)

    imgarr = imgarr.astype(np.float32)

    return imgarr



def _addToHDF5(imgarr,maskarr,phase,splitspathname):
    
    """
    imgarr : input image sample (ex 2 slices of the image)
    maskarr : output mask (ex. lesion segmentation mask)
    phase : phase of that image (train,test,val)
    splitspathname : name of the file (json) which has train test splits info 
    """

    outputfolder = fr"outputs/hdf5/{splitspathname}"

    hdf5_file = tables.open_file(fr'{outputfolder}/{phase}.h5', mode='a')

    data = hdf5_file.root["data"]
    mask = hdf5_file.root["mask"]

    data.append(imgarr[None])
    mask.append(maskarr[None])
    
    hdf5_file.close()


def addToHDF5(img,ls,lm,phase,splitspathname,label,name,_min,_max,patchSize,depth,clipValue=None):
    
    """ 
    Collect samples from the cropped volume and add them into HDF5 file 
    img: CT volume 
    ls: lung lesions volume  
    lm: lm: lung mask volume 
    phase: train, val or test 
    splitspathname: name of the file (json) which has train test splits info 
    label: mild (0) or severe (1) COVID-19 
    name: patient ID or case ID 
    _min: min value for normalization 
    _max: max value for normalization 
    patchSize: patch size to be resampled to (x-y plane)
    depth: Number of patches
    clipValue: clipping values in the volume. i.e arr[arr > clipValue] = clipValue 
    """

    detName = [] 
    detLabel = []

    cc = sitk.ConnectedComponent(lm)
    relabelled = sitk.RelabelComponent(cc)

    arr = sitk.GetArrayFromImage(relabelled)

    if np.where(arr==1)[0].size > 2*np.where(arr==2)[0].size:
        arr[arr > 1] = 0
    else:
        arr[arr > 2] = 0 
        arr[arr !=0 ] = 1 

    props = regionprops(arr)
    startz, starty, startx, endz, endy, endx = props[0].bbox 

    starty = starty - 10
    endy = endy + 10 

    imgarr = normalizeImage(img,_min,_max,clipValue)
    lsarr = sitk.GetArrayFromImage(ls)
    lmarr = sitk.GetArrayFromImage(lm)

    rps = regionprops(lmarr)
    startz, starty, startx, endz, endy, endx = rps[0].bbox

    sample = imgarr[startz:endz, starty:endy, startx:endx]
    mask  = lsarr[startz:endz, starty:endy, startx:endx]


    sample = sitk.GetArrayFromImage(DataUtil.resampleimagebysize(sitk.GetImageFromArray(sample),(patchSize[1],patchSize[0],depth),sitk.sitkLinear))
    mask = sitk.GetArrayFromImage(DataUtil.resampleimagebysize(sitk.GetImageFromArray(mask),(patchSize[1],patchSize[0],depth),sitk.sitkNearestNeighbor))

    _addToHDF5(sample,mask,phase,splitspathname)


    return [name],[label]


def get_labels():
    pass 

    "Write code to obtain a dictionary with patient IDs or case IDs as keys and labels (0 or 1) as values "

    return labelsdict 

if __name__ == "__main__":

    labelsdict = get_labels()

    cvsplits = 3

    for cv in range(cvsplits):
        splitspathname = fr"<name of the splits file that contains case IDs and corresponding phases (train, test, val)>"
        inputfoldername = fr"2_Preprocessed"
        
        # input size to the network newsize2D as x-y plane and depth as number of patches 
        newsize2D = (192,192) 
        depth = 64
        
        # value to clip. Anything above clipValue will be clipped to clipValue (to remove artifacts) 
        clipValue = 4096

        splitspath = fr"outputs/splits/{splitspathname}.json"
        splitsdict = DataUtil.readJson(splitspath)

        cases = list(splitsdict.keys())
        values = [labelsdict[x] for x in cases]

        # create an empty hdf5
        createHDF5(splitspathname,newsize2D,depth)

        casenames = {} 
        casenames["train"] = [] 
        casenames["val"] = [] 
        casenames["test"] = [] 

        caselabels = {} 
        caselabels["train"] = [] 
        caselabels["val"] = [] 
        caselabels["test"] = [] 

        patwiseData = {}
        patwiseData["train"] = [] 
        patwiseData["val"] = [] 
        patwiseData["test"] = [] 


        # min and max value to normalize 
        _min = 0 
        _max = 4096

        if clipValue is not None:
            if _max > clipValue:
                _max = clipValue


        for j,name in enumerate(cases):

            dataset = name.split("_")[0]
            sb = Path(fr"../Data/{dataset}/{inputfoldername}/{name}")

            label = labelsdict[name]

            # number of augmentations to be performed.
            nosamples = 5

            print(name)
            phase = splitsdict[name]

            ret = None 

            if phase == "train":
                ret = getAugmentedData(sb,name,nosamples=nosamples)
            else:
                ret = getAugmentedData(sb,name,nosamples=None)

            for k,aug in enumerate(ret):
        
                augimg = aug[0][0]
                augls = aug[1][0]
                auglm = aug[1][1]
                            
                _img = augimg
                _ls = augls 
                _lm = auglm 

                casename = name if k == 0 else fr"{name}_A{k}"
                augmented = False if k == 0 else True 

                _casenames,_caselabels = addToHDF5(_img,_ls,_lm,phase,splitspathname,label,casename,_min,_max,newsize2D,depth,clipValue=clipValue)

                casenames[phase].extend(_casenames)
                caselabels[phase].extend(_caselabels)


        outputfolder = fr"outputs/hdf5/{splitspathname}"

        for phase in ["train","test","val"]:
            hdf5_file = tables.open_file(fr'{outputfolder}/{phase}.h5', mode='a')
            hdf5_file.create_array(hdf5_file.root, fr'names', casenames[phase])
            hdf5_file.create_array(hdf5_file.root, fr'labels', caselabels[phase])
            hdf5_file.close()

