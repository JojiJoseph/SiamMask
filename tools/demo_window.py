# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from tools.test import *
import pyautogui
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

STATE_PRESELECT = 0
STATE_TRACKING = 1
demo_state = STATE_PRESELECT
if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # Parse Image file
    #img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    #ims = [cv2.imread(imf) for imf in img_files]

    #cap = cv2.VideoCapture(0)
    #ret, img = cap.read()
    print(pyautogui.__dir__())
    img = np.asarray(pyautogui.getWindowsWithTitle("Penrose Museum (DEBUG)")[0].screenshot())[:,:,::-1].copy()
    img = cv2.resize(img, (1280,360))

    # Select ROI
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('SiamMask', img, False, False)
        x, y, w, h = init_rect
    except:
        exit()

    toc = 0
    f = 0 # frame
    while True:
        tic = cv2.getTickCount()
        #ret, img = cap.read()
        img = np.asarray(pyautogui.screenshot())[:,:,::-1].copy()
        img = cv2.resize(img, (1280,360))
        print(img.shape, img.dtype)
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(img, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        elif f > 0:  # tracking
            state = siamese_track(state, img, mask_enable=True, refine_enable=True, device=device)  # track
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr

            img[:, :, 2] = (mask > 0) * 255 + (mask == 0) * img[:, :, 2]
            print("$",img.shape, img.dtype)
            cv2.polylines(img, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            cv2.imshow('SiamMask', img)
            key = cv2.waitKey(1)
            if key > 0:
                break

        toc += cv2.getTickCount() - tic
        f += 1
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
