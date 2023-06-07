# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from tools.test import *
import os

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
parser.add_argument('--video', default=None)
args = parser.parse_args()

assert isfile(args.video), "Please specify the video"

STATE_PRESELECT = 0
STATE_TRACKING = 1
demo_state = STATE_PRESELECT

ROI_IDLE = 0
ROI_START = 1
ROI_FINISH = 2

roi_p1 = None
roi_p2 = None

roi_state = ROI_IDLE

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
    cap = cv2.VideoCapture(args.video)
    frames = []
    frame_idx = 0
    while True:
        ret, img = cap.read()
        if ret:
            frames.append(img)
        else:
            break
            cap.release()
    img = frames[frame_idx]
    img = cv2.flip(img, 1)

    def cb(event, x, y, flags, params):
        global roi_state, roi_p1, roi_p2
        if roi_state == ROI_IDLE:
            if event == cv2.EVENT_LBUTTONDOWN:
                roi_state = ROI_START
                roi_p1 = x, y
                roi_p2 = x, y
        elif roi_state == ROI_START:
            if event == cv2.EVENT_MOUSEMOVE:
                roi_p2 = x, y
            if event == cv2.EVENT_LBUTTONUP:
                roi_state = ROI_FINISH
                
    def update_frame_idx(val):
        global frame_idx
        frame_idx = val
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    cv2.createTrackbar("frame", "SiamMask", 0, len(frames)-1,update_frame_idx)
    cv2.setMouseCallback("SiamMask", cb)

    paused = True
    # Select ROI
    # while demo_state == STATE_PRESELECT:
    #     #if not paused:
    #         #ret, img = cap.read()
    #     img = frames[frame_idx]
    #     img = cv2.flip(img, 1)
    #     img_out = img.copy()
    #     if roi_state == ROI_START or roi_state == ROI_FINISH:
    #         cv2.rectangle(img_out, roi_p1, roi_p2, (255, 0, 0), 4)
    #     if paused:
    #         cv2.putText(img_out, "Paused!", [40, 40],cv2.FONT_HERSHEY_PLAIN,2,(0,255,0))
    #     cv2.imshow("SiamMask", img_out)
    #     key = cv2.waitKey(100)&0xFF
    #     if key == ord(' ') and roi_state == ROI_FINISH:
    #         x, y = roi_p1
    #         w = roi_p2[0] - x
    #         h = roi_p2[1] - y
    #         demo_state = STATE_TRACKING
    #         break
    #     if key == ord('p'):
    #         paused = not paused
    #     if not paused:
    #         frame_idx += 1
    #         if frame_idx == len(frames):
    #             frame_idx -= 1
    #         cv2.setTrackbarPos("frame", "SiamMask", frame_idx)
    #cv2.destroyWindow("SiamMask")
    
    #cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # try:
    #     init_rect = cv2.selectROI('SiamMask', img, False, False)
    #     x, y, w, h = init_rect
    # except:
    #     exit()
    #cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    #cv2.createTrackbar("frame", "SiamMask", 0, len(frames)-1,update_frame_idx)
    toc = 0
    f = 0 # frame
    paused = False
    while True:
        tic = cv2.getTickCount()
        #if not paused:
            #ret, img = cap.read()
        img = frames[frame_idx]
        img = cv2.flip(img, 1)
        img_out = img.copy()
        if demo_state == STATE_PRESELECT:
            if roi_state == ROI_START or roi_state == ROI_FINISH:
                cv2.rectangle(img_out, roi_p1, roi_p2, (255, 0, 0), 4)
            if paused:
                cv2.putText(img_out, "Paused!", [40, 40],cv2.FONT_HERSHEY_PLAIN,2,(0,255,0))
            cv2.imshow("SiamMask", img_out)
            key = cv2.waitKey(int(1000/30))&0xFF
            if key == ord(' ') and roi_state == ROI_FINISH:
                x, y = roi_p1
                w = roi_p2[0] - x
                h = roi_p2[1] - y
                demo_state = STATE_TRACKING
                f = -1
            if key == ord('p'):
                paused = not paused
            if not paused:
                frame_idx += 1
                if frame_idx == len(frames):
                    frame_idx -= 1
                cv2.setTrackbarPos("frame", "SiamMask", frame_idx)
        else:
            if f == 0:  # init
                target_pos = np.array([x + w / 2, y + h / 2])
                target_sz = np.array([w, h])
                cfg['hp']['windowing'] = 'uniform'
                state = siamese_init(img_out, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
            elif f > 0:  # tracking
                state = siamese_track(state, img_out, mask_enable=True, refine_enable=True, device=device)  # track
                # print(state.keys(), state['score'])
                # print(state['window'].shape)
                score = state['score']
                location = state['ploygon'].flatten()
                mask = state['mask'] > state['p'].seg_thr

                img_out[:, :, 2] = (mask > 0) * 255 + (mask == 0) * img_out[:, :, 2]
                cv2.polylines(img_out, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                target_pos = state['target_pos']
                target_sz = state['target_sz']
                p1 = target_pos - target_sz/2
                p2 = target_pos + target_sz/2
                cv2.putText(img_out, str(int(score*100)), np.int0(target_pos),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0))
                cv2.rectangle(img_out, np.int0(p1), np.int0(p2), (255,0,0), 3)
                if 'crop_box' in state:
                    crop_box = state['crop_box'] # x1, y1, w, h
                    cv2.rectangle(img_out, (crop_box[0], crop_box[1]),
                        (crop_box[0] + crop_box[2], crop_box[1] + crop_box[3]), (0, 0, 255), 5)
                if paused:
                    cv2.putText(img_out, "Paused!", [40, 40],cv2.FONT_HERSHEY_PLAIN,2,(0,255,0))
                cv2.imshow('SiamMask', img_out)
                key = cv2.waitKey(int(1000/30))
                if key == ord('p'):
                    paused = not paused
                if key  == ord('q'):
                    break
                if key == ord('r'):
                    demo_state = STATE_PRESELECT
                    roi_state = ROI_IDLE
                if not paused:
                    frame_idx += 1
                    if frame_idx == len(frames):
                        frame_idx -= 1
                    cv2.setTrackbarPos("frame", "SiamMask", frame_idx)

        toc += cv2.getTickCount() - tic
        f += 1
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
