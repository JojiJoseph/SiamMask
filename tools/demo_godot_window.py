# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from tools.test import *
import socket

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
parser.add_argument('--port', type=int, default=3333)
args = parser.parse_args()

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server_address = ("127.0.0.1", args.port)
sock.bind(server_address)

sock.listen()

print("Waiting for godot program to connect!")

def godot_read():
    conn, addr = sock.accept()
    data = b""
    while True:
        new_data = conn.recv(1024)
        data = data + new_data
        if not new_data: # Disconnected
            break
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return True, img


STATE_PRESELECT = 0 # Indicates ROI is not selected
STATE_TRACKING = 1 # ROI is selected, now the tracker is tracking
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

    ret, img = godot_read()

    def mouse_callback(event, x, y, flags, params):
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
                
        
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    cv2.setMouseCallback("SiamMask", mouse_callback)


    # Select ROI
    while demo_state == STATE_PRESELECT:
        ret, img = godot_read()
        img_out = img.copy()
        if roi_state == ROI_START or roi_state == ROI_FINISH:
            cv2.rectangle(img_out, roi_p1, roi_p2, (255, 0, 0), 4)
        cv2.imshow("SiamMask", img_out)
        if cv2.waitKey(1)&0xFF == ord(' ') and roi_state == ROI_FINISH:
            x, y = roi_p1
            w = roi_p2[0] - x
            h = roi_p2[1] - y
            demo_state = STATE_TRACKING
            break

    toc = 0
    frame_idx = 0
    while True:
        tic = cv2.getTickCount()
        ret, img = godot_read()
        if frame_idx == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(img, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        elif frame_idx > 0:  # tracking
            state = siamese_track(state, img, mask_enable=True, refine_enable=True, device=device)  # track
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr

            img[:, :, 2] = (mask > 0) * 255 + (mask == 0) * img[:, :, 2]
            cv2.polylines(img, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            cv2.imshow('SiamMask', img)
            key = cv2.waitKey(1)
            if key > 0:
                break

        toc += cv2.getTickCount() - tic
        frame_idx += 1
    toc /= cv2.getTickFrequency()
    fps = frame_idx / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
