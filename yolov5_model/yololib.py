import os

import cv2
import numpy as np
from rknnlite.api import RKNNLite


class YoloRKNN:
    def __init__(self, model_path, input_size=(640, 640), conf_thres=0.45, nms_thres=0.45):
        self.input_size = input_size
        self.conf_thres = float(conf_thres)
        self.nms_thres = float(nms_thres)
        self.core_mask = RKNNLite.NPU_CORE_0_1_2
        self.debug = os.environ.get("YOLO_RKNN_DEBUG", "0") == "1"
        self.keep_bgr = os.environ.get("YOLO_RKNN_KEEP_BGR", "0") == "1"
        self.float_input = os.environ.get("YOLO_RKNN_FLOAT_INPUT", "0") == "1"
        self._printed_debug = False

        # YOLOv5 default anchors, grouped by stride 8/16/32.
        self.anchors = [
            [[10, 13], [16, 30], [33, 23]],
            [[30, 61], [62, 45], [59, 119]],
            [[116, 90], [156, 198], [373, 326]],
        ]

        self.rknn = RKNNLite()
        print(f"--> [YoloLib] Loading model: {model_path}", flush=True)
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"Load RKNN failed: {ret}")

        ret = self.rknn.init_runtime(core_mask=self.core_mask)
        if ret != 0:
            raise RuntimeError(f"Init RKNN runtime failed: {ret}")
        print("--> [YoloLib] RKNN Init Done.", flush=True)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))

    @staticmethod
    def _maybe_sigmoid(pred):
        if pred.size == 0:
            return pred
        pred_min = float(np.min(pred))
        pred_max = float(np.max(pred))
        if pred_min < -1e-3 or pred_max > 1.0 + 1e-3:
            return YoloRKNN.sigmoid(pred)
        return pred

    def _input_h(self):
        if isinstance(self.input_size, (tuple, list)) and len(self.input_size) >= 2:
            return int(self.input_size[1])
        return int(self.input_size)

    def _anchor_index(self, stride):
        if stride == 8:
            return 0
        if stride == 16:
            return 1
        if stride == 32:
            return 2
        return None

    def _branch_pred(self, output):
        output = np.asarray(output)

        # RKNN/ONNX may expose YOLO heads in several equivalent layouts.
        if output.ndim == 5 and output.shape[1] == 3 and output.shape[-1] == 6:
            _, _, h, w, _ = output.shape
            return output, h, w
        if output.ndim == 5 and output.shape[-2] == 3 and output.shape[-1] == 6:
            _, h, w, _, _ = output.shape
            return output.transpose(0, 3, 1, 2, 4), h, w
        if output.ndim == 5 and output.shape[1] == 3 and output.shape[2] == 6:
            _, _, _, h, w = output.shape
            return output.transpose(0, 1, 3, 4, 2), h, w

        if output.ndim == 4 and output.shape[1] == 18:
            _, _, h, w = output.shape
            pred = output.reshape(1, 3, 6, h, w).transpose(0, 1, 3, 4, 2)
            return pred, h, w
        if output.ndim == 4 and output.shape[-1] == 18:
            _, h, w, _ = output.shape
            pred = output.reshape(1, h, w, 3, 6).transpose(0, 3, 1, 2, 4)
            return pred, h, w

        if output.ndim == 3 and output.shape[-1] == 6:
            n = int(output.shape[1])
            h = w = int(np.sqrt(n / 3))
            if h * w * 3 == n:
                return output.reshape(1, 3, h, w, 6), h, w

        return None

    def _output_info(self, output):
        output = np.asarray(output)
        if output.ndim in (2, 3) and output.shape[-1] == 6:
            n = int(np.prod(output.shape[:-1]))
            if n == 25200:
                return ("decoded", None, None)

        branch = self._branch_pred(output)
        if branch is None:
            return None
        _, h, _ = branch
        stride = int(round(self._input_h() / float(h)))
        aidx = self._anchor_index(stride)
        if aidx is None:
            return None
        return ("branch", stride, aidx)

    def _decode_branch(self, output, anchors, stride):
        branch = self._branch_pred(output)
        if branch is None:
            return np.empty((0, 6), dtype=np.float32)

        pred, h, w = branch
        pred = self._maybe_sigmoid(pred)

        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack((grid_x, grid_y), axis=-1).reshape(1, 1, h, w, 2)

        xy = (pred[..., 0:2] * 2.0 - 0.5 + grid) * float(stride)
        anchors_t = np.array(anchors, dtype=np.float32).reshape(1, 3, 1, 1, 2)
        wh = (pred[..., 2:4] * 2.0) ** 2 * anchors_t

        out = np.concatenate((xy, wh, pred[..., 4:]), axis=-1)
        return out.reshape(-1, 6)

    def _post_process(self, outputs):
        if outputs is None:
            return []

        all_boxes = []
        for out in outputs:
            info = self._output_info(out)
            if info is None:
                continue

            kind, stride, aidx = info
            if kind == "decoded":
                decoded = np.asarray(out).reshape(-1, 6)
            else:
                decoded = self._decode_branch(out, self.anchors[aidx], stride)

            if decoded.size == 0:
                continue

            obj_conf = decoded[:, 4]
            cls_conf = decoded[:, 5]
            scores = obj_conf * cls_conf
            keep = scores > self.conf_thres
            if not np.any(keep):
                continue

            valid_boxes = decoded[keep]
            valid_scores = scores[keep]
            x = valid_boxes[:, 0]
            y = valid_boxes[:, 1]
            bw = valid_boxes[:, 2]
            bh = valid_boxes[:, 3]

            x1 = x - bw / 2.0
            y1 = y - bh / 2.0
            x2 = x + bw / 2.0
            y2 = y + bh / 2.0

            for i in range(len(valid_scores)):
                all_boxes.append([x1[i], y1[i], x2[i], y2[i], valid_scores[i], 0])

        if not all_boxes:
            return []

        all_boxes = np.array(all_boxes)
        nms_boxes = []
        for x1, y1, x2, y2 in all_boxes[:, :4]:
            nms_boxes.append([float(x1), float(y1), float(max(1.0, x2 - x1)), float(max(1.0, y2 - y1))])

        scores = all_boxes[:, 4].tolist()
        indices = cv2.dnn.NMSBoxes(nms_boxes, scores, self.conf_thres, self.nms_thres)
        if len(indices) == 0:
            return []

        res = []
        for i in np.array(indices).flatten():
            box = all_boxes[int(i)]
            res.append([float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4]), 0])
        return np.array(res)

    def _debug_outputs(self, input_data, outputs, dets):
        if not self.debug or self._printed_debug:
            return
        self._printed_debug = True
        print(
            f"--> [YoloLib][debug] input shape={input_data.shape} dtype={input_data.dtype} "
            f"min={float(np.min(input_data)):.3f} max={float(np.max(input_data)):.3f} "
            f"keep_bgr={self.keep_bgr} float_input={self.float_input}",
            flush=True,
        )
        for idx, out in enumerate(outputs if outputs is not None else []):
            arr = np.asarray(out)
            info = self._output_info(arr)
            print(
                f"--> [YoloLib][debug] out{idx} shape={arr.shape} dtype={arr.dtype} "
                f"min={float(np.min(arr)):.6f} max={float(np.max(arr)):.6f} "
                f"mean={float(np.mean(arr)):.6f} info={info}",
                flush=True,
            )
        det_count = 0 if dets is None else len(dets)
        top = []
        if det_count:
            top = sorted(dets.tolist() if hasattr(dets, "tolist") else dets, key=lambda d: d[4], reverse=True)[:5]
        print(f"--> [YoloLib][debug] det_count={det_count} top={top}", flush=True)

    def infer(self, img_bgr):
        if img_bgr is None or img_bgr.size == 0:
            return []

        if img_bgr.shape[:2] != self.input_size:
            img_in = cv2.resize(img_bgr, self.input_size)
        else:
            img_in = img_bgr

        if not self.keep_bgr:
            img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
        if self.float_input:
            img_in = img_in.astype(np.float32) / 255.0

        input_data = np.expand_dims(img_in, axis=0)
        outputs = self.rknn.inference(inputs=[input_data])
        dets = self._post_process(outputs)
        self._debug_outputs(input_data, outputs, dets)
        return dets

    def release(self):
        self.rknn.release()
