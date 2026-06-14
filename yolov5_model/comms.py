import socket
import cv2
import json
import struct

MSG_VIDEO = 0x01
MSG_INIT = 0x03

class VideoSender:
    def __init__(self, ip, port, width=640, height=360, quality=50):
        self.ip = ip
        self.port = port
        self.width = width
        self.height = height
        self.quality = quality
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 增加UDP发送缓冲区大小，防止高码率丢包
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 2 * 1024 * 1024)
        self.MAX_UDP_SIZE = 60000 

    def send(self, board_id, cam_id, frame):
        try:
            if frame is None: return
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
                
            _, encoded = cv2.imencode('.jpg', frame,[cv2.IMWRITE_JPEG_QUALITY, self.quality])
            jpeg_data = encoded.tobytes()
            
            b_id_bytes = str(board_id).encode('utf-8')[:8].ljust(8, b'\x00')
            header = struct.pack('!BB8s', MSG_VIDEO, int(cam_id), b_id_bytes)
            packet = header + jpeg_data
            
            if len(packet) < self.MAX_UDP_SIZE:
                self.sock.sendto(packet, (self.ip, self.port))
        except Exception:
            pass


class DataSender:
    def __init__(self, targets, board_id):
        # targets 形如: {"gimbal": ("192.168.1.100", 8888), ...}
        self.targets = targets
        self.board_id = board_id
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_packet(self, msg_type, cam_id, content=None, target=None):
        if target not in self.targets:
            return
            
        ip, port = self.targets[target]
        payload = {
            "board": self.board_id,
            "cam": cam_id,
            "type": msg_type
        }
        
        if msg_type == "data": 
            payload["objs"] = content
        elif msg_type in ["status", "error"]: 
            payload["msg"] = content
            
        try:
            msg = json.dumps(payload).encode('utf-8')
            self.sock.sendto(msg, (ip, port))
        except Exception:
            pass
